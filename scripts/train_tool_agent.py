"""TrainYourAgent, Chapter 4 as a script: teach a small LLM to use tools with GRPO.

A shop assistant gets two tools, a price `lookup` (the prices are made up, so the model cannot
know them) and a `calculator`. Reward = 1 only if the final answer is exactly right.

    python scripts/train_tool_agent.py                      # Qwen2.5-0.5B-Instruct, 25 GRPO steps
    python scripts/train_tool_agent.py --steps 3 --eval-size 8 --questions-per-step 2   # smoke test

Everything is written from scratch in plain PyTorch + Hugging Face `transformers`: multi-turn
rollouts with real tool calls, loss masking of tool output, group-relative advantages, the clipped
GRPO loss, an optional KL leash, and LoRA. No TRL, no verl, no peft.

The notebook `notebooks/04_llm_tool_use_grpo.ipynb` walks through the same functions step by step.
"""
import argparse
import ast
import json
import math
import operator
import random
import re
import time

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------------
# 1. The tasks: questions that are easy with the tools and (nearly) impossible without them.
# ----------------------------------------------------------------------------------------------
PRICES = {"notebook": 3.75, "pencil": 0.45, "backpack": 27.90, "eraser": 0.85, "marker": 1.65,
          "ruler": 2.30, "stapler": 8.40, "folder": 0.95, "highlighter": 1.35, "glue stick": 1.15,
          "paint set": 9.80, "sticker pack": 2.05}

WORD_PROBLEMS = [
    ("A farmer plants {a} rows of trees with {b} trees in each row. "
     "How many trees does the farmer plant?", lambda a, b, c: a * b),
    ("A factory makes {a} toys every day. How many toys does it make in {b} days?",
     lambda a, b, c: a * b),
    ("A school buys {a} boxes of pencils. Each box holds {b} pencils. "
     "The school gives away {c} pencils. How many pencils are left?", lambda a, b, c: a * b - c),
    ("A train travels {a} km every day for {b} days, and then {c} km more. "
     "How far does it travel in total?", lambda a, b, c: a * b + c),
    ("A library has {a} shelves with {b} books on each shelf, plus {c} books in storage. "
     "How many books does it have?", lambda a, b, c: a * b + c),
    ("A bakery sells {a} cakes for {b} dollars each and spends {c} dollars on flour. "
     "How much money is left?", lambda a, b, c: a * b - c),
]

TASK_KINDS = ("math", "shop1", "shop2", "budget")


def make_task(rng, kind=None):
    """Return {"question", "answer", "kind"}. Kinds:
    math   - word problem, needs the calculator          ("A farmer plants 347 rows of ...")
    shop1  - one price lookup + calculator               ("How much do 37 notebooks cost?")
    shop2  - two price lookups + calculator              ("... 37 notebooks and 12 rulers ...")
    budget - lookup + calculator, then subtract          ("I have 300 dollars and buy ...")
    """
    kind = kind or rng.choice(TASK_KINDS)
    if kind == "math":
        text, solve = rng.choice(WORD_PROBLEMS)
        a, b = rng.randint(101, 999), rng.randint(12, 99)
        c = rng.randint(100, a * b - 1)
        return {"question": text.format(a=a, b=b, c=c), "answer": solve(a, b, c), "kind": kind}
    item1, item2 = rng.sample(sorted(PRICES), 2)
    n1, n2 = rng.randint(12, 99), rng.randint(12, 99)
    cost1, cost2 = n1 * PRICES[item1], n2 * PRICES[item2]
    if kind == "shop1":
        return {"question": f"How much do {n1} {item1}s cost?", "answer": round(cost1, 2), "kind": kind}
    if kind == "shop2":
        return {"question": f"How much do {n1} {item1}s and {n2} {item2}s cost in total?",
                "answer": round(cost1 + cost2, 2), "kind": kind}
    budget = rng.randint(int(cost1) + 5, int(cost1) + 300)
    return {"question": f"I have {budget} dollars and buy {n1} {item1}s. How much money do I have left?",
            "answer": round(budget - cost1, 2), "kind": kind}


# ----------------------------------------------------------------------------------------------
# 2. The tools: a safe calculator (no `eval`!), a price lookup, and the instructions for the model.
# ----------------------------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a shop assistant with two tools.\n"
    "- To find the price of an item in dollars, write the item name inside <lookup></lookup> tags, "
    "for example <lookup>pencil</lookup>, and then stop.\n"
    "- To calculate, write an arithmetic expression inside <calculator></calculator> tags, "
    "for example <calculator>12 * (3 + 4)</calculator>, and then stop.\n"
    "Each tool result is returned to you inside <result></result> tags. You can use the tools several times.\n"
    "When you know the final answer, write the number inside <answer></answer> tags, "
    "for example <answer>84</answer>."
)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression):
    """Evaluate + - * / // % and parentheses on numbers. Anything else returns "error"."""
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.operand))
        raise ValueError("unsupported expression")

    try:
        cleaned = " ".join(expression.replace(",", "").replace("×", "*").replace("÷", "/").split())
        value = evaluate(ast.parse(cleaned, mode="eval").body)
        return str(int(value)) if float(value).is_integer() else str(round(value, 4))
    except Exception:
        return "error"


def lookup(item):
    """Price of an item in dollars ("notebooks" and "Notebook" both work)."""
    name = " ".join(item.lower().strip(" .?!").split())
    for candidate in (name, name[:-1] if name.endswith("s") else name):
        if candidate in PRICES:
            return f"{PRICES[candidate]:.2f}"
    return "unknown item"


TOOLS = {"calculator": calculator, "lookup": lookup}
TOOL_CALL = re.compile(r"<(calculator|lookup)>(.*?)</\1>", re.S)
STOP_STRINGS = ["</calculator>", "</lookup>", "</answer>"]


# ----------------------------------------------------------------------------------------------
# 3. The reward: 1 if the final <answer> is the right number, else 0. Nothing for "using tools".
# ----------------------------------------------------------------------------------------------
def parse_answer(text):
    found = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def reward_fn(text, answer):
    got = parse_answer(text)
    return 1.0 if got is not None and abs(got - answer) < 0.01 else 0.0


# ----------------------------------------------------------------------------------------------
# 4. LoRA from scratch: freeze the model, learn a small low-rank "patch" on each linear layer.
#    Turning the patches off gives back the original model, our free KL reference.
# ----------------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / rank, True
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # B = 0, so the patch starts as a no-op

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            out = out + ((x.to(self.A.dtype) @ self.A.T) @ self.B.T * self.scale).to(out.dtype)
        return out


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def add_lora(model, rank=16, alpha=32):
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in LORA_TARGETS and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))
    return [p for p in model.parameters() if p.requires_grad]


def set_lora(model, enabled):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


# ----------------------------------------------------------------------------------------------
# 5. Rollouts: generate -> if the model called a tool, run it, paste the result back, and let the
#    model continue. Tool tokens get loss_mask = 0 (the model didn't write them).
# ----------------------------------------------------------------------------------------------
def build_prompt(tokenizer, question):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False).input_ids


def left_pad(sequences, pad_id):
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, width - len(seq):] = torch.tensor(seq)
        attention[row, width - len(seq):] = 1
    return ids, attention


@torch.no_grad()
def rollout(model, tokenizer, questions, temperature=0.8, greedy=False,
            max_tool_calls=4, max_new_tokens=128):
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id
    prompt_ids = [build_prompt(tokenizer, q) for q in questions]
    completion_ids = [[] for _ in questions]
    loss_mask = [[] for _ in questions]
    tool_calls = [[] for _ in questions]   # names of the tools each episode used
    sampling = (dict(do_sample=False) if greedy else
                dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0))

    active = list(range(len(questions)))
    for turn in range(max_tool_calls + 1):
        if not active:
            break
        ids, attention = left_pad([prompt_ids[i] + completion_ids[i] for i in active], pad)
        output = model.generate(
            input_ids=ids.to(device), attention_mask=attention.to(device),
            max_new_tokens=max_new_tokens, stop_strings=STOP_STRINGS,
            tokenizer=tokenizer, pad_token_id=pad, repetition_penalty=1.0, **sampling)
        still_active = []
        for row, i in enumerate(active):
            new_tokens = output[row, ids.shape[1]:].tolist()
            while new_tokens and new_tokens[-1] == pad:      # drop padding after a finished row
                new_tokens.pop()
            completion_ids[i] += new_tokens
            loss_mask[i] += [1] * len(new_tokens)          # model-written tokens: train on them
            text = tokenizer.decode(new_tokens)
            calls = TOOL_CALL.findall(text)
            if calls and "</answer>" not in text and turn < max_tool_calls:
                name, argument = calls[-1]
                result = TOOLS[name](argument)
                tool_tokens = tokenizer(f"\n<result>{result}</result>\n",
                                        add_special_tokens=False).input_ids
                completion_ids[i] += tool_tokens
                loss_mask[i] += [0] * len(tool_tokens)     # tool-written tokens: don't train
                tool_calls[i].append(name)
                still_active.append(i)
        active = still_active

    return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "loss_mask": loss_mask,
            "texts": [tokenizer.decode(c) for c in completion_ids], "tool_calls": tool_calls}


# ----------------------------------------------------------------------------------------------
# 6. GRPO: group-relative advantages + clipped, masked policy-gradient loss (+ optional KL).
# ----------------------------------------------------------------------------------------------
def group_advantages(rewards, group_size):
    r = torch.tensor(rewards, dtype=torch.float32).view(-1, group_size)
    return ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)


def right_pad(prompt_ids, completion_ids, loss_mask, pad_id):
    """Stack prompt+completion into one batch. mask[t] = 1 where token t should be trained."""
    seqs = [p + c for p, c in zip(prompt_ids, completion_ids)]
    masks = [[0] * len(p) + m for p, m in zip(prompt_ids, loss_mask)]
    width = max(len(s) for s in seqs)
    ids = torch.tensor([s + [pad_id] * (width - len(s)) for s in seqs])
    attention = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
    mask = torch.tensor([m + [0] * (width - len(m)) for m in masks], dtype=torch.float32)
    return ids, attention, mask


def token_logprobs(model, ids, attention, temperature):
    """log pi(token_t | tokens_<t) for every position t >= 1 (shape: batch x (len-1))."""
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1]
    logits = logits.float() / temperature
    return torch.gather(logits.log_softmax(-1), 2, ids[:, 1:, None]).squeeze(-1)


def grpo_loss(logp, old_logp, advantages, mask, n_tokens, clip=0.2, ref_logp=None, beta=0.0):
    ratio = torch.exp(logp - old_logp)
    A = advantages[:, None]
    per_token = -torch.min(ratio * A, ratio.clamp(1 - clip, 1 + clip) * A)
    if ref_logp is not None and beta > 0:
        diff = ref_logp - logp
        per_token = per_token + beta * (torch.exp(diff) - diff - 1)
    return (per_token * mask).sum() / n_tokens   # average over every trained token in the batch


def free_cached_memory(device):
    """Apple-silicon GPUs keep freed memory cached; hand it back between generation and training."""
    if device.type == "mps":
        torch.mps.empty_cache()


def grpo_step(model, tokenizer, optimizer, params, tasks, group_size, temperature,
              micro_batch=4, clip=0.2, beta=0.0, tool_cost=0.0, max_grad_norm=1.0):
    device = next(model.parameters()).device
    questions = [t["question"] for t in tasks for _ in range(group_size)]
    answers = [t["answer"] for t in tasks for _ in range(group_size)]

    model.eval()
    batch = rollout(model, tokenizer, questions, temperature=temperature)
    free_cached_memory(device)
    rewards = [reward_fn(text, a) - tool_cost * len(calls)       # optional price per tool call
               for text, a, calls in zip(batch["texts"], answers, batch["tool_calls"])]
    advantages = group_advantages(rewards, group_size)
    keep = [i for i in range(len(questions)) if advantages[i].abs() > 1e-6]  # skip no-signal groups

    model.train()
    n_tokens = max(1, sum(sum(batch["loss_mask"][i]) for i in keep))
    for start in range(0, len(keep), micro_batch):
        idx = keep[start:start + micro_batch]
        ids, attention, mask = right_pad([batch["prompt_ids"][i] for i in idx],
                                         [batch["completion_ids"][i] for i in idx],
                                         [batch["loss_mask"][i] for i in idx], tokenizer.pad_token_id)
        ids, attention, mask = ids.to(device), attention.to(device), mask[:, 1:].to(device)
        ref_logp = None
        if beta > 0:
            with torch.no_grad():
                set_lora(model, False)
                ref_logp = token_logprobs(model, ids, attention, temperature)
                set_lora(model, True)
        logp = token_logprobs(model, ids, attention, temperature)
        # One update per batch, so the "old" policy is the current one (ratio = 1 at this point).
        loss = grpo_loss(logp, logp.detach(), advantages[idx].to(device), mask, n_tokens,
                         clip, ref_logp, beta)
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item() if keep else 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    free_cached_memory(device)
    return {"reward": sum(rewards) / len(rewards),
            "lookup_use": sum("lookup" in c for c in batch["tool_calls"]) / len(questions),
            "calculator_use": sum("calculator" in c for c in batch["tool_calls"]) / len(questions),
            "trained_on": len(keep), "grad_norm": grad_norm, "example": batch["texts"][0]}


def evaluate(model, tokenizer, tasks):
    """Greedy decoding on held-out tasks: overall accuracy and accuracy per task kind."""
    model.eval()
    out = rollout(model, tokenizer, [t["question"] for t in tasks], greedy=True)
    rewards = [reward_fn(text, t["answer"]) for text, t in zip(out["texts"], tasks)]
    per_kind, calls_per_kind = {}, {}
    for kind in TASK_KINDS:
        rows = [(r, len(c)) for r, c, t in zip(rewards, out["tool_calls"], tasks) if t["kind"] == kind]
        if rows:
            per_kind[kind] = sum(r for r, _ in rows) / len(rows)
            calls_per_kind[kind] = sum(n for _, n in rows) / len(rows)
    return {"accuracy": sum(rewards) / len(rewards), "per_kind": per_kind,
            "calls_per_kind": calls_per_kind, "texts": out["texts"], "tool_calls": out["tool_calls"]}


# ----------------------------------------------------------------------------------------------
# 7. Main: load the model, measure, train, measure again.
# ----------------------------------------------------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        return "cuda", (torch.bfloat16 if bf16 else torch.float32)
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--questions-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=0.0, help="KL leash strength (0 = off)")
    parser.add_argument("--tool-cost", type=float, default=0.0, help="reward penalty per tool call (0 = off)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="tool_agent_lora.pt", help="where to save the LoRA weights")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, dtype = pick_device()
    print(f"device={device} dtype={dtype} model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = add_lora(model, rank=args.lora_rank, alpha=2 * args.lora_rank)
    print(f"trainable LoRA parameters: {sum(p.numel() for p in params) / 1e6:.1f}M "
          f"of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)

    train_rng = random.Random(args.seed)
    eval_tasks = [make_task(random.Random(10_000 + i)) for i in range(args.eval_size)]

    def report(name, result):
        kinds = " ".join(f"{k}={v:.0%}" for k, v in result["per_kind"].items())
        calls = " ".join(f"{k}={v:.1f}" for k, v in result["calls_per_kind"].items())
        print(f"{name}: accuracy={result['accuracy']:.0%} ({kinds}) | tool calls per question: {calls}")

    before = evaluate(model, tokenizer, eval_tasks)
    report("before training", before)

    history = []
    for step in range(1, args.steps + 1):
        start = time.time()
        tasks = [make_task(train_rng) for _ in range(args.questions_per_step)]
        stats = grpo_step(model, tokenizer, optimizer, params, tasks, args.group_size,
                          args.temperature, args.micro_batch, beta=args.beta, tool_cost=args.tool_cost)
        history.append({k: v for k, v in stats.items() if k != "example"})
        print(f"step {step:3d} | reward {stats['reward']:.2f} | lookup {stats['lookup_use']:.0%} | "
              f"calculator {stats['calculator_use']:.0%} | trained on {stats['trained_on']:2d} | "
              f"{time.time() - start:.0f}s", flush=True)

    after = evaluate(model, tokenizer, eval_tasks)
    print()
    report("before", before)
    report("after ", after)
    print(f"\nQ: {eval_tasks[0]['question']}  (answer {eval_tasks[0]['answer']})")
    print(f"--- before ---\n{before['texts'][0]}\n--- after ---\n{after['texts'][0]}")

    lora_state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(lora_state, args.save)
    with open(args.save.rsplit(".", 1)[0] + "_history.json", "w") as f:
        keys = ("accuracy", "per_kind", "calls_per_kind")
        json.dump({"before": {k: before[k] for k in keys}, "after": {k: after[k] for k in keys},
                   "history": history}, f)
    print(f"saved LoRA weights to {args.save}")


if __name__ == "__main__":
    main()
