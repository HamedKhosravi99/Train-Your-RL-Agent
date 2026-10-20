"""Is 100% the whole story? Test a trained shop agent on questions it never practiced on.

    python scripts/evaluate_generalization.py --adapters run1.pt run2.pt run3.pt --out results.json

Training only ever sees the question templates, products and sizes in `train_tool_agent.py`.
This script builds harder test sets that differ from training in one way each:

    in_distribution  new numbers and item combinations, same templates as training
    new_phrasing     the same kinds of tasks, worded in ways never seen in training
    new_products     items the shop only sells at test time, so the price must be looked up
    three_items      three-item orders (training has at most two)
    bigger_numbers   4-digit word problems and 3-digit quantities

It also measures two baselines without any RL: the base model as used in training (zero-shot),
and the base model with one worked example added to its instructions (few-shot prompting).
"""
import argparse
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_tool_agent as agent  # noqa: E402

N_PER_SPLIT = 64

NEW_PRODUCTS = {"binder": 4.15, "protractor": 2.85, "sharpener": 0.65, "crayon pack": 3.20,
                "tape roll": 1.90, "glue bottle": 2.45, "index card": 0.30, "sketchbook": 6.40}

NEW_SHOP_TEMPLATES = {
    "shop1": ["What is the total cost of {n1} {i1}s?", "I need {n1} {i1}s. What will I pay?",
              "Price check, please: {n1} {i1}s."],
    "shop2": ["What do {n1} {i1}s plus {n2} {i2}s cost altogether?",
              "Please ring up {n1} {i1}s and {n2} {i2}s. What is the total?"],
    "budget": ["My budget is {budget} dollars. After buying {n1} {i1}s, how much remains?",
               "With {budget} dollars in my wallet, I pay for {n1} {i1}s. What is left over?"],
}
NEW_WORD_PROBLEMS = [
    ("A warehouse stores {a} crates on each of its {b} floors. How many crates are there?", lambda a, b, c: a * b),
    ("Each of {b} classrooms has {a} chairs, and {c} chairs are broken. How many chairs still work?",
     lambda a, b, c: a * b - c),
    ("A runner covers {a} meters per lap for {b} laps, then jogs {c} meters more. How far does she go?",
     lambda a, b, c: a * b + c),
]

FEW_SHOT_EXAMPLE = (
    "\n\nHere is an example.\nQuestion: How much do 3 pencils and 2 rulers cost in total?\n"
    "<lookup>pencil</lookup>\n<result>0.45</result>\n<lookup>ruler</lookup>\n<result>2.30</result>\n"
    "<calculator>3 * 0.45 + 2 * 2.30</calculator>\n<result>5.95</result>\n<answer>5.95</answer>"
)


def shop_task(rng, kind, catalog, template=None, qty=(12, 99)):
    i1, i2 = rng.sample(sorted(catalog), 2)
    n1, n2 = rng.randint(*qty), rng.randint(*qty)
    c1, c2 = n1 * catalog[i1], n2 * catalog[i2]
    budget = rng.randint(int(c1) + 5, int(c1) + 300)
    answer = {"shop1": c1, "shop2": c1 + c2, "budget": budget - c1}[kind]
    if template is None:   # the training wording
        template = {"shop1": "How much do {n1} {i1}s cost?",
                    "shop2": "How much do {n1} {i1}s and {n2} {i2}s cost in total?",
                    "budget": "I have {budget} dollars and buy {n1} {i1}s. How much money do I have left?"}[kind]
    q = template.format(n1=n1, n2=n2, i1=i1, i2=i2, budget=budget)
    return {"question": q, "answer": round(answer, 2), "kind": kind}


def word_problem(rng, templates, a_range=(101, 999), b_range=(12, 99)):
    text, solve = rng.choice(templates)
    a, b = rng.randint(*a_range), rng.randint(*b_range)
    c = rng.randint(100, a * b - 1)
    return {"question": text.format(a=a, b=b, c=c), "answer": solve(a, b, c), "kind": "math"}


def build_splits():
    def make(seed_base, fn):
        return [fn(random.Random(seed_base + i), i) for i in range(N_PER_SPLIT)]

    kinds = ("math", "shop1", "shop2", "budget")

    def new_phrasing(rng, i):
        kind = kinds[i % 4]
        if kind == "math":
            return word_problem(rng, NEW_WORD_PROBLEMS)
        return shop_task(rng, kind, agent.PRICES, rng.choice(NEW_SHOP_TEMPLATES[kind]))

    def new_products(rng, i):
        return shop_task(rng, ("shop1", "shop2", "budget")[i % 3], NEW_PRODUCTS)

    def three_items(rng, i):
        i1, i2, i3 = rng.sample(sorted(agent.PRICES), 3)
        n = [rng.randint(12, 99) for _ in range(3)]
        total = sum(k * agent.PRICES[it] for k, it in zip(n, (i1, i2, i3)))
        return {"question": f"How much do {n[0]} {i1}s, {n[1]} {i2}s and {n[2]} {i3}s cost in total?",
                "answer": round(total, 2), "kind": "shop3"}

    def bigger_numbers(rng, i):
        if i % 2 == 0:
            return word_problem(rng, agent.WORD_PROBLEMS, a_range=(1001, 9999), b_range=(101, 999))
        return shop_task(rng, ("shop1", "shop2")[i % 4 // 2], agent.PRICES, qty=(100, 999))

    return {
        "in_distribution": [agent.make_task(random.Random(10_000 + i)) for i in range(N_PER_SPLIT)],
        "new_phrasing": make(20_000, new_phrasing),
        "new_products": make(30_000, new_products),
        "three_items": make(40_000, three_items),
        "bigger_numbers": make(50_000, bigger_numbers),
    }


@contextmanager
def shop_sells(extra):
    """Temporarily add products to the shop (the lookup tool reads agent.PRICES)."""
    agent.PRICES.update(extra)
    try:
        yield
    finally:
        for k in extra:
            agent.PRICES.pop(k)


@contextmanager
def instructions(prompt):
    old = agent.SYSTEM_PROMPT
    agent.SYSTEM_PROMPT = prompt
    try:
        yield
    finally:
        agent.SYSTEM_PROMPT = old


def run_splits(model, tokenizer, splits):
    scores = {}
    for name, tasks in splits.items():
        with shop_sells(NEW_PRODUCTS if name == "new_products" else {}):
            result = agent.evaluate(model, tokenizer, tasks)
        scores[name] = result["accuracy"]
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapters", nargs="*", default=[], help="LoRA files saved by train_tool_agent.py")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--out", default="generalization_results.json")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = agent.pick_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    agent.add_lora(model, rank=args.lora_rank, alpha=2 * args.lora_rank)
    splits = build_splits()
    rows = {}

    agent.set_lora(model, False)                       # the untouched base model
    start = time.time()
    rows["base, zero-shot"] = run_splits(model, tokenizer, splits)
    with instructions(agent.SYSTEM_PROMPT + FEW_SHOT_EXAMPLE):
        rows["base, one worked example"] = run_splits(model, tokenizer, splits)
    agent.set_lora(model, True)
    for path in args.adapters:
        state = torch.load(path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and len(state) > 0, "not a LoRA file from train_tool_agent.py"
        rows[f"GRPO: {Path(path).stem}"] = run_splits(model, tokenizer, splits)
        print(f"evaluated {path} ({time.time() - start:.0f}s so far)", flush=True)

    names = list(splits)
    print("\n| model | " + " | ".join(names) + " |")
    print("|:--|" + ":-:|" * len(names))
    for row, scores in rows.items():
        print(f"| {row} | " + " | ".join(f"{scores[n]:.0%}" for n in names) + " |")
    json.dump({"n_per_split": N_PER_SPLIT, "splits": names, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
