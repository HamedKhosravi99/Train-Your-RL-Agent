"""Unit tests for the Chapter 4 building blocks. They run on CPU in a few seconds, no model download."""
import random
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_tool_agent as agent  # noqa: E402


# --- tools ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("expression, expected", [
    ("12 * (3 + 4)", "84"),
    ("37 * 3.75", "138.75"),
    ("1,000 * 3", "3000"),
    ("7 / 2", "3.5"),
    ("1 / 3", "0.3333"),
    ("-5 + 2", "-3"),
    ("624 * 74\n - 26637", "19539"),   # multi-line expressions are joined
    ("12 × 4", "48"),
])
def test_calculator_math(expression, expected):
    assert agent.calculator(expression) == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('ls')", "2 ** 1000000", "x = 3", "1 / 0", "", "open('f')", "True + 1",
])
def test_calculator_rejects_unsafe_or_invalid_input(expression):
    assert agent.calculator(expression) == "error"


@pytest.mark.parametrize("item, expected", [
    ("notebook", "3.75"), ("notebooks", "3.75"), ("Glue Sticks", "1.15"), (" paint set. ", "9.80"),
    ("unicorn", "unknown item"),
])
def test_lookup(item, expected):
    assert agent.lookup(item) == expected


# --- tasks and reward ----------------------------------------------------------------------------
def numbers(text):
    return [int(n) for n in re.findall(r"\d+", text)]


@pytest.mark.parametrize("kind", agent.TASK_KINDS)
def test_tasks_have_correct_answers(kind):
    rng = random.Random(0)
    for _ in range(50):
        task = agent.make_task(rng, kind)
        q = task["question"]
        assert task["kind"] == kind
        if kind == "math":
            assert task["answer"] > 0
            continue
        items = [name for name in agent.PRICES if f" {name}s" in q]
        n = numbers(q)
        if kind == "shop1":
            expected = n[0] * agent.PRICES[items[0]]
        elif kind == "shop2":
            first, second = sorted(items, key=q.index)
            expected = n[0] * agent.PRICES[first] + n[1] * agent.PRICES[second]
        else:  # budget
            expected = n[0] - n[1] * agent.PRICES[items[0]]
            assert expected > 0
        assert task["answer"] == pytest.approx(expected, abs=0.005)


def test_tasks_are_reproducible():
    a = [agent.make_task(random.Random(10_000 + i)) for i in range(20)]
    b = [agent.make_task(random.Random(10_000 + i)) for i in range(20)]
    assert a == b


@pytest.mark.parametrize("text, answer, reward", [
    ("<lookup>pen</lookup> ... <answer>138.75</answer>", 138.75, 1.0),
    ("<answer>1,234</answer>", 1234, 1.0),
    ("<answer>$12.50</answer>", 12.5, 1.0),
    ("<answer>3</answer> no wait <answer>4</answer>", 4, 1.0),   # the last answer counts
    ("<answer>138.7</answer>", 138.75, 0.0),
    ("The answer is 138.75", 138.75, 0.0),                       # no tags, no reward
    ("<answer>a lot</answer>", 5, 0.0),
])
def test_reward(text, answer, reward):
    assert agent.reward_fn(text, answer) == reward


# --- GRPO pieces ---------------------------------------------------------------------------------
def test_group_advantages():
    adv = agent.group_advantages([1, 0, 0, 0, 1, 1, 1, 1], group_size=4)
    first, second = adv[:4], adv[4:]
    assert first[0] > 0 and (first[1:] < 0).all()      # the lone success stands out
    assert torch.allclose(second, torch.zeros(4))       # all equal -> no signal
    assert abs(first.mean().item()) < 1e-5


def test_right_pad_masks_prompt_tool_and_padding():
    prompt = [[1, 2, 3], [4, 5]]
    completion = [[10, 11, 99, 12], [20]]
    loss_mask = [[1, 1, 0, 1], [1]]                     # token 99 was written by a tool
    ids, attention, mask = agent.right_pad(prompt, completion, loss_mask, pad_id=0)
    assert ids.tolist() == [[1, 2, 3, 10, 11, 99, 12], [4, 5, 20, 0, 0, 0, 0]]
    assert attention.tolist() == [[1] * 7, [1, 1, 1, 0, 0, 0, 0]]
    assert mask.tolist() == [[0, 0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 0, 0]]


def test_grpo_loss_pushes_good_answers_up_and_bad_answers_down():
    logp = torch.full((2, 3), -1.0, requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    advantages = torch.tensor([1.0, -1.0])
    loss = agent.grpo_loss(logp, logp.detach(), advantages, mask, n_tokens=mask.sum())
    loss.backward()
    assert (logp.grad[0, :2] < 0).all()                  # gradient descent raises good tokens
    assert (logp.grad[1] > 0).all()                      # ... and lowers bad tokens
    assert logp.grad[0, 2] == 0                          # masked tokens get no gradient


def test_grpo_loss_clipping_stops_the_push():
    logp = torch.tensor([[0.5]], requires_grad=True)     # ratio = e^0.5 ≈ 1.65 > 1.2
    loss = agent.grpo_loss(logp, torch.zeros(1, 1), torch.tensor([1.0]), torch.ones(1, 1), 1.0)
    loss.backward()
    assert logp.grad.item() == 0


def test_kl_term_is_zero_at_the_reference():
    logp = torch.full((1, 4), -2.0)
    mask = torch.ones(1, 4)
    without = agent.grpo_loss(logp, logp, torch.tensor([0.5]), mask, 4.0)
    with_kl = agent.grpo_loss(logp, logp, torch.tensor([0.5]), mask, 4.0, ref_logp=logp, beta=0.1)
    assert torch.allclose(without, with_kl)


# --- LoRA ----------------------------------------------------------------------------------------
class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.down_proj = nn.Linear(8, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, x):
        return self.norm(self.down_proj(self.q_proj(x)))


def test_lora_starts_as_a_no_op_and_can_be_switched_off():
    torch.manual_seed(0)
    model, x = TinyBlock(), torch.randn(3, 8)
    original = model(x)
    params = agent.add_lora(model, rank=2, alpha=4)
    assert len(params) == 4                              # A and B for q_proj and down_proj
    assert all(not p.requires_grad for n, p in model.named_parameters() if "base" in n or "norm" in n)
    assert torch.allclose(model(x), original)            # B = 0 at init

    with torch.no_grad():
        for p in params:
            p.add_(0.5)
    assert not torch.allclose(model(x), original)        # the patch changes the output
    agent.set_lora(model, False)
    assert torch.allclose(model(x), original)            # switched off = the original model
