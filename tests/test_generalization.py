"""Tests for the harder evaluation splits in scripts/evaluate_generalization.py (no model needed)."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_generalization as ev  # noqa: E402
import train_tool_agent as agent  # noqa: E402

SPLITS = ev.build_splits()


def test_every_split_has_the_same_size():
    assert set(SPLITS) == {"in_distribution", "new_phrasing", "new_products", "three_items", "bigger_numbers"}
    assert all(len(tasks) == ev.N_PER_SPLIT for tasks in SPLITS.values())


def test_in_distribution_is_the_original_held_out_set():
    import random
    assert SPLITS["in_distribution"] == [agent.make_task(random.Random(10_000 + i)) for i in range(ev.N_PER_SPLIT)]


def test_new_phrasings_never_reuse_training_wording():
    training_starts = {"How much do", "I have", "A farmer", "A factory", "A school", "A train", "A library", "A bakery"}
    for task in SPLITS["new_phrasing"]:
        assert not any(task["question"].startswith(s) for s in training_starts), task["question"]


def test_new_products_are_outside_the_training_catalog():
    assert not set(ev.NEW_PRODUCTS) & set(agent.PRICES)
    for task in SPLITS["new_products"]:
        assert any(f" {item}s" in task["question"] for item in ev.NEW_PRODUCTS)
        assert not any(f" {item}s" in task["question"] for item in agent.PRICES)


def numbers_and_items(question, catalog):
    return [(int(n), item) for n, item in re.findall(r"(\d+) ([a-z ]+?)s\b", question) if item in catalog]


@pytest.mark.parametrize("split", ["new_products", "three_items"])
def test_shop_answers_are_correct(split):
    catalog = {**agent.PRICES, **ev.NEW_PRODUCTS}
    for task in SPLITS[split]:
        q = task["question"]
        pairs = numbers_and_items(q, catalog)
        cost = sum(n * catalog[item] for n, item in pairs)
        if task["kind"] == "budget":
            budget = int(re.search(r"(\d+) dollars", q).group(1))
            cost = budget - cost
        assert task["answer"] == pytest.approx(cost, abs=0.005), q


def test_three_item_orders_need_three_lookups():
    for task in SPLITS["three_items"]:
        assert len(numbers_and_items(task["question"], agent.PRICES)) == 3


def test_test_only_products_disappear_after_the_test():
    before = dict(agent.PRICES)
    with ev.shop_sells(ev.NEW_PRODUCTS):
        assert agent.lookup("binders") == "4.15"
    assert agent.lookup("binders") == "unknown item"
    assert agent.PRICES == before


def test_few_shot_prompt_is_restored():
    original = agent.SYSTEM_PROMPT
    with ev.instructions(original + ev.FEW_SHOT_EXAMPLE):
        assert agent.SYSTEM_PROMPT.endswith("<answer>5.95</answer>")
    assert agent.SYSTEM_PROMPT == original


def test_few_shot_example_is_correct():
    assert agent.calculator("3 * 0.45 + 2 * 2.30") == "5.95"
    assert agent.lookup("pencil") == "0.45" and agent.lookup("ruler") == "2.30"
