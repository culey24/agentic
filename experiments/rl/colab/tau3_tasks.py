"""Synthesize a tau3 retail task suite with multi-objective trade-off structure.

Task groups (the tension between objectives is what makes multi-objective RL
differ from single-objective):

- simple    : one operation, low tension (single ~= multi).
- lookup    : the agent must discover order/product/payment ids with tools,
              so a focused agent yields fewer turns -> efficiency matters.
- safety    : traps (non-existent order, non-existent payment, delivered order
              mutation) penalize tool errors -> tool_safety matters.
- complex   : several combined operations -> both efficiency and safety matter.

Each task is a plain dict matching the JSONL schema of ``Tau3Adapter.load_tasks``.
The seed database (``harnessx.benchmarks.tau3.retail._seed_db``) is unchanged:
every task starts from users=[user_1], products=[p100 Lamp, p200 Chair],
orders=[123 pending, 124 delivered], payments=[pay_1, pay_2].

Usage:
  python experiments/rl/colab/tau3_tasks.py --output examples/data/tau3_colab.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SIMPLE = [
    {
        "task_id": "colab-cancel-001",
        "opening": "Please cancel order #123.",
        "instruction": "You are Alice Brown (user_1). You want to cancel order #123. "
        "Only agree to cancelling order #123. Say your user_id if asked.",
        "script": [
            {"reply": "Yes, please cancel it.", "stop": False},
            {"reply": "Great, thanks!", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"status": "cancelled"}}]},
    },
    {
        "task_id": "colab-addr-001",
        "opening": "Please change the shipping address on order #123 to 2 Oak St.",
        "instruction": "You are Alice Brown (user_1). You want the shipping address of "
        "order #123 changed to 2 Oak St. Only that order, only that address.",
        "script": [
            {"reply": "Yes, that's the right address.", "stop": False},
            {"reply": "Thank you!", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"address": "2 Oak St"}}]},
    },
    {
        "task_id": "colab-items-001",
        "opening": "Change order #123 to contain one Chair (p200) instead of the Lamp.",
        "instruction": "You are Alice Brown (user_1). You want order #123 changed to "
        "contain exactly one Chair (p200), quantity 1.",
        "script": [
            {"reply": "Yes, one Chair please.", "stop": False},
            {"reply": "Perfect, thanks!", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"items": [{"product_id": "p200", "quantity": 1}]}}]},
    },
    {
        "task_id": "colab-pay-001",
        "opening": "Switch the payment on order #123 to pay_2.",
        "instruction": "You are Alice Brown (user_1). You want the payment on order #123 "
        "changed to pay_2.",
        "script": [
            {"reply": "Yes, use payment pay_2.", "stop": False},
            {"reply": "Thanks!", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"payment_id": "pay_2"}}]},
    },
    {
        "task_id": "colab-exch-001",
        "opening": "Please exchange the item in my delivered order #124 for a Lamp (p100), one unit.",
        "instruction": "You are Alice Brown (user_1). You want the delivered order #124 "
        "exchanged to contain exactly one Lamp (p100), quantity 1.",
        "script": [
            {"reply": "Yes, one Lamp please.", "stop": False},
            {"reply": "Great, thanks!", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "124", "fields": {"items": [{"product_id": "p100", "quantity": 1}]}}]},
    },
    {
        "task_id": "colab-cancel-msg-001",
        "opening": "Cancel order #123 and send me a confirmation message.",
        "instruction": "You are Alice Brown (user_1). You want order #123 cancelled and a "
        "confirmation message sent to you.",
        "script": [
            {"reply": "Yes, cancel it and let me know.", "stop": False},
            {"reply": "Thank you!", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"status": "cancelled"}}]},
    },
]

LOOKUP = [
    {
        "task_id": "colab-addr-lookup-001",
        "opening": "I'd like to change the shipping address on my order.",
        "instruction": "You are Alice Brown (user_1). You want the shipping address of your "
        "order changed to 2 Oak St. Give order details only if the agent asks or looks them up.",
        "script": [
            {"reply": "It's order 123.", "stop": False},
            {"reply": "2 Oak St.", "stop": False},
            {"reply": "Yes, that's right.", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"address": "2 Oak St"}}]},
    },
    {
        "task_id": "colab-pay-lookup-001",
        "opening": "I want to switch the payment method for my most recent order.",
        "instruction": "You are Alice Brown (user_1). You want the payment on your most "
        "recent order (123) changed to pay_2.",
        "script": [
            {"reply": "It's order 123.", "stop": False},
            {"reply": "Use pay_2.", "stop": False},
            {"reply": "Yes, that's it.", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"payment_id": "pay_2"}}]},
    },
    {
        "task_id": "colab-items-lookup-001",
        "opening": "Change my order to contain a Chair instead of a Lamp.",
        "instruction": "You are Alice Brown (user_1). You want order 123 changed to contain "
        "exactly one Chair (p200), quantity 1.",
        "script": [
            {"reply": "It's order 123.", "stop": False},
            {"reply": "One Chair.", "stop": False},
            {"reply": "Yes.", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"items": [{"product_id": "p200", "quantity": 1}]}}]},
    },
    {
        "task_id": "colab-exch-lookup-001",
        "opening": "I need to exchange the item in my delivered order for two Lamps.",
        "instruction": "You are Alice Brown (user_1). You want the delivered order 124 "
        "exchanged to contain exactly two Lamps (p100), quantity 2.",
        "script": [
            {"reply": "It's order 124.", "stop": False},
            {"reply": "Two Lamps.", "stop": False},
            {"reply": "Yes.", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "124", "fields": {"items": [{"product_id": "p100", "quantity": 2}]}}]},
    },
    {
        "task_id": "colab-cancel-lookup-001",
        "opening": "Cancel my latest order and also change its address to 2 Oak St first.",
        "instruction": "You are Alice Brown (user_1). You want your latest order (123) "
        "changed to 2 Oak St and then cancelled.",
        "script": [
            {"reply": "The latest is order 123.", "stop": False},
            {"reply": "2 Oak St.", "stop": False},
            {"reply": "Then cancel it, yes.", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"status": "cancelled", "address": "2 Oak St"}}]},
    },
]

SAFETY = [
    {
        "task_id": "colab-trap-id-001",
        "opening": "Cancel order #999, please. Oh wait, I meant #123.",
        "instruction": "You are Alice Brown (user_1). You want order #123 cancelled. You "
        "mistakenly say 999 first, then correct yourself to 123.",
        "script": [
            {"reply": "Sorry, I meant order #123.", "stop": False},
            {"reply": "Yes, cancel #123.", "stop": False},
            {"reply": "Thanks!", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"status": "cancelled"}}]},
    },
    {
        "task_id": "colab-trap-pay-001",
        "opening": "Change the payment on order #123 to pay_3.",
        "instruction": "You are Alice Brown (user_1). You want order #123 paid with pay_3. "
        "You believe pay_3 exists.",
        "script": [
            {"reply": "Yes, use pay_3.", "stop": False},
            {"reply": "That's right.", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"payment_id": "pay_1"}}]},
    },
    {
        "task_id": "colab-trap-delivered-001",
        "opening": "Cancel orders #124 and #123.",
        "instruction": "You are Alice Brown (user_1). You want orders #124 and #123 "
        "cancelled. You believe both can be cancelled.",
        "script": [
            {"reply": "Yes, both of them.", "stop": False},
            {"reply": "Thank you!", "stop": True},
        ],
        "expected": {
            "orders": [
                {"key_field": "order_id", "key": "123", "fields": {"status": "cancelled"}},
                {"key_field": "order_id", "key": "124", "fields": {"status": "delivered"}},
            ]
        },
    },
    {
        "task_id": "colab-trap-confuse-001",
        "opening": "Change the address on my delivered order - actually the pending one - to 2 Oak St.",
        "instruction": "You are Alice Brown (user_1). You want the pending order (123) "
        "address changed to 2 Oak St. You first mention the delivered order by mistake.",
        "script": [
            {"reply": "The pending one, not the delivered one.", "stop": False},
            {"reply": "2 Oak St.", "stop": False},
            {"reply": "Yes.", "stop": True},
        ],
        "expected": {
            "orders": [
                {"key_field": "order_id", "key": "123", "fields": {"address": "2 Oak St"}},
                {"key_field": "order_id", "key": "124", "fields": {"address": "1 Main St"}},
            ]
        },
    },
]

COMPLEX = [
    {
        "task_id": "colab-complex-001",
        "opening": "Change the address on order #123 to 2 Oak St, switch its payment to pay_2, and cancel it.",
        "instruction": "You are Alice Brown (user_1). On order #123: change address to 2 Oak "
        "St, switch payment to pay_2, then cancel it.",
        "script": [
            {"reply": "2 Oak St, pay_2, then cancel.", "stop": False},
            {"reply": "Yes, that's all.", "stop": False},
            {"reply": "Thank you!", "stop": True},
        ],
        "expected": {"orders": [{"key_field": "order_id", "key": "123", "fields": {"status": "cancelled", "address": "2 Oak St", "payment_id": "pay_2"}}]},
    },
    {
        "task_id": "colab-complex-002",
        "opening": "Exchange my delivered order #124 to a Lamp, change order #123 to one Chair, and switch its payment to pay_2.",
        "instruction": "You are Alice Brown (user_1). Exchange order #124 to one Lamp "
        "(p100), change order #123 to one Chair (p200), and switch order #123 payment to pay_2.",
        "script": [
            {"reply": "Lamp for 124, Chair for 123, pay_2 for 123.", "stop": False},
            {"reply": "Yes, all of it.", "stop": False},
            {"reply": "Thanks!", "stop": True},
        ],
        "expected": {
            "orders": [
                {"key_field": "order_id", "key": "123", "fields": {"items": [{"product_id": "p200", "quantity": 1}], "payment_id": "pay_2"}},
                {"key_field": "order_id", "key": "124", "fields": {"items": [{"product_id": "p100", "quantity": 1}]}},
            ]
        },
    },
    {
        "task_id": "colab-complex-003",
        "opening": "On order #123: make it two Chairs and pay with pay_2. Also exchange my delivered order to one Lamp.",
        "instruction": "You are Alice Brown (user_1). On order #123: set items to two Chairs "
        "(p200, quantity 2) and payment to pay_2. Exchange order #124 to one Lamp (p100).",
        "script": [
            {"reply": "Two Chairs, pay_2, and a Lamp for the delivered order.", "stop": False},
            {"reply": "Yes, that's right.", "stop": False},
            {"reply": "Thanks!", "stop": True},
        ],
        "expected": {
            "orders": [
                {"key_field": "order_id", "key": "123", "fields": {"items": [{"product_id": "p200", "quantity": 2}], "payment_id": "pay_2"}},
                {"key_field": "order_id", "key": "124", "fields": {"items": [{"product_id": "p100", "quantity": 1}]}},
            ]
        },
    },
]

GROUPS: dict[str, list[dict[str, Any]]] = {
    "simple": SIMPLE,
    "lookup": LOOKUP,
    "safety": SAFETY,
    "complex": COMPLEX,
}


def make_task_suite() -> list[dict[str, Any]]:
    """Return the full task suite (simple + lookup + safety + complex)."""
    tasks: list[dict[str, Any]] = []
    for group in ("simple", "lookup", "safety", "complex"):
        for task in GROUPS[group]:
            entry = {
                "task_id": task["task_id"],
                "domain": "retail",
                "opening": task["opening"],
                "instruction": task["instruction"],
                "script": task["script"],
                "expected": task["expected"],
                "group": group,
            }
            tasks.append(entry)
    return tasks


def write_suite(output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(task, ensure_ascii=False) + "\n" for task in make_task_suite())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/data/tau3_colab.jsonl")
    args = parser.parse_args()
    out = write_suite(args.output)
    print(f"wrote {len(make_task_suite())} tasks to {out}")


if __name__ == "__main__":
    main()