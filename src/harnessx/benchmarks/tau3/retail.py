"""Retail domain for τ³-Bench (reference domain).

Implements a representative tool set over a seeded in-memory database, a
scripted user simulator, and a rule-compliance verifier that checks the final
database state against the task's expected outcome. The structure mirrors
τ³-Bench and is extended to Airline / Telecom by adding sibling domains.
"""

from __future__ import annotations

import json
from typing import Any

from harnessx.benchmarks.tau3.db import Database
from harnessx.benchmarks.tau3.domain import Domain, ToolSpec, UserMessage, UserSimulator


def _seed_db() -> Database:
    return Database(
        {
            "users": [
                {
                    "user_id": "user_1",
                    "name": "Alice Brown",
                    "email": "alice@example.com",
                    "address": "1 Main St",
                }
            ],
            "products": [
                {"product_id": "p100", "name": "Lamp", "price": 29.99, "available": True},
                {"product_id": "p200", "name": "Chair", "price": 89.5, "available": True},
            ],
            "orders": [
                {
                    "order_id": "123",
                    "user_id": "user_1",
                    "status": "pending",
                    "address": "1 Main St",
                    "items": [{"product_id": "p100", "quantity": 2}],
                    "payment_id": "pay_1",
                },
                {
                    "order_id": "124",
                    "user_id": "user_1",
                    "status": "delivered",
                    "address": "1 Main St",
                    "items": [{"product_id": "p200", "quantity": 1}],
                    "payment_id": "pay_2",
                },
            ],
            "payments": [
                {"payment_id": "pay_1", "amount": 59.98, "status": "paid"},
                {"payment_id": "pay_2", "amount": 89.5, "status": "paid"},
            ],
        }
    )


async def _get_user_details(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    user = db.get("users", "user_id", args["user_id"])
    if user is None:
        return {"error": "user not found"}
    return user


async def _get_user_orders(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    orders = [
        {k: o[k] for k in ("order_id", "status", "items", "payment_id")}
        for o in db.rows("orders")
        if o["user_id"] == args["user_id"]
    ]
    return {"orders": orders}


async def _get_order_details(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    order = db.get("orders", "order_id", args["order_id"])
    if order is None:
        return {"error": "order not found"}
    return order


async def _get_product_details(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    product = db.get("products", "product_id", args["product_id"])
    if product is None:
        return {"error": "product not found"}
    return product


async def _cancel_order(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    order = db.get("orders", "order_id", args["order_id"])
    if order is None:
        return {"error": "order not found"}
    if order["status"] not in ("pending", "processing"):
        return {"error": f"order is {order['status']}, cannot cancel"}
    db.update("orders", "order_id", args["order_id"], {"status": "cancelled"})
    return {"status": "success", "order_status": "cancelled"}


async def _modify_pending_order_address(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    order = db.get("orders", "order_id", args["order_id"])
    if order is None:
        return {"error": "order not found"}
    if order["status"] != "pending":
        return {"error": f"order is {order['status']}, cannot modify address"}
    db.update("orders", "order_id", args["order_id"], {"address": args["address"]})
    return {"status": "success", "address": args["address"]}


async def _modify_pending_order_items(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    order = db.get("orders", "order_id", args["order_id"])
    if order is None:
        return {"error": "order not found"}
    if order["status"] != "pending":
        return {"error": f"order is {order['status']}, cannot modify items"}
    db.update("orders", "order_id", args["order_id"], {"items": args["items"]})
    return {"status": "success", "items": args["items"]}


async def _exchange_delivered_order_items(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    order = db.get("orders", "order_id", args["order_id"])
    if order is None:
        return {"error": "order not found"}
    if order["status"] != "delivered":
        return {"error": f"order is {order['status']}, cannot exchange"}
    db.update("orders", "order_id", args["order_id"], {"items": args["items"]})
    return {"status": "success", "items": args["items"]}


async def _get_payment_details(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    payment = db.get("payments", "payment_id", args["payment_id"])
    if payment is None:
        return {"error": "payment not found"}
    return payment


async def _modify_pending_order_payment(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    order = db.get("orders", "order_id", args["order_id"])
    if order is None:
        return {"error": "order not found"}
    if order["status"] != "pending":
        return {"error": f"order is {order['status']}, cannot modify payment"}
    if db.get("payments", "payment_id", args["payment_id"]) is None:
        return {"error": "payment not found"}
    db.update("orders", "order_id", args["order_id"], {"payment_id": args["payment_id"]})
    return {"status": "success", "payment_id": args["payment_id"]}


async def _message_user(db: Database, args: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", "message": args["message"]}


class RetailDomain(Domain):
    name = "retail"

    def build_db(self) -> Database:
        return _seed_db()

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "get_user_details",
                "Get a user's details by user_id.",
                {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
                _get_user_details,
            ),
            ToolSpec(
                "get_user_orders",
                "List orders for a user.",
                {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
                _get_user_orders,
            ),
            ToolSpec(
                "get_order_details",
                "Get full details of an order.",
                {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
                _get_order_details,
            ),
            ToolSpec(
                "get_product_details",
                "Get product details by product_id.",
                {"type": "object", "properties": {"product_id": {"type": "string"}}, "required": ["product_id"]},
                _get_product_details,
            ),
            ToolSpec(
                "cancel_order",
                "Cancel a pending/processing order.",
                {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
                _cancel_order,
            ),
            ToolSpec(
                "modify_pending_order_address",
                "Change the shipping address of a pending order.",
                {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}, "address": {"type": "string"}},
                    "required": ["order_id", "address"],
                },
                _modify_pending_order_address,
            ),
            ToolSpec(
                "modify_pending_order_items",
                "Change the items of a pending order.",
                {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}, "items": {"type": "array"}},
                    "required": ["order_id", "items"],
                },
                _modify_pending_order_items,
            ),
            ToolSpec(
                "exchange_delivered_order_items",
                "Exchange items of a delivered order.",
                {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}, "items": {"type": "array"}},
                    "required": ["order_id", "items"],
                },
                _exchange_delivered_order_items,
            ),
            ToolSpec(
                "get_payment_details",
                "Get payment details by payment_id.",
                {"type": "object", "properties": {"payment_id": {"type": "string"}}, "required": ["payment_id"]},
                _get_payment_details,
            ),
            ToolSpec(
                "modify_pending_order_payment",
                "Change the payment method of a pending order.",
                {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}, "payment_id": {"type": "string"}},
                    "required": ["order_id", "payment_id"],
                },
                _modify_pending_order_payment,
            ),
            ToolSpec(
                "message_user",
                "Send a message to the user.",
                {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
                _message_user,
            ),
        ]

    def user_simulator(self, task: Any, provider: Any = None) -> UserSimulator:
        instruction = getattr(task, "instruction", None)
        if provider is not None and instruction:
            from harnessx.benchmarks.tau3.usersim import PolicyUserSimulator

            return PolicyUserSimulator(provider, instruction)
        return ScriptedUserSimulator(task)


class ScriptedUserSimulator(UserSimulator):
    """Deterministic, script-driven persona for a minimal reproduction.

    Steps through the task's script, replying with the next line and stopping
    when the script is exhausted. An LLM persona conditioned on the policy can
    replace this class without changing the runner or verifier.
    """

    def __init__(self, task: Any) -> None:
        self.task = task
        self.index = 0

    async def respond(self, agent_message: str) -> UserMessage:
        script = getattr(self.task, "script", []) or []
        if self.index >= len(script):
            return UserMessage(content="", stop=True)
        turn = script[self.index]
        self.index += 1
        return UserMessage(content=turn.get("reply", ""), stop=bool(turn.get("stop", False)))


def verify_retail(task: Any, db_state: dict[str, Any]) -> bool:
    expected = getattr(task, "expected", None) or {}
    for table, checks in expected.items():
        if table not in db_state:
            return False
        rows = db_state[table]
        for check in checks:
            key_field = check.get("key_field", _key_field_for(table))
            row = next((r for r in rows if r.get(key_field) == check["key"]), None)
            if row is None:
                return False
            for field, want in check.get("fields", {}).items():
                if row.get(field) != want:
                    return False
    return True


def _key_field_for(table: str) -> str:
    return {
        "orders": "order_id",
        "users": "user_id",
        "payments": "payment_id",
        "products": "product_id",
    }.get(table, "id")


DOMAINS: dict[str, Domain] = {
    "retail": RetailDomain(),
}


def get_domain(name: str) -> Domain:
    if name not in DOMAINS:
        raise KeyError(f"unknown tau3 domain {name!r}")
    return DOMAINS[name]


def stringify_result(result: Any) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)
