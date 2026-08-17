"""Unit tests for the local Qwen provider (parse + wiring, no GPU required)."""

from __future__ import annotations

import asyncio

from experiments.rl.colab.local_provider import LocalQwenProvider, parse_assistant
from harnessx.events import Message, MessageRole

_TOOLS = [
    {
        "type": "function",
        "function": {"name": "cancel_order", "description": "Cancel an order.", "parameters": {}},
    }
]


def test_parse_tool_call() -> None:
    text = '{"tool_call": {"name": "cancel_order", "arguments": {"order_id": "123"}}}'
    content, calls = parse_assistant(text, _TOOLS)
    assert content is None
    assert len(calls) == 1
    assert calls[0]["name"] == "cancel_order"
    assert calls[0]["arguments"] == {"order_id": "123"}


def test_parse_tool_call_within_code_fence() -> None:
    text = '```json\n{"tool_call": {"name": "cancel_order", "arguments": {}}}\n```'
    content, calls = parse_assistant(text, _TOOLS)
    assert content is None
    assert calls[0]["name"] == "cancel_order"


def test_parse_rejects_unknown_tool() -> None:
    text = '{"tool_call": {"name": "hack_db", "arguments": {}}}'
    content, calls = parse_assistant(text, _TOOLS)
    assert content == text
    assert calls == []


def test_parse_plain_text() -> None:
    text = "I've cancelled your order."
    content, calls = parse_assistant(text, _TOOLS)
    assert content == text
    assert calls == []


def test_parse_empty() -> None:
    assert parse_assistant("  ", _TOOLS) == (None, [])


def test_generate_wiring() -> None:
    class FakeProvider(LocalQwenProvider):
        def __init__(self) -> None:
            super().__init__(model_name="fake")
            self._loaded = True

        def _generate(self, conv: list[dict]) -> tuple[str, list[int]]:
            assert any("cancel_order" in str(c) for c in conv)
            return '{"tool_call": {"name": "cancel_order", "arguments": {"order_id": "123"}}}', [1, 2, 3]

    async def main() -> None:
        provider = FakeProvider()
        resp = await provider.generate(
            [Message(role=MessageRole.USER, content="cancel #123")],
            tools=_TOOLS,
        )
        assert resp.tool_calls[0]["name"] == "cancel_order"
        assert resp.raw["text"]

    asyncio.run(main())


def test_generate_plain_answer_wiring() -> None:
    class FakeProvider(LocalQwenProvider):
        def __init__(self) -> None:
            super().__init__(model_name="fake")
            self._loaded = True

        def _generate(self, conv: list[dict]) -> tuple[str, list[int]]:
            return "I need to look that up.", [4, 5]

    async def main() -> None:
        provider = FakeProvider()
        resp = await provider.generate([Message(role=MessageRole.USER, content="hi")], tools=None)
        assert resp.content == "I need to look that up."
        assert resp.tool_calls == []

    asyncio.run(main())


def test_repr() -> None:
    assert "LocalQwenProvider" in repr(LocalQwenProvider(model_name="Qwen/Qwen2.5-3B-Instruct"))