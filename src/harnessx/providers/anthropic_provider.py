"""Anthropic Messages API provider."""

from __future__ import annotations

import os
from typing import Any

import httpx

from harnessx.events import Message, MessageRole
from harnessx.providers.base import Provider, ProviderResponse


class AnthropicProvider(Provider):
    DEFAULT_BASE = "https://api.anthropic.com"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url or self.DEFAULT_BASE
        self.max_tokens = max_tokens

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        system, body_messages = _to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": body_messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {k: v for k, v in t.items() if k != "type"}
                if t.get("type") == "function"
                else t
                for t in tools
            ]
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = ""
        tool_calls: list[dict[str, Any]] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "arguments": block.get("input", {}),
                    }
                )
        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=data.get("stop_reason"),
            raw=data,
        )


def _to_anthropic(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    system = ""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == MessageRole.SYSTEM:
            system += (m.content + "\n")
            continue
        if m.role == MessageRole.TOOL:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                        }
                    ],
                }
            )
            continue
        if m.role == MessageRole.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": tc.get("name"),
                        "input": tc.get("arguments", {}),
                    }
                )
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": "user", "content": m.content})
    return system.rstrip("\n"), out
