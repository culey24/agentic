"""OpenAI-compatible Chat Completions provider."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from harnessx.events import Message, MessageRole
from harnessx.providers.base import Provider, ProviderResponse


class OpenAIProvider(Provider):
    DEFAULT_BASE = "https://api.openai.com"
    api_key_env = "OPENAI_API_KEY"
    endpoint_suffix = "/v1/chat/completions"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.environ.get(self.api_key_env)
        self.base_url = base_url or self.DEFAULT_BASE

    def _extra_headers(self) -> dict[str, str]:
        return {}

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai(messages),
        }
        if tools:
            payload["tools"] = tools
        headers = {
            "Authorization": f"Bearer {self.api_key or ''}",
            "Content-Type": "application/json",
            **self._extra_headers(),
        }
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self.base_url}{self.endpoint_suffix}",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except (TypeError, ValueError):
                arguments = {"_raw": fn.get("arguments", "{}")}
            tool_calls.append(
                {"id": tc.get("id"), "name": fn.get("name"), "arguments": arguments}
            )
        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            stop_reason=choice.get("finish_reason"),
            raw=data,
        )


def _to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == MessageRole.TOOL:
            out.append(
                {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
            )
            continue
        entry: dict[str, Any] = {"role": m.role.value, "content": m.content}
        if m.role == MessageRole.ASSISTANT and m.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": json.dumps(tc.get("arguments", {})),
                    },
                }
                for tc in m.tool_calls
            ]
        out.append(entry)
    return out
