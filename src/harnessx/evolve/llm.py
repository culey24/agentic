"""LLM helper: call the meta-agent provider with a prompt and parse JSON.

Every AEGIS stage invokes the meta-agent (Claude Opus in the paper). This helper
centralizes the prompt/parse/retry mechanics and the structured-output contract.
"""

from __future__ import annotations

import json
import re
from typing import Any

from harnessx.events import Message, MessageRole
from harnessx.providers.base import Provider


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


async def call_json(
    provider: Provider,
    system: str,
    user: str,
    max_tokens: int = 8192,
) -> Any:
    messages = [
        Message(role=MessageRole.SYSTEM, content=system),
        Message(role=MessageRole.USER, content=user),
    ]
    response = await provider.generate(messages, max_tokens=max_tokens)
    return _extract_json(response.content)
