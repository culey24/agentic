"""OpenRouter provider.

OpenRouter exposes an OpenAI-compatible Chat Completions API. This provider
subclasses :class:`OpenAIProvider`, overriding the default base URL, the API key
environment variable, and adding the recommended OpenRouter headers.
"""

from __future__ import annotations

import os
from typing import Any

from harnessx.providers.openai_provider import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    DEFAULT_BASE = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"
    endpoint_suffix = "/chat/completions"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key=api_key, base_url=base_url, **kwargs)

    def _extra_headers(self) -> dict[str, str]:
        return {
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", ""),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "harnessx"),
        }
