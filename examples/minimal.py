"""Minimal runnable example (matches the paper's Python SDK snippet)."""

import asyncio

from harnessx import HarnessConfig, ModelConfig
from harnessx.providers.anthropic_provider import AnthropicProvider


class Task:
    id = "example"
    description = "What is 2 + 2?"


async def main() -> None:
    model = ModelConfig(main=AnthropicProvider("claude-sonnet-4-6"))
    harness = model.agentic(HarnessConfig())
    result = await harness.run(Task())
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
