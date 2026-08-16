"""LLM policy-conditioned user simulator (τ³-Bench / tau-bench style).

Faithful to how HarnessX runs τ³-Bench: a dedicated model role plays the user
persona, conditioned on an ``instruction`` (persona + policy + the actions the
user will perform). It replies in natural language and emits ``###STOP###``
when the conversation is complete. The scripted simulator remains available as
a deterministic fallback for offline runs and tests.
"""

from __future__ import annotations

from harnessx.benchmarks.tau3.domain import UserMessage, UserSimulator
from harnessx.events import Message, MessageRole
from harnessx.providers.base import Provider

_STOP = "###STOP###"

_SYSTEM_PROMPT = (
    "You are a user interacting with a customer-service agent. Follow the "
    "given instruction describing who you are and what you want. Stay in "
    "character, be natural, and only agree to actions that match your policy. "
    "When the interaction is fully resolved (your goal is achieved or it is "
    f"clear it cannot be), reply with exactly the token {_STOP} as your entire "
    "response."
)


class PolicyUserSimulator(UserSimulator):
    def __init__(self, provider: Provider, instruction: str) -> None:
        self.provider = provider
        self.instruction = instruction
        self.messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
            Message(role=MessageRole.USER, content=instruction),
        ]

    async def respond(self, agent_message: str) -> UserMessage:
        self.messages.append(Message(role=MessageRole.ASSISTANT, content=agent_message))
        response = await self.provider.generate(self.messages)
        content = response.content.strip()
        stop = _STOP in content
        if stop:
            content = content.replace(_STOP, "").strip()
        self.messages.append(Message(role=MessageRole.USER, content=content or _STOP))
        return UserMessage(content=content, stop=stop)
