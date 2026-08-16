"""Model configuration: which model serves which role and the fallback policy.

``ModelConfig`` and ``HarnessConfig`` address disjoint concerns. ``ModelConfig``
records model identity per role (main, judge, evaluator) plus fallback; it is
independent of the agent's behavior.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Role(str, enum.Enum):
    MAIN = "main"
    JUDGE = "judge"
    EVALUATOR = "evaluator"
    META = "meta"
    USER = "user"


@dataclass
class ModelConfig:
    """Maps each role to a provider plus a fallback chain.

    ``main`` is the task agent; ``judge``/``evaluator`` score outcomes;
    ``meta`` drives harness evolution (AEGIS); ``user`` plays the τ³-Bench
    user simulator persona. Any role may fall back to ``main``.
    """

    main: Any = None
    judge: Any = None
    evaluator: Any = None
    meta: Any = None
    user: Any = None
    fallbacks: dict[Role, list[Any]] = field(default_factory=dict)

    def provider_for(self, role: Role | str = Role.MAIN) -> Any:
        role = Role(role) if isinstance(role, str) else role
        provider = getattr(self, role.value)
        if provider is None and role != Role.MAIN:
            provider = self.main
        return provider

    def agentic(self, harness_config: Any) -> Any:
        from harnessx.core.harness import Harness

        return Harness(model_config=self, harness_config=harness_config)
