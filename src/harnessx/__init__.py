"""HarnessX: a composable, adaptive, and evolvable agent harness foundry.

From-scratch reimplementation of the paper "HarnessX: A Composable, Adaptive,
and Evolvable Agent Harness Foundry" (arXiv:2606.14249).
"""

from harnessx.core.harness import Harness
from harnessx.core.harness_config import HarnessConfig
from harnessx.core.hooks import Hook
from harnessx.core.model_config import ModelConfig, Role
from harnessx.core.processor import Order, Processor
from harnessx.core.trajectory import Trajectory

__version__ = "0.1.0"

__all__ = [
    "Harness",
    "HarnessConfig",
    "Hook",
    "ModelConfig",
    "Order",
    "Processor",
    "Role",
    "Trajectory",
    "__version__",
]
