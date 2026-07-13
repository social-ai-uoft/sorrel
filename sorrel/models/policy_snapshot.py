from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """Read-only policy snapshot tied to a specific model version."""

    policy: Any
    version: int
