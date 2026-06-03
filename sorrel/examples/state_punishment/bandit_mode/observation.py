"""Bandit observation encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BanditObservationState:
    """Per-trial menu: resource entity identity per arm (aligned with grid type-only coding)."""

    options: Sequence[str]


class BanditObservationSpec:
    """Shape (5, K, 1): per arm, one-hot over resource entities A–E (grid-style type coding).

    Social harm accrued to this agent is only in the scalar tail of ``generate_single_view``,
    not per-arm.
    """

    RESOURCE_ORDER = ("A", "B", "C", "D", "E")

    def __init__(self, n_options: int = 3) -> None:
        self.n_options = int(n_options)
        self.input_size = (len(self.RESOURCE_ORDER), self.n_options, 1)
        self.vision_radius = None  # API compatibility with slot spec

    def observe(self, state: BanditObservationState) -> np.ndarray:
        obs = np.zeros(self.input_size, dtype=np.float32)
        for idx, resource in enumerate(state.options):
            if idx >= self.n_options:
                break
            if resource in self.RESOURCE_ORDER:
                j = self.RESOURCE_ORDER.index(resource)
                obs[j, idx, 0] = 1.0
        return obs
