"""Recommended PettingZoo wrapper stack for Sorrel AEC environments."""

from __future__ import annotations

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


def apply_recommended_wrappers(
    env: AECEnv,
    *,
    capture_stdout: bool = False,
    clip_out_of_bounds: bool = False,
) -> AECEnv:
    """Apply a sensible default wrapper stack for PettingZoo AEC environments."""

    wrapped = env
    if capture_stdout:
        wrapped = wrappers.CaptureStdoutWrapper(wrapped)

    if clip_out_of_bounds:
        wrapped = wrappers.ClipOutOfBoundsWrapper(wrapped)
    else:
        wrapped = wrappers.AssertOutOfBoundsWrapper(wrapped)

    wrapped = wrappers.OrderEnforcingWrapper(wrapped)
    return wrapped
