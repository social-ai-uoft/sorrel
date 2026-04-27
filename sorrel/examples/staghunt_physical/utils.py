from __future__ import annotations

from typing import Any


def apply_num_groups_to_agent_kinds(agent_config: dict[int, dict[str, Any]], num_groups: int) -> None:
    """Overwrite each agent's ``kind`` with equal-sized contiguous groups (AgentKindA, B, ...).

    Requires ``num_agents % num_groups == 0``. Agent IDs are ordered by numeric sort of keys.
    """
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    if num_groups > 26:
        raise ValueError("num_groups cannot exceed 26 (AgentKindA through AgentKindZ).")

    def _sort_key(k: int | str) -> int:
        if isinstance(k, int):
            return k
        try:
            return int(k)
        except (TypeError, ValueError) as e:
            raise ValueError(f"agent_config keys must be integer agent ids, got {k!r}") from e

    ids = sorted(agent_config.keys(), key=_sort_key)
    n = len(ids)
    if n == 0:
        raise ValueError("agent_config is empty; cannot apply num_groups.")
    if n % num_groups != 0:
        raise ValueError(
            f"num_groups={num_groups} must divide num_agents={n} evenly " "(equal-sized groups)."
        )
    per_group = n // num_groups
    for idx, aid in enumerate(ids):
        g = idx // per_group
        agent_config[aid]["kind"] = f"AgentKind{chr(ord('A') + g)}"

