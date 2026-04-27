"""Minimal world object for bandit training loops."""


class BanditWorldStub:
    """Holds scalar fields used by logging and coordination."""

    def __init__(self) -> None:
        self.is_done = False
        self.total_reward = 0.0

    def create_world(self) -> None:
        """Compatibility hook (no grid)."""
        return

    def record_punishment_level(self) -> None:
        """Compatibility hook."""
        return
