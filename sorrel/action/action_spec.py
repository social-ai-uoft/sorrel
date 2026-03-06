from __future__ import annotations


class ActionSpec:
    """Base class for action specifications.

    Attributes:
        n_actions (int): Number of possible actions.
        actions (dict[int, str]): Mapping from integer indices to action strings.
    """

    n_actions: int
    actions: dict[int, str]

    def __init__(self, actions: list[str]):  # e.g., ["up", "down", "left", "right"]
        """Initialize the specification with a list of action strings.

        Args:
            actions: List of action identifiers.
        """
        self.n_actions = len(actions)
        # Map each index to its action string
        self.actions = {i: v for i, v in enumerate(actions)}
        # Reverse lookup for convenience
        self._action_to_index = {v: i for i, v in self.actions.items()}

    def get_readable_action(self, action: int) -> str:
        """Return a human‑readable action from a model output integer.

        Args:
            action: The model action index.

        Returns:
            The corresponding action string.
        """
        return self.actions[action]

    def get_action_index(self, action_str: str) -> int | None:
        """Return the integer index for a given action string.

        Args:
            action_str: The action string (e.g., "e2e4").

        Returns:
            The corresponding index, or ``None`` if not found.
        """
        return self._action_to_index.get(action_str)
