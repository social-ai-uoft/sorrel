class ActionSpec:
    """A class that specifies the possible actions that can be taken by an agent.

    Attributes:
      - n_actions: The number of actions
      - actions: A dictionary matching integers from the model output with the action to be taken.
    """

    n_actions: int
    actions: dict[int, str]

    def __init__(self, actions: list[str]):  # e.g., ["up", "down", "left", "right"]

        self.n_actions = len(actions)
        self.actions = {
            k: v for k, v in zip([i for i in range(self.n_actions)], actions)
        }

    def get_readable_action(self, action: int) -> str:
        """Return a human-readable action from a model output integer.

        Args:
          action: The model action.

        Return:
          str: The human-readable action.
        """
        return self.actions[action]
