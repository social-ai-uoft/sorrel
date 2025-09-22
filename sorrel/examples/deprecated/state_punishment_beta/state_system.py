"""State system for managing punishment levels and voting in the state punishment
game."""

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np


def generate_exponential_function(intercept: float, base: float):
    """
    Returns an exponential function of the form:
    y = intercept * (base ** x)

    Parameters:
    intercept (float): The multiplicative constant (scales the output).
    base (float): The base of the exponential.

    Returns:
    function: A function that computes y for a given x.
    """

    def exponential_function(x):
        return intercept + (base ** (x))

    return exponential_function


def compile_punishment_vals(
    num_resources=5, num_steps=10, exponentialness=0.12, intercept_increase_speed=2.0
):
    """Generates a set of punishment values based on an exponential function.

    Args:
        num_resources (int): Number of resources to consider.
        num_steps (int): Number of steps in the exponential growth.
        exponentialness (float): Parameter controlling the steepness of the curve.
        intercept_increase_speed (float): Parameter controlling the speed of the increase of the intercept.
    """
    vals = []

    # Generate exponential functions for each step
    # and calculate the values for each resource
    for i in range(num_steps):
        exp_func = generate_exponential_function(
            1 + (i / intercept_increase_speed), exponentialness * i + 2
        )
        vals.append([exp_func(k) for k in range(num_resources)])

    vals = np.array(vals)
    max_val = np.max(vals)
    max_val = int(max_val)

    # Normalized values
    punishment_probs = vals / max_val
    punishment_probs = punishment_probs * [1, 1.1, 1.25, 1.45, 1.95]
    punishment_probs = np.clip(punishment_probs, 0.0, 1.0)

    return punishment_probs.T


class StateSystem:
    """Manages the punishment state system including voting and punishment
    calculations."""

    def __init__(
        self,
        init_prob: float = 0.1,
        magnitude: float = -10.0,
        change_per_vote: float = 0.2,
        taboo_resources: Optional[List[str]] = None,
        num_resources: int = 5,
        num_steps: int = 10,
        exponentialness: float = 0.12,
        intercept_increase_speed: float = 2,
        resource_punishment_is_ambiguous: bool = False,
        only_punish_taboo: bool = True,
    ):
        """Initialize the state system.

        Args:
            init_prob: Initial punishment probability (0-1)
            magnitude: Magnitude of punishment (negative for punishment)
            change_per_vote: How much the probability changes per vote
            taboo_resources: List of resources that are considered taboo
            num_resources: Number of resources in the system
            num_steps: Number of punishment levels
            exponentialness: Parameter controlling exponential curve steepness
            intercept_increase_speed: Speed of intercept increase
            resource_punishment_is_ambiguous: Whether punishment is ambiguous
            only_punish_taboo: Whether to only punish taboo resources
        """
        self.prob = init_prob
        self.init_prob = init_prob
        self.magnitude = magnitude
        self.change_per_vote = change_per_vote
        self.taboo_resources = taboo_resources or ["A", "B", "C", "D", "E"]
        self.vote_history = []
        self.punishment_history = []

        # Advanced punishment system parameters
        self.num_resources = num_resources
        self.num_steps = num_steps
        self.exponentialness = exponentialness
        self.intercept_increase_speed = intercept_increase_speed
        self.resource_punishment_is_ambiguous = resource_punishment_is_ambiguous
        self.only_punish_taboo = only_punish_taboo

        # Generate complex punishment probability matrices
        self.punishments_prob_matrix = compile_punishment_vals(
            num_resources, num_steps, exponentialness, intercept_increase_speed
        )

        # Resource-specific punishment schedules
        self.resource_schedules = self._generate_resource_schedules()

        # Transgression and punishment tracking
        self.transgression_record = {resource: [] for resource in self.taboo_resources}
        self.punishment_record = {resource: [] for resource in self.taboo_resources}

        # Vote tracking per epoch
        self.epoch_vote_up = 0
        self.epoch_vote_down = 0
        self.epoch_vote_history = []

    def _generate_resource_schedules(self) -> Dict[str, List[float]]:
        """Generate resource-specific punishment schedules."""
        schedules = {}
        resource_names = ["A", "B", "C", "D", "E"]

        for i, resource in enumerate(resource_names[: self.num_resources]):
            schedules[resource] = self.punishments_prob_matrix[i].tolist()

        return schedules

    def vote_increase(self) -> None:
        """Increase punishment probability."""
        # In simple foraging mode, punishment level is fixed
        if hasattr(self, "simple_foraging") and self.simple_foraging:
            return
        self.prob = min(1.0, self.prob + self.change_per_vote)
        self.vote_history.append(1)
        self.epoch_vote_up += 1

    def vote_decrease(self) -> None:
        """Decrease punishment probability."""
        # In simple foraging mode, punishment level is fixed
        if hasattr(self, "simple_foraging") and self.simple_foraging:
            return
        self.prob = max(0.0, self.prob - self.change_per_vote)
        self.vote_history.append(-1)
        self.epoch_vote_down += 1

    def calculate_punishment(self, resource_kind: str) -> float:
        """Calculate punishment for collecting a resource using complex schedules.

        Args:
            resource_kind: The kind of resource being collected

        Returns:
            Punishment value (negative for punishment)
        """
        # Record transgression
        if resource_kind in self.taboo_resources:
            self.transgression_record[resource_kind].append(1)
        else:
            # Record non-taboo resource collection
            if resource_kind not in self.transgression_record:
                self.transgression_record[resource_kind] = []
            self.transgression_record[resource_kind].append(0)

        # Calculate punishment based on resource-specific schedules
        if resource_kind in self.resource_schedules:
            # Get current punishment level (0 to num_steps-1)
            current_level = min(
                round(self.prob * (self.num_steps - 1)), self.num_steps - 1
            )

            if self.resource_punishment_is_ambiguous:
                # Use ambiguous punishment calculation
                punishment_prob = self.resource_schedules[resource_kind][current_level]
            else:
                if self.only_punish_taboo and resource_kind not in self.taboo_resources:
                    punishment_prob = 0.0
                else:
                    punishment_prob = self.resource_schedules[resource_kind][
                        current_level
                    ]

            punishment_value = self.magnitude * punishment_prob
            self.punishment_history.append(punishment_value)

            # Record punishment
            if resource_kind in self.punishment_record:
                self.punishment_record[resource_kind].append(
                    1 if punishment_value < 0 else 0
                )

            return punishment_value

        return 0.0

    def get_social_harm(self, resource_kind: str) -> float:
        """Get social harm caused by collecting a resource.

        Args:
            resource_kind: The kind of resource being collected

        Returns:
            Social harm value
        """
        # Social harm is independent of punishment level
        # This could be made more sophisticated
        if resource_kind in self.taboo_resources:
            return 1.0
        return 0.0

    def get_social_harm_from_entity(self, entity) -> float:
        """Get social harm from an entity object.

        Args:
            entity: The entity object being collected

        Returns:
            Social harm value from the entity
        """
        if hasattr(entity, "social_harm"):
            return entity.social_harm
        return 0.0

    def reset_epoch(self) -> None:
        """Reset epoch-specific tracking."""
        self.epoch_vote_up = 0
        self.epoch_vote_down = 0
        self.epoch_vote_history.append(
            {
                "vote_up": self.epoch_vote_up,
                "vote_down": self.epoch_vote_down,
                "punishment_level": self.prob,
            }
        )

    def get_epoch_vote_stats(self) -> Dict:
        """Get vote statistics for the current epoch."""
        return {
            "vote_up": self.epoch_vote_up,
            "vote_down": self.epoch_vote_down,
            "total_votes": self.epoch_vote_up + self.epoch_vote_down,
            "punishment_level": self.prob,
        }

    def get_transgression_stats(self) -> Dict:
        """Get transgression and punishment statistics."""
        stats = {}
        for resource in self.taboo_resources:
            if resource in self.transgression_record:
                transgressions = sum(self.transgression_record[resource])
                punishments = sum(self.punishment_record.get(resource, []))
                stats[f"{resource}_transgressions"] = transgressions
                stats[f"{resource}_punishments"] = punishments
                stats[f"{resource}_punishment_rate"] = punishments / max(
                    transgressions, 1
                )
        return stats

    def reset(self) -> None:
        """Reset the state system to initial values."""
        self.prob = self.init_prob
        self.vote_history = []
        self.punishment_history = []
        self.transgression_record = {resource: [] for resource in self.taboo_resources}
        self.punishment_record = {resource: [] for resource in self.taboo_resources}
        self.epoch_vote_up = 0
        self.epoch_vote_down = 0
        self.epoch_vote_history = []

    def get_state_info(self) -> Dict:
        """Get current state information for observations."""
        return {
            "punishment_prob": self.prob,
            "recent_votes": self.vote_history[-5:] if self.vote_history else [],
            "total_votes": len(self.vote_history),
        }
