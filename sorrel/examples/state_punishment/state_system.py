"""State system for managing punishment levels and voting in the state punishment
game."""

from copy import deepcopy
from typing import Dict, List

import numpy as np
import random

predefined_punishment_probs = np.array([
    [0.50, 0.00, 0.00, 0.00, 0.00],  # s = 0
    [0.55, 0.05, 0.00, 0.00, 0.00],  # s = 1
    [0.60, 0.10, 0.00, 0.00, 0.00],  # s = 2
    [0.65, 0.10, 0.05, 0.00, 0.00],  # s = 3
    [0.70, 0.10, 0.10, 0.00, 0.00],  # s = 4
    [0.75, 0.10, 0.10, 0.05, 0.00],  # s = 5
    [0.80, 0.10, 0.10, 0.10, 0.00],  # s = 6
    [0.85, 0.15, 0.10, 0.10, 0.05],  # s = 7
    [0.90, 0.15, 0.15, 0.10, 0.10],  # s = 8
    [0.95, 0.20, 0.15, 0.15, 0.10],  # s = 9
])

# increase the punishment prob for A
# predefined_punishment_probs = np.array([
#     [0.40, 0.00, 0.00, 0.00, 0.00],  # s = 0
#     [0.45, 0.05, 0.00, 0.00, 0.00],  # s = 1
#     [0.55, 0.10, 0.00, 0.00, 0.00],  # s = 2
#     [0.70, 0.10, 0.05, 0.00, 0.00],  # s = 3
#     [0.80, 0.10, 0.10, 0.00, 0.00],  # s = 4
#     [0.90, 0.10, 0.10, 0.05, 0.00],  # s = 5
#     [0.95, 0.10, 0.10, 0.10, 0.00],  # s = 6
#     [1.0, 0.15, 0.10, 0.10, 0.05],  # s = 7
#     [1.0, 0.15, 0.15, 0.10, 0.10],  # s = 8
#     [1.0, 0.20, 0.15, 0.15, 0.10],  # s = 9
# ])

# predefined_punishment_probs = np.array([
#     [0.50, 0.00, 0.00, 0.00, 0.00],  # s = 0
#     [0.55, 0.05, 0.00, 0.00, 0.00],  # s = 1
#     [0.60, 0.10, 0.00, 0.00, 0.00],  # s = 2 here (25) 
#     [0.65, 0.15, 0.05, 0.00, 0.00],  # s = 3
#     [0.70, 0.20, 0.10, 0.00, 0.00],  # s = 4
#     [0.75, 0.25, 0.10, 0.05, 0.00],  # s = 5
#     [0.80, 0.30, 0.10, 0.10, 0.00],  # s = 6
#     [0.85, 0.35, 0.10, 0.10, 0.05],  # s = 7 here (16)
#     [0.90, 0.40, 0.15, 0.10, 0.10],  # s = 8
#     [0.95, 0.45, 0.15, 0.15, 0.10],  # s = 9
# ])

# in total there are three bad resources: value (22, 17, 11, 8, 8); harm (12, 9, 6, 0, 0); punishment value 25
# predefined_punishment_probs = np.array([
#     [0.50, 0.30, 0.10, 0.00, 0.00],  # s = 0
#     [0.60, 0.35, 0.10, 0.00, 0.00],  # s = 1 (b1 less than good)
#     [0.70, 0.40, 0.10, 0.00, 0.00],  # s = 2 (b2 less than good)
#     [0.80, 0.45, 0.15, 0.00, 0.00],  # s = 3 (b3 less than good) (all bad less value than good) 
#     [0.90, 0.50, 0.20, 0.00, 0.00],  # s = 4 (b1 less than 0)
#     [1.00, 0.60, 0.25, 0.05, 0.00],  # s = 5 
#     [1.00, 0.70, 0.35, 0.10, 0.00],  # s = 6 (b2 less than 0)
#     [1.00, 0.80, 0.45, 0.10, 0.05],  # s = 7 (b3 less than 0)
#     [1.00, 0.90, 0.55, 0.10, 0.10],  # s = 8 
#     [1.00, 1.00, 0.65, 0.15, 0.10],  # s = 9 
# ])

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
    num_resources=5, num_steps=10, exponentialness=0.12, intercept_increase_speed=2
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
        taboo_resources: List[str] = None,
        num_resources: int = 5,
        num_steps: int = 10,
        exponentialness: float = 0.12,
        intercept_increase_speed: float = 2,
        resource_punishment_is_ambiguous: bool = False,
        only_punish_taboo: bool = True,
        use_probabilistic_punishment: bool = True,
        use_predefined_punishment_schedule: bool = False,
        reset_punishment_level_per_epoch: bool = True,  # NEW: Control punishment reset at epoch start
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
            use_probabilistic_punishment: Whether to use probabilistic punishment (True) or deterministic (False)
            use_predefined_punishment_schedule: If True, use the predefined_punishment_probs
                array from this module. The array must have shape (num_steps, num_resources).
                If False (default), use compile_punishment_vals() to generate schedules.
        
        Raises:
            ValueError: If predefined schedule dimensions don't match num_steps/num_resources.
        """
        self.prob = init_prob
        self.init_prob = init_prob
        self.reset_punishment_level_per_epoch = reset_punishment_level_per_epoch  # NEW
        self.magnitude = magnitude
        self.change_per_vote = change_per_vote
        self.taboo_resources = taboo_resources or ["A", "B", "C", "D", "E"]
        self.vote_history = []
        self.punishment_history = []
        
        # Voting season tracking
        self.voting_season_enabled = False
        self.voting_season_interval = 10
        self.voting_season_reset_per_epoch = True
        self.voting_season_counter = 0  # Steps since last voting season (0 = voting season)
        self.is_voting_season = False  # Current voting season status
        
        # Note: Counter starts at 0, so if voting season is enabled and reset_per_epoch=True,
        # the first turn of each epoch will be a voting season (counter == 0).

        # Advanced punishment system parameters
        self.num_resources = num_resources
        self.num_steps = num_steps
        self.exponentialness = exponentialness
        self.intercept_increase_speed = intercept_increase_speed
        self.resource_punishment_is_ambiguous = resource_punishment_is_ambiguous
        self.only_punish_taboo = only_punish_taboo
        self.use_probabilistic_punishment = use_probabilistic_punishment

        # Select punishment schedule based on parameter
        if use_predefined_punishment_schedule:
            # Use predefined punishment schedule
            # predefined_punishment_probs shape: (num_steps, num_resources) = (10, 5)
            # Need to transpose to (num_resources, num_steps) = (5, 10)
            
            # Validate predefined array dimensions match config
            if predefined_punishment_probs.shape[0] != num_steps:
                raise ValueError(
                    f"Predefined schedule has {predefined_punishment_probs.shape[0]} steps, "
                    f"but num_steps={num_steps}. Expected {num_steps} steps. "
                    f"Either set num_steps={predefined_punishment_probs.shape[0]} or disable --use_predefined_punishment_schedule."
                )
            if predefined_punishment_probs.shape[1] != num_resources:
                raise ValueError(
                    f"Predefined schedule has {predefined_punishment_probs.shape[1]} resources, "
                    f"but num_resources={num_resources}. Expected {num_resources} resources. "
                    f"Either set --num_resources={predefined_punishment_probs.shape[1]} or disable --use_predefined_punishment_schedule."
                )
            
            # Transpose to match expected format: (num_resources, num_steps)
            self.punishments_prob_matrix = predefined_punishment_probs.T
        else:
            # Use compiled punishment values (existing behavior)
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
        
        # Punishment level tracking for epoch averaging
        self.punishment_level_history = []

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

            # Calculate punishment based on mode
            if self.use_probabilistic_punishment:
                # Probabilistic punishment: random chance based on probability
                if random.random() < punishment_prob:
                    punishment_value = self.magnitude
                else:
                    punishment_value = 0.0
            else:
                # Deterministic punishment: proportional to probability
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

    def set_voting_season_config(
        self, 
        enabled: bool, 
        interval: int, 
        reset_per_epoch: bool
    ) -> None:
        """Configure voting season parameters.
        
        Args:
            enabled: Whether voting season mode is enabled
            interval: Steps between voting seasons (must be > 0)
            reset_per_epoch: Whether to reset counter at epoch start
            
        Raises:
            ValueError: If interval <= 0
        """
        if interval <= 0:
            raise ValueError(f"voting_season_interval must be > 0, got {interval}")
        
        self.voting_season_enabled = enabled
        self.voting_season_interval = interval
        self.voting_season_reset_per_epoch = reset_per_epoch
        if not enabled:
            self.is_voting_season = False
            self.voting_season_counter = 0

    def update_voting_season(self) -> None:
        """Update voting season status based on step counter.
        
        Called at the start of each turn. If counter == 0, it's voting season.
        After checking, increment counter. When counter reaches interval, reset to 0.
        """
        if not self.voting_season_enabled:
            self.is_voting_season = False
            return
        
        # Check if it's voting season (counter == 0 means voting time)
        self.is_voting_season = (self.voting_season_counter == 0)
        
        # Increment counter for next turn
        self.voting_season_counter += 1
        
        # Reset counter if interval reached (next turn will be voting season)
        if self.voting_season_counter >= self.voting_season_interval:
            self.voting_season_counter = 0

    def reset_voting_season_counter(self) -> None:
        """Reset voting season counter (called at epoch start if reset_per_epoch=True).
        
        This ensures that if reset_per_epoch=True, each epoch starts with a voting season
        (counter = 0, is_voting_season = True).
        """
        if self.voting_season_reset_per_epoch:
            self.voting_season_counter = 0
            self.is_voting_season = True  # Start new epoch with voting season

    def reset_epoch(self) -> None:
        """Reset epoch-specific tracking.
        
        This is called on shared_state_system at the start of each epoch in run_experiment().
        Individual world state_systems are reset via world.reset() -> state_system.reset().
        """
        # Reset punishment level if configured to do so (for shared_state_system)
        # This handles the case where shared_state_system.reset() is not called
        if self.reset_punishment_level_per_epoch:
            self.prob = self.init_prob
        
        self.epoch_vote_up = 0
        self.epoch_vote_down = 0
        self.epoch_vote_history.append(
            {
                "vote_up": self.epoch_vote_up,
                "vote_down": self.epoch_vote_down,
                "punishment_level": self.prob,
            }
        )
        # Reset punishment level history for new epoch
        self.punishment_level_history = []
        
        # Reset voting season counter if configured
        if self.voting_season_reset_per_epoch:
            self.reset_voting_season_counter()

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
        """Reset the state system to initial values.
        
        NOTE: For the shared_state_system used by agents, this method is rarely called.
        Individual world state_systems call this, but agents use shared_state_system.
        The shared_state_system is reset via reset_epoch() in run_experiment().
        """
        # Only reset punishment level if configured to do so
        if self.reset_punishment_level_per_epoch:
            self.prob = self.init_prob
        
        # Always reset these tracking variables
        self.vote_history = []
        self.punishment_history = []
        self.transgression_record = {resource: [] for resource in self.taboo_resources}
        self.punishment_record = {resource: [] for resource in self.taboo_resources}
        self.epoch_vote_up = 0
        self.epoch_vote_down = 0
        self.epoch_vote_history = []
        self.punishment_level_history = []

    def record_punishment_level(self) -> None:
        """Record current punishment level for epoch averaging."""
        self.punishment_level_history.append(self.prob)

    def get_average_punishment_level(self) -> float:
        """Get average punishment level for the current epoch."""
        if not self.punishment_level_history:
            return self.prob
        return sum(self.punishment_level_history) / len(self.punishment_level_history)

    def get_state_info(self) -> Dict:
        """Get current state information for observations."""
        return {
            "punishment_prob": self.prob,
            "recent_votes": self.vote_history[-5:] if self.vote_history else [],
            "total_votes": len(self.vote_history),
        }
