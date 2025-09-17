"""State system for managing punishment levels and voting in the state punishment game."""

import numpy as np
from typing import Dict, List


class StateSystem:
    """Manages the punishment state system including voting and punishment calculations."""
    
    def __init__(self, 
                 init_prob: float = 0.1,
                 magnitude: float = -10.0,
                 change_per_vote: float = 0.2,
                 taboo_resources: List[str] = None):
        """
        Initialize the state system.
        
        Args:
            init_prob: Initial punishment probability (0-1)
            magnitude: Magnitude of punishment (negative for punishment)
            change_per_vote: How much the probability changes per vote
            taboo_resources: List of resources that are considered taboo
        """
        self.prob = init_prob
        self.init_prob = init_prob
        self.magnitude = magnitude
        self.change_per_vote = change_per_vote
        self.taboo_resources = taboo_resources or ["Gem", "Bone"]
        self.vote_history = []
        self.punishment_history = []
        
    def vote_increase(self) -> None:
        """Increase punishment probability."""
        self.prob = min(1.0, self.prob + self.change_per_vote)
        self.vote_history.append(1)
        
    def vote_decrease(self) -> None:
        """Decrease punishment probability."""
        self.prob = max(0.0, self.prob - self.change_per_vote)
        self.vote_history.append(-1)
        
    def calculate_punishment(self, resource_kind: str) -> float:
        """
        Calculate punishment for collecting a resource.
        
        Args:
            resource_kind: The kind of resource being collected
            
        Returns:
            Punishment value (negative for punishment)
        """
        if resource_kind in self.taboo_resources:
            # Punishment probability based on current state
            punishment_prob = self.prob
            punishment_value = self.magnitude * punishment_prob
            self.punishment_history.append(punishment_value)
            return punishment_value
        return 0.0
        
    def get_social_harm(self, resource_kind: str) -> float:
        """
        Get social harm caused by collecting a resource.
        
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
        
    def reset(self) -> None:
        """Reset the state system to initial values."""
        self.prob = self.init_prob
        self.vote_history = []
        self.punishment_history = []
        
    def get_state_info(self) -> Dict:
        """Get current state information for observations."""
        return {
            "punishment_prob": self.prob,
            "recent_votes": self.vote_history[-5:] if self.vote_history else [],
            "total_votes": len(self.vote_history)
        }
