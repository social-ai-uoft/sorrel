"""Entity map shuffler for changing visual appearances in observations."""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional


class EntityMapShuffler:
    """Manages shuffling of entity appearances in observation specs."""
    
    def __init__(
        self, 
        resource_entities: List[str],
        csv_file_path: Path,
        enable_logging: bool = True,
        shuffle_constraint: str = "no_fixed"
    ):
        """Initialize the entity map shuffler.
        
        Args:
            resource_entities: List of resource entity types to shuffle (e.g., ["A", "B", "C", "D", "E"])
            csv_file_path: Path to CSV log file
            enable_logging: Whether to enable CSV logging
            shuffle_constraint: Constraint type for shuffling
        """
        self.resource_entities = resource_entities.copy()
        self.csv_file_path = csv_file_path
        self.enable_logging = enable_logging
        self.shuffle_constraint = shuffle_constraint
        
        # Track current appearance mapping (entity -> visual appearance)
        self.current_mapping = {entity: entity for entity in resource_entities}
        
        # Initialize CSV file if logging is enabled
        if self.enable_logging:
            self._initialize_csv()
    
    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers."""
        self.csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['epoch', 'shuffle_occurred']
            for entity in self.resource_entities:
                headers.append(f'{entity}_appears_as')
            writer.writerow(headers)
    
    def shuffle_appearances(self) -> Dict[str, str]:
        """Shuffle entity appearances based on constraint.
        
        Returns:
            New mapping dictionary (entity -> visual appearance)
        """
        if self.shuffle_constraint == "no_fixed":
            self.current_mapping = self._shuffle_no_fixed()
        elif self.shuffle_constraint == "allow_fixed":
            self.current_mapping = self._shuffle_allow_fixed()
        elif self.shuffle_constraint == "force_all_different":
            self.current_mapping = self._shuffle_force_all_different()
        
        return self.current_mapping.copy()
    
    def _shuffle_no_fixed(self) -> Dict[str, str]:
        """Shuffle ensuring no entity stays the same."""
        max_attempts = 1000  # Prevent infinite loops
        
        for attempt in range(max_attempts):
            shuffled_appearances = self.resource_entities.copy()
            random.shuffle(shuffled_appearances)
            
            new_mapping = {}
            has_fixed = False
            
            for i, entity in enumerate(self.resource_entities):
                new_mapping[entity] = shuffled_appearances[i]
                if entity == shuffled_appearances[i]:
                    has_fixed = True
                    break
            
            if not has_fixed:
                return new_mapping
        
        # Fallback: if we can't find a no-fixed solution, use allow_fixed
        print("Warning: Could not find no-fixed solution, using allow_fixed")
        return self._shuffle_allow_fixed()
    
    def _shuffle_allow_fixed(self) -> Dict[str, str]:
        """Shuffle allowing any mapping (including fixed positions)."""
        shuffled_appearances = self.resource_entities.copy()
        random.shuffle(shuffled_appearances)
        
        new_mapping = {}
        for i, entity in enumerate(self.resource_entities):
            new_mapping[entity] = shuffled_appearances[i]
        
        return new_mapping
    
    def _shuffle_force_all_different(self) -> Dict[str, str]:
        """Shuffle ensuring all entities change (derangement)."""
        max_attempts = 1000
        
        for attempt in range(max_attempts):
            shuffled_appearances = self.resource_entities.copy()
            random.shuffle(shuffled_appearances)
            
            all_different = True
            for i, entity in enumerate(self.resource_entities):
                if entity == shuffled_appearances[i]:
                    all_different = False
                    break
            
            if all_different:
                new_mapping = {}
                for i, entity in enumerate(self.resource_entities):
                    new_mapping[entity] = shuffled_appearances[i]
                return new_mapping
        
        # Fallback: if we can't find a derangement, use no_fixed
        print("Warning: Could not find derangement, using no_fixed")
        return self._shuffle_no_fixed()
    
    def get_current_mapping(self) -> Dict[str, str]:
        """Get current appearance mapping."""
        return self.current_mapping.copy()
    
    def log_to_csv(self, epoch: int, shuffle_occurred: bool = False) -> None:
        """Log current mapping to CSV file."""
        if not self.enable_logging:
            return
        
        with open(self.csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [epoch, shuffle_occurred]
            for entity in self.resource_entities:
                row.append(self.current_mapping[entity])
            writer.writerow(row)
    
    def get_appearance_for_entity(self, entity_type: str) -> str:
        """Get what visual appearance an entity should have."""
        return self.current_mapping.get(entity_type, entity_type)
    
    def reset_to_original(self) -> None:
        """Reset mapping to original (no shuffling)."""
        self.current_mapping = {entity: entity for entity in self.resource_entities}
    
    def update_csv_path(self, run_folder: str) -> None:
        """Update the CSV file path to include run_folder as prefix."""
        new_csv_path = self.csv_file_path.parent / f"{run_folder}_entity_appearances.csv"
        self.csv_file_path = new_csv_path
        
        # Reinitialize CSV file with new path if logging is enabled
        if self.enable_logging:
            self._initialize_csv()
    
    def apply_to_entity_map(self, entity_map: Dict[str, any]) -> Dict[str, any]:
        """Apply current appearance mapping to an entity_map from OneHotObservationSpec.
        
        Args:
            entity_map: The entity_map from OneHotObservationSpec
            
        Returns:
            New entity_map with shuffled appearances
        """
        new_entity_map = {}
        
        # Copy non-resource entities as-is
        for entity_type, appearance in entity_map.items():
            if entity_type not in self.resource_entities:
                new_entity_map[entity_type] = appearance
        
        # Apply shuffled appearances to resource entities
        for entity_type in self.resource_entities:
            if entity_type in entity_map:
                # Get the visual appearance this entity should have
                visual_appearance = self.current_mapping[entity_type]
                # Use the appearance from the original entity_map
                new_entity_map[entity_type] = entity_map[visual_appearance]
        
        return new_entity_map
