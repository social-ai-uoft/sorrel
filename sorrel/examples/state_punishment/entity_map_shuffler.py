"""Entity map shuffler for changing visual appearances in observations."""

import csv
import json
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
        shuffle_constraint: str = "no_fixed",
        mapping_file_path: Optional[Path] = None,
    ):
        """Initialize the entity map shuffler.

        Args:
            resource_entities: List of resource entity types to shuffle (e.g., ["A", "B", "C", "D", "E"])
            csv_file_path: Path to CSV log file
            enable_logging: Whether to enable CSV logging
            shuffle_constraint: Constraint type for shuffling
            mapping_file_path: Optional path to pre-generated mapping file
        """
        self.resource_entities = resource_entities.copy()
        self.csv_file_path = csv_file_path
        self.enable_logging = enable_logging
        self.shuffle_constraint = shuffle_constraint
        self.mapping_file_path = mapping_file_path

        # Track current appearance mapping (entity -> visual appearance)
        self.current_mapping = {entity: entity for entity in resource_entities}

        # Track previous mapping for adjacent diversity
        self.previous_mapping = None

        # Load pre-generated mappings if file is provided
        self.loaded_mappings = []
        self.current_mapping_index = 0
        if self.mapping_file_path and self.mapping_file_path.exists():
            self.loaded_mappings = self._load_mappings_from_file()
            print(
                f"Loaded {len(self.loaded_mappings)} mappings from {self.mapping_file_path}"
            )
        elif self.mapping_file_path:
            print(f"Warning: Mapping file not found: {self.mapping_file_path}")

        # Initialize CSV file if logging is enabled
        if self.enable_logging:
            self._initialize_csv()

    def _initialize_csv(self) -> None:
        """Initialize CSV file with headers."""
        self.csv_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            headers = ["epoch", "shuffle_occurred"]
            for entity in self.resource_entities:
                headers.append(f"{entity}_appears_as")
            writer.writerow(headers)

    def shuffle_appearances(self) -> Dict[str, str]:
        """Shuffle entity appearances based on constraint or loaded mappings.

        Returns:
            New mapping dictionary (entity -> visual appearance)
        """
        if self.loaded_mappings:
            # Use pre-generated mapping
            if self.current_mapping_index < len(self.loaded_mappings):
                self.current_mapping = self.loaded_mappings[self.current_mapping_index]
                self.current_mapping_index += 1
            else:
                # If we've used all mappings, cycle back to the beginning
                self.current_mapping_index = 0
                self.current_mapping = self.loaded_mappings[self.current_mapping_index]
                self.current_mapping_index += 1
        else:
            # Use random shuffling
            if self.shuffle_constraint == "no_fixed":
                self.current_mapping = self._shuffle_no_fixed_with_adjacent_diversity()
            elif self.shuffle_constraint == "allow_fixed":
                self.current_mapping = self._shuffle_allow_fixed()

        return self.current_mapping.copy()

    def _shuffle_no_fixed(self) -> Dict[str, str]:
        """Shuffle ensuring no entity maps to itself and all targets are unique."""
        max_attempts = 1000  # Prevent infinite loops

        for attempt in range(max_attempts):
            shuffled_appearances = self.resource_entities.copy()
            random.shuffle(shuffled_appearances)

            # Check if no entity maps to itself AND all targets are unique
            if not any(
                entity == shuffled_appearances[i]
                for i, entity in enumerate(self.resource_entities)
            ) and len(set(shuffled_appearances)) == len(shuffled_appearances):
                new_mapping = {}
                for i, entity in enumerate(self.resource_entities):
                    new_mapping[entity] = shuffled_appearances[i]
                return new_mapping

        # Fallback: if we can't find a valid mapping, use allow_fixed
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

    def _shuffle_no_fixed_with_adjacent_diversity(self) -> Dict[str, str]:
        """Shuffle ensuring no entity maps to itself, all targets unique, and no shared
        components with previous mapping."""
        max_attempts = 1000

        for attempt in range(max_attempts):
            shuffled_appearances = self.resource_entities.copy()
            random.shuffle(shuffled_appearances)

            # Check if no entity maps to itself AND all targets are unique
            if not any(
                entity == shuffled_appearances[i]
                for i, entity in enumerate(self.resource_entities)
            ) and len(set(shuffled_appearances)) == len(shuffled_appearances):

                # Check if this mapping shares any components with the previous mapping
                if self.previous_mapping is not None:
                    has_shared_components = False
                    for entity in self.resource_entities:
                        if (
                            shuffled_appearances[self.resource_entities.index(entity)]
                            == self.previous_mapping[entity]
                        ):
                            has_shared_components = True
                            break

                    if has_shared_components:
                        continue  # Try again

                new_mapping = {}
                for i, entity in enumerate(self.resource_entities):
                    new_mapping[entity] = shuffled_appearances[i]

                # Update previous mapping
                self.previous_mapping = new_mapping.copy()
                return new_mapping

        # Fallback: use regular no_fixed
        print(
            "Warning: Could not find adjacent-diverse no_fixed mapping, using regular no_fixed"
        )
        mapping = self._shuffle_no_fixed()
        self.previous_mapping = mapping.copy()
        return mapping

    def get_current_mapping(self) -> Dict[str, str]:
        """Get current appearance mapping."""
        return self.current_mapping.copy()

    def log_to_csv(self, epoch: int, shuffle_occurred: bool = False) -> None:
        """Log current mapping to CSV file."""
        if not self.enable_logging:
            return

        with open(self.csv_file_path, "a", newline="") as csvfile:
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
        new_csv_path = (
            self.csv_file_path.parent / f"{run_folder}_entity_appearances.csv"
        )
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

    def _load_mappings_from_file(self) -> List[Dict[str, str]]:
        """Load pre-generated mappings from file."""
        if not self.mapping_file_path or not self.mapping_file_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_file_path}")

        with open(self.mapping_file_path) as f:
            mapping_data = json.load(f)

        # Validate that the file matches current configuration
        metadata = mapping_data.get("metadata", {})
        file_entities = metadata.get("resource_entities", [])

        if file_entities != self.resource_entities:
            raise ValueError(
                f"Resource entities mismatch: file has {file_entities}, expected {self.resource_entities}"
            )

        file_constraint = metadata.get("shuffle_constraint", "unknown")
        if file_constraint != self.shuffle_constraint:
            print(
                f"Warning: Shuffle constraint mismatch: file has {file_constraint}, expected {self.shuffle_constraint}"
            )

        mappings = mapping_data.get("mappings", [])
        print(f"Loaded mapping metadata: {metadata.get('diversity_stats', {})}")

        return mappings

    def get_mapping_info(self) -> Dict[str, any]:
        """Get information about current mapping state.

        Returns:
            Dictionary with mapping information
        """
        info = {
            "using_loaded_mappings": len(self.loaded_mappings) > 0,
            "total_loaded_mappings": len(self.loaded_mappings),
            "current_mapping_index": self.current_mapping_index,
            "shuffle_constraint": self.shuffle_constraint,
        }

        if self.loaded_mappings:
            info["remaining_mappings"] = (
                len(self.loaded_mappings) - self.current_mapping_index
            )
            info["mapping_file"] = (
                str(self.mapping_file_path) if self.mapping_file_path else None
            )

        return info
