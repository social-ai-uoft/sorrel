"""ASCII Map-based world generation for Ingroup Bias environment.

This module provides functionality to parse ASCII map files and generate world layouts
based on the map specifications. The parser follows the map layout for the ingroup bias
game with specific entity types for agents and resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


@dataclass
class IngroupBiasMapData:
    """Data structure containing parsed ASCII map information for ingroup bias game."""

    dimensions: Tuple[int, int]
    spawn_points: List[Tuple[int, int]]
    red_resource_locations: List[Tuple[int, int]]
    green_resource_locations: List[Tuple[int, int]]
    blue_resource_locations: List[Tuple[int, int]]
    wall_locations: List[Tuple[int, int]]
    empty_locations: List[Tuple[int, int]]


class IngroupBiasMapGenerator:
    """Parses ASCII map files and provides exact layout data for ingroup bias game.

    The parser follows the ASCII map format:
    - W: Wall (impassable)
    - P: Spawn point (agent starting location)
    - R: Red resource
    - G: Green resource
    - B: Blue resource
    - (space): Empty floor tile
    """

    def __init__(self, map_file_path: Union[str, Path]):
        """Initialize the map generator with a map file path.

        Args:
            map_file_path: Path to the ASCII map file
        """
        self.map_file_path = Path(map_file_path)
        self.raw_map = self._load_map()
        self.height, self.width = self._get_dimensions()

    def _load_map(self) -> List[str]:
        """Load the ASCII map file and return as list of strings.

        Returns:
            List of strings representing map rows

        Raises:
            FileNotFoundError: If map file doesn't exist
            ValueError: If map file is empty or invalid
        """
        # Construct the full path by combining docs directory with map file name
        docs_dir = Path(__file__).parent / "docs"
        full_map_path = docs_dir / self.map_file_path.name

        if not full_map_path.exists():
            raise FileNotFoundError(f"Map file not found: {full_map_path}")

        with open(full_map_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Filter out comment lines and empty lines, keep only map data
        map_lines = []
        for line in lines:
            line = line.rstrip()
            # Skip empty lines and comment lines (starting with non-map characters)
            if line and not line.startswith(
                ("Legend:", "W –", "P –", "R –", "G –", "B –", "(space)")
            ):
                map_lines.append(line)

        if not map_lines:
            raise ValueError(
                f"Map file is empty or contains no valid map data: {self.map_file_path}"
            )

        return map_lines

    def _get_dimensions(self) -> Tuple[int, int]:
        """Extract dimensions from the raw map.

        Returns:
            Tuple of (height, width)

        Raises:
            ValueError: If map has inconsistent row lengths
        """
        if not self.raw_map:
            raise ValueError("Cannot determine dimensions of empty map")

        height = len(self.raw_map)
        width = len(self.raw_map[0])

        # Validate all rows have the same width
        for i, row in enumerate(self.raw_map):
            if len(row) != width:
                raise ValueError(
                    f"Map row {i} has inconsistent width: {len(row)} vs {width}"
                )

        return height, width

    def parse_map(self) -> IngroupBiasMapData:
        """Parse ASCII map and return structured layout data.

        Returns:
            IngroupBiasMapData containing all parsed map information

        Raises:
            ValueError: If map contains invalid characters
        """
        spawn_points = []
        red_resource_locations = []
        green_resource_locations = []
        blue_resource_locations = []
        wall_locations = []
        empty_locations = []

        valid_chars = {"W", "P", "R", "G", "B", " "}

        for y, row in enumerate(self.raw_map):
            for x, char in enumerate(row):
                if char not in valid_chars:
                    raise ValueError(
                        f"Invalid character '{char}' at position ({y}, {x})"
                    )

                if char == "W":
                    wall_locations.append((y, x))
                elif char == "P":
                    spawn_points.append((y, x))
                elif char == "R":
                    red_resource_locations.append((y, x))
                elif char == "G":
                    green_resource_locations.append((y, x))
                elif char == "B":
                    blue_resource_locations.append((y, x))
                elif char == " ":
                    empty_locations.append((y, x))

        return IngroupBiasMapData(
            dimensions=(self.height, self.width),
            spawn_points=spawn_points,
            red_resource_locations=red_resource_locations,
            green_resource_locations=green_resource_locations,
            blue_resource_locations=blue_resource_locations,
            wall_locations=wall_locations,
            empty_locations=empty_locations,
        )

    def validate_map_for_agents(
        self, map_data: IngroupBiasMapData, num_agents: int
    ) -> None:
        """Validate that the map has sufficient spawn points for the number of agents.

        Args:
            map_data: Parsed map data
            num_agents: Number of agents that need to spawn

        Raises:
            ValueError: If insufficient spawn points
        """
        if len(map_data.spawn_points) < num_agents:
            raise ValueError(
                f"Map has {len(map_data.spawn_points)} spawn points but {num_agents} agents required. "
                f"Add more 'P' characters to the map or reduce the number of agents."
            )

    def check_overlaps(self, map_data: IngroupBiasMapData = None) -> dict:
        """Check for overlaps between different location types.

        Args:
            map_data: Optional pre-parsed map data. If None, will parse the map.

        Returns:
            Dictionary containing overlap information with detailed reports
        """
        if map_data is None:
            map_data = self.parse_map()

        # Combine all resource locations for comparison
        all_resource_locations = (
            map_data.red_resource_locations
            + map_data.green_resource_locations
            + map_data.blue_resource_locations
        )

        # Check all possible overlaps
        overlaps = {
            "spawn_vs_walls": self._find_overlaps(
                map_data.spawn_points, map_data.wall_locations
            ),
            "spawn_vs_resources": self._find_overlaps(
                map_data.spawn_points, all_resource_locations
            ),
            "spawn_vs_empty": self._find_overlaps(
                map_data.spawn_points, map_data.empty_locations
            ),
            "walls_vs_resources": self._find_overlaps(
                map_data.wall_locations, all_resource_locations
            ),
            "walls_vs_empty": self._find_overlaps(
                map_data.wall_locations, map_data.empty_locations
            ),
            "resources_vs_empty": self._find_overlaps(
                all_resource_locations, map_data.empty_locations
            ),
            "spawn_vs_spawn": self._find_duplicates(map_data.spawn_points),
            "walls_vs_walls": self._find_duplicates(map_data.wall_locations),
            "resources_vs_resources": self._find_duplicates(all_resource_locations),
            "empty_vs_empty": self._find_duplicates(map_data.empty_locations),
        }

        # Calculate summary statistics
        total_overlaps = sum(len(overlap_list) for overlap_list in overlaps.values())
        has_overlaps = total_overlaps > 0

        return {
            "has_overlaps": has_overlaps,
            "total_overlaps": total_overlaps,
            "overlap_details": overlaps,
            "summary": self._generate_overlap_summary(overlaps),
        }

    def _find_overlaps(
        self, list1: List[Tuple[int, int]], list2: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Find coordinates that appear in both lists.

        Args:
            list1: First list of (y, x) coordinates
            list2: Second list of (y, x) coordinates

        Returns:
            List of overlapping coordinates
        """
        set1 = set(list1)
        set2 = set(list2)
        return list(set1.intersection(set2))

    def _find_duplicates(
        self, coord_list: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Find duplicate coordinates within a single list.

        Args:
            coord_list: List of (y, x) coordinates

        Returns:
            List of duplicate coordinates
        """
        seen = set()
        duplicates = set()
        for coord in coord_list:
            if coord in seen:
                duplicates.add(coord)
            else:
                seen.add(coord)
        return list(duplicates)

    def _generate_overlap_summary(self, overlaps: dict) -> str:
        """Generate a human-readable summary of overlaps.

        Args:
            overlaps: Dictionary of overlap results

        Returns:
            Summary string describing all overlaps found
        """
        summary_parts = []

        for overlap_type, overlap_list in overlaps.items():
            if overlap_list:
                if "vs" in overlap_type:
                    type1, type2 = overlap_type.split("_vs_")
                    summary_parts.append(
                        f"{type1} vs {type2}: {len(overlap_list)} overlaps at {overlap_list}"
                    )
                else:
                    summary_parts.append(
                        f"Duplicate {overlap_type}: {len(overlap_list)} duplicates at {overlap_list}"
                    )

        if not summary_parts:
            return "No overlaps found - all location types are properly separated."

        return "Overlaps found:\n" + "\n".join(f"- {part}" for part in summary_parts)

    def validate_no_overlaps(self, map_data: IngroupBiasMapData = None) -> None:
        """Validate that there are no overlaps between location types.

        Args:
            map_data: Optional pre-parsed map data. If None, will parse the map.

        Raises:
            ValueError: If overlaps are found between location types
        """
        overlap_info = self.check_overlaps(map_data)

        if overlap_info["has_overlaps"]:
            raise ValueError(f"Map contains overlaps:\n{overlap_info['summary']}")

    def get_map_info(self) -> dict:
        """Get basic information about the loaded map.

        Returns:
            Dictionary with map statistics
        """
        map_data = self.parse_map()
        return {
            "dimensions": map_data.dimensions,
            "spawn_points": len(map_data.spawn_points),
            "walls": len(map_data.wall_locations),
            "red_resources": len(map_data.red_resource_locations),
            "green_resources": len(map_data.green_resource_locations),
            "blue_resources": len(map_data.blue_resource_locations),
            "total_resources": (
                len(map_data.red_resource_locations)
                + len(map_data.green_resource_locations)
                + len(map_data.blue_resource_locations)
            ),
            "empty_spaces": len(map_data.empty_locations),
        }
