"""Utility functions and classes for the Stag Hunt environment.

This module contains utility functions and classes that are specific to
the stag hunt environment, including the ASCII map parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


class ASCIIMapParser:
    """Parser for ASCII map files used in gridworld environments.
    
    This class can parse ASCII map files and extract structured information
    about walls, spawn points, resources, and other map features.
    """
    
    def __init__(self, map_file_path: str | Path) -> None:
        """Initialize the ASCII map parser.
        
        Args:
            map_file_path: Path to the ASCII map file to parse.
        """
        self.map_file_path = Path(map_file_path)
        self.spawn_points: List[Tuple[int, int, int]] = []
        self.resource_points: List[Tuple[int, int, int, str]] = []  # (y, x, layer, type)
        self.wall_points: List[Tuple[int, int, int]] = []
        self.random_resource_points: List[Tuple[int, int, int]] = []
        self.dimensions: Tuple[int, int] = (0, 0)
        self.raw_map: List[str] = []
        
    def parse(self) -> Dict[str, Any]:
        """Parse the ASCII map file and extract structured data.
        
        Returns:
            Dictionary containing parsed map data with keys:
            - 'spawn_points': List of (y, x, layer) tuples for agent spawn locations
            - 'resource_points': List of (y, x, layer, type) tuples for fixed resources
            - 'random_resource_points': List of (y, x, layer) tuples for random resource spawns
            - 'wall_points': List of (y, x, layer) tuples for wall locations
            - 'dimensions': (height, width) tuple of map dimensions
            - 'raw_map': List of strings representing the original ASCII map
        """
        if not self.map_file_path.exists():
            raise FileNotFoundError(f"ASCII map file not found: {self.map_file_path}")
            
        # Read the ASCII map file
        with open(self.map_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remove empty lines and strip whitespace
        self.raw_map = [line.rstrip('\n\r') for line in lines if line.strip()]
        
        # Find the actual map data (skip any header/legend lines)
        map_start = 0
        for i, line in enumerate(self.raw_map):
            if line and not line.startswith('Legend:') and not line.startswith('Stag Hunt'):
                # Check if this looks like a map line (contains W, P, 1, 2, a, or spaces)
                if any(char in line for char in 'WP12a '):
                    map_start = i
                    break
        
        # Extract map lines
        map_lines = self.raw_map[map_start:]
        
        # Find the end of the map (before legend starts)
        map_end = len(map_lines)
        for i, line in enumerate(map_lines):
            if line.startswith('Legend:') or line.startswith('W – Wall') or not line.strip():
                map_end = i
                break
        
        map_lines = map_lines[:map_end]
        
        if not map_lines:
            raise ValueError("No valid map data found in ASCII file")
        
        # Set dimensions
        self.dimensions = (len(map_lines), len(map_lines[0]))
        height, width = self.dimensions
        
        # Parse each line of the map
        for y, line in enumerate(map_lines):
            for x, char in enumerate(line):
                if char == 'W':
                    # Wall - place on terrain layer (0)
                    self.wall_points.append((y, x, 0))
                elif char == 'P':
                    # Spawn point - place on dynamic layer (1)
                    self.spawn_points.append((y, x, 1))
                elif char == '1':
                    # Stag resource - place on dynamic layer (1)
                    self.resource_points.append((y, x, 1, 'StagResource'))
                elif char == '2':
                    # Hare resource - place on dynamic layer (1)
                    self.resource_points.append((y, x, 1, 'HareResource'))
                elif char == 'a':
                    # Random resource spawn point - place on dynamic layer (1)
                    self.random_resource_points.append((y, x, 1))
                elif char == ' ':
                    # Empty space - no specific entity needed
                    pass
                else:
                    # Unknown character - treat as empty space
                    pass
        
        return {
            'spawn_points': self.spawn_points,
            'resource_points': self.resource_points,
            'random_resource_points': self.random_resource_points,
            'wall_points': self.wall_points,
            'dimensions': self.dimensions,
            'raw_map': self.raw_map
        }
    
    def get_spawn_points(self) -> List[Tuple[int, int, int]]:
        """Get the list of spawn points from the parsed map.
        
        Returns:
            List of (y, x, layer) tuples representing spawn locations.
        """
        return self.spawn_points.copy()
    
    def get_resource_points(self) -> List[Tuple[int, int, int, str]]:
        """Get the list of fixed resource points from the parsed map.
        
        Returns:
            List of (y, x, layer, type) tuples representing resource locations.
        """
        return self.resource_points.copy()
    
    def get_random_resource_points(self) -> List[Tuple[int, int, int]]:
        """Get the list of random resource spawn points from the parsed map.
        
        Returns:
            List of (y, x, layer) tuples representing random resource spawn locations.
        """
        return self.random_resource_points.copy()
    
    def get_wall_points(self) -> List[Tuple[int, int, int]]:
        """Get the list of wall points from the parsed map.
        
        Returns:
            List of (y, x, layer) tuples representing wall locations.
        """
        return self.wall_points.copy()
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the parsed map.
        
        Returns:
            (height, width) tuple representing map dimensions.
        """
        return self.dimensions
