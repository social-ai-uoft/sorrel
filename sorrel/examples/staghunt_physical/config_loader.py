"""CSV-based agent configuration loader.

This module provides utilities for loading agent configurations from CSV files
and merging them with dictionary-based configurations.
"""

import csv
from pathlib import Path
from typing import Any


# CSV column to config field mapping
CSV_COLUMN_MAPPING = {
    "agent_id": "agent_id",  # Used as dict key
    "agent_group": "kind",   # Maps to "kind" in config
    "can_hunt_stag": "can_hunt",  # Maps to "can_hunt"
    "can_receive_shared_reward": "can_receive_shared_reward",  # Direct mapping
    "exclusive_reward": "exclusive_reward",  # Direct mapping
}

# Default values for optional attributes
DEFAULT_AGENT_ATTRIBUTES = {
    "can_hunt": True,
    "can_receive_shared_reward": True,
    "exclusive_reward": False,
}

# Required CSV columns
REQUIRED_COLUMNS = ["agent_id", "agent_group", "can_hunt_stag"]


def _convert_bool(value: str) -> bool:
    """Convert string to boolean.
    
    Handles common boolean representations: "True", "true", "1", "False", "false", "0"
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in ("true", "1", "yes", "on"):
            return True
        if value_lower in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"Cannot convert '{value}' to boolean")
    raise TypeError(f"Cannot convert {type(value)} to boolean")


def _convert_int(value: str) -> int:
    """Convert string to integer."""
    if isinstance(value, int):
        return value
    return int(value.strip())


def load_agent_config_from_csv(csv_path: str | Path) -> dict[int, dict[str, Any]]:
    """Load agent configuration from CSV file.
    
    Args:
        csv_path: Path to CSV file (relative paths resolved relative to caller)
    
    Returns:
        Dictionary mapping agent_id to agent configuration dict, compatible with
        existing agent_config format.
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid (missing columns, invalid data, etc.
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. "
            f"Please ensure the path is correct (relative paths are resolved relative to main.py)."
        )
    
    agent_config = {}
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Validate required columns exist
        if not reader.fieldnames:
            raise ValueError(f"CSV file {csv_path} is empty or has no header row")
        
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(
                f"CSV file {csv_path} is missing required columns: {missing_columns}. "
                f"Required columns: {REQUIRED_COLUMNS}. "
                f"Found columns: {list(reader.fieldnames)}"
            )
        
        # Process each row
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
            # Validate and convert agent_id
            try:
                agent_id = _convert_int(row["agent_id"])
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"Invalid agent_id in CSV file {csv_path} at row {row_num}: {row.get('agent_id', 'missing')}. "
                    f"agent_id must be a non-negative integer."
                ) from e
            
            if agent_id < 0:
                raise ValueError(
                    f"Invalid agent_id in CSV file {csv_path} at row {row_num}: {agent_id}. "
                    f"agent_id must be non-negative."
                )
            
            # Check for duplicate agent_id
            if agent_id in agent_config:
                raise ValueError(
                    f"Duplicate agent_id {agent_id} in CSV file {csv_path} at row {row_num}. "
                    f"Each agent_id must be unique."
                )
            
            # Validate and convert agent_group (kind)
            agent_group = row.get("agent_group", "").strip()
            if not agent_group:
                raise ValueError(
                    f"Empty agent_group in CSV file {csv_path} at row {row_num}. "
                    f"agent_group must be a non-empty string."
                )
            
            # Validate and convert can_hunt_stag
            try:
                can_hunt = _convert_bool(row["can_hunt_stag"])
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"Invalid can_hunt_stag in CSV file {csv_path} at row {row_num}: {row.get('can_hunt_stag', 'missing')}. "
                    f"can_hunt_stag must be a boolean (True/False, 1/0, yes/no)."
                ) from e
            
            # Build agent config dict
            agent_cfg = {
                "kind": agent_group,
                "can_hunt": can_hunt,
            }
            
            # Add optional columns if present
            if "can_receive_shared_reward" in row and row["can_receive_shared_reward"].strip():
                try:
                    agent_cfg["can_receive_shared_reward"] = _convert_bool(row["can_receive_shared_reward"])
                except ValueError as e:
                    raise ValueError(
                        f"Invalid can_receive_shared_reward in CSV file {csv_path} at row {row_num}: "
                        f"{row['can_receive_shared_reward']}. Must be a boolean."
                    ) from e
            else:
                agent_cfg["can_receive_shared_reward"] = DEFAULT_AGENT_ATTRIBUTES["can_receive_shared_reward"]
            
            if "exclusive_reward" in row and row["exclusive_reward"].strip():
                try:
                    agent_cfg["exclusive_reward"] = _convert_bool(row["exclusive_reward"])
                except ValueError as e:
                    raise ValueError(
                        f"Invalid exclusive_reward in CSV file {csv_path} at row {row_num}: "
                        f"{row['exclusive_reward']}. Must be a boolean."
                    ) from e
            else:
                agent_cfg["exclusive_reward"] = DEFAULT_AGENT_ATTRIBUTES["exclusive_reward"]
            
            agent_config[agent_id] = agent_cfg
    
    if not agent_config:
        raise ValueError(f"CSV file {csv_path} contains no agent data (only header row)")
    
    return agent_config


def merge_agent_configs(
    dict_config: dict[int, dict[str, Any]] | None,
    csv_config: dict[int, dict[str, Any]] | None
) -> dict[int, dict[str, Any]]:
    """Merge dictionary-based and CSV-based agent configurations.
    
    Args:
        dict_config: Dictionary-based agent config (from config dict in main.py)
        csv_config: CSV-based agent config (from load_agent_config_from_csv)
    
    Returns:
        Merged agent configuration. CSV config takes precedence for overlapping agent_ids.
    
    Raises:
        ValueError: If both configs are None or empty
    """
    if dict_config is None:
        dict_config = {}
    if csv_config is None:
        csv_config = {}
    
    if not dict_config and not csv_config:
        raise ValueError("Cannot merge: both dict_config and csv_config are empty")
    
    # Start with dict config, then override with CSV config
    merged = dict(dict_config)
    merged.update(csv_config)  # CSV takes precedence
    
    return merged







