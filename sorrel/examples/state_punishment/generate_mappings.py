#!/usr/bin/env python3
"""Standalone script for generating entity appearance mappings with position diversity."""

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def generate_random_mapping(entities: List[str], constraint: str) -> Dict[str, str]:
    """Generate a single random mapping based on constraint."""
    if constraint == "no_fixed":
        return _shuffle_no_fixed(entities)
    elif constraint == "allow_fixed":
        return _shuffle_allow_fixed(entities)
    else:
        raise ValueError(f"Unknown constraint: {constraint}")


def _shuffle_no_fixed(entities: List[str]) -> Dict[str, str]:
    """Shuffle ensuring no entity maps to itself and all targets are unique."""
    max_attempts = 1000
    
    for attempt in range(max_attempts):
        shuffled_appearances = entities.copy()
        random.shuffle(shuffled_appearances)
        
        # Check if no entity maps to itself AND all targets are unique
        if (not any(entity == shuffled_appearances[i] for i, entity in enumerate(entities)) and
            len(set(shuffled_appearances)) == len(shuffled_appearances)):
            new_mapping = {}
            for i, entity in enumerate(entities):
                new_mapping[entity] = shuffled_appearances[i]
            return new_mapping
    
    # Fallback: use allow_fixed if we can't find valid mapping
    print("Warning: Could not find no_fixed mapping, using allow_fixed")
    return _shuffle_allow_fixed(entities)


def _shuffle_allow_fixed(entities: List[str]) -> Dict[str, str]:
    """Shuffle allowing any mapping."""
    shuffled_appearances = entities.copy()
    random.shuffle(shuffled_appearances)
    
    new_mapping = {}
    for i, entity in enumerate(entities):
        new_mapping[entity] = shuffled_appearances[i]
    
    return new_mapping


def is_position_diverse(new_mapping: Dict[str, str], position_history: List[Dict[str, str]]) -> bool:
    """Check if new mapping avoids position conflicts with previous mappings."""
    for prev_mapping in position_history:
        for entity in new_mapping:
            if new_mapping[entity] == prev_mapping[entity]:
                return False  # Same entity in same position
    return True


def generate_mapping_with_diversity(
    entities: List[str], 
    constraint: str, 
    position_history: List[Dict[str, str]]
) -> Dict[str, str]:
    """Generate a single mapping avoiding position conflicts."""
    max_attempts = 1000
    
    # If we have many mappings already, increase attempts for harder cases
    if len(position_history) > len(entities) * 2:
        max_attempts = 5000
    
    for attempt in range(max_attempts):
        mapping = generate_random_mapping(entities, constraint)
        
        if is_position_diverse(mapping, position_history):
            return mapping
    
    # Fallback: return any valid mapping
    print(f"Warning: Could not find position-diverse mapping after {max_attempts} attempts")
    return generate_random_mapping(entities, constraint)


def generate_diverse_mappings(
    entities: List[str], 
    num_mappings: int, 
    constraint: str,
    ensure_diversity: bool = True
) -> List[Dict[str, str]]:
    """Generate mappings with optional position diversity."""
    mappings = []
    position_history = []
    
    # Validate and auto-adjust num_mappings for position diversity
    if ensure_diversity:
        max_possible = math.factorial(len(entities))
        if num_mappings > max_possible:
            print(f"Warning: Requested {num_mappings} diverse mappings, but maximum possible is {max_possible}.")
            print(f"Generating {max_possible} mappings instead.")
            num_mappings = max_possible
        elif num_mappings == max_possible:
            print(f"Generating maximum possible diverse mappings: {num_mappings}")
    
    print(f"Generating {num_mappings} mappings with constraint '{constraint}'...")
    if ensure_diversity:
        print("Ensuring position diversity across mappings...")
    
    for i in range(num_mappings):
        if ensure_diversity:
            mapping = generate_mapping_with_diversity(entities, constraint, position_history)
        else:
            # For no_fixed constraint, ensure adjacent mappings don't share components
            if constraint == "no_fixed" and mappings:
                mapping = generate_no_fixed_with_adjacent_diversity(entities, mappings[-1])
            else:
                mapping = generate_random_mapping(entities, constraint)
        
        mappings.append(mapping)
        position_history.append(mapping)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_mappings} mappings...")
    
    return mappings


def generate_no_fixed_with_adjacent_diversity(entities: List[str], previous_mapping: Dict[str, str]) -> Dict[str, str]:
    """Generate a no_fixed mapping that doesn't share any components with the previous mapping."""
    max_attempts = 1000
    
    for attempt in range(max_attempts):
        shuffled_appearances = entities.copy()
        random.shuffle(shuffled_appearances)
        
        # Check if no entity maps to itself AND all targets are unique
        if (not any(entity == shuffled_appearances[i] for i, entity in enumerate(entities)) and
            len(set(shuffled_appearances)) == len(shuffled_appearances)):
            
            # Check if this mapping shares any components with the previous mapping
            has_shared_components = False
            for entity in entities:
                if shuffled_appearances[entities.index(entity)] == previous_mapping[entity]:
                    has_shared_components = True
                    break
            
            if not has_shared_components:
                new_mapping = {}
                for i, entity in enumerate(entities):
                    new_mapping[entity] = shuffled_appearances[i]
                return new_mapping
    
    # Fallback: return any valid no_fixed mapping
    print("Warning: Could not find adjacent-diverse no_fixed mapping, using regular no_fixed")
    return _shuffle_no_fixed(entities)


def calculate_diversity_stats(mappings: List[Dict[str, str]]) -> Dict[str, float]:
    """Calculate diversity statistics for the generated mappings."""
    if not mappings:
        return {}
    
    entities = list(mappings[0].keys())
    total_positions = len(entities) * len(mappings)
    conflicts = 0
    
    # Count position conflicts
    for i in range(len(mappings)):
        for j in range(i + 1, len(mappings)):
            for entity in entities:
                if mappings[i][entity] == mappings[j][entity]:
                    conflicts += 1
    
    diversity_ratio = 1.0 - (conflicts / total_positions) if total_positions > 0 else 1.0
    
    return {
        "total_mappings": len(mappings),
        "total_positions": total_positions,
        "position_conflicts": conflicts,
        "diversity_ratio": diversity_ratio,
        "perfect_diversity": conflicts == 0
    }


def save_mappings_to_file(
    mappings: List[Dict[str, str]], 
    output_path: Path, 
    entities: List[str], 
    constraint: str,
    ensure_diversity: bool
) -> None:
    """Save mappings to JSON file with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    diversity_stats = calculate_diversity_stats(mappings)
    
    mapping_data = {
        "metadata": {
            "resource_entities": entities,
            "shuffle_constraint": constraint,
            "num_mappings": len(mappings),
            "ensure_position_diversity": ensure_diversity,
            "generated_at": datetime.now().isoformat(),
            "diversity_stats": diversity_stats
        },
        "mappings": mappings
    }
    
    with open(output_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"Saved {len(mappings)} mappings to: {output_path}")
    print(f"Diversity ratio: {diversity_stats['diversity_ratio']:.3f}")
    print(f"Position conflicts: {diversity_stats['position_conflicts']}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate entity appearance mappings")
    
    parser.add_argument(
        "--entities", 
        nargs="+", 
        default=["A", "B", "C", "D", "E"],
        help="Resource entities to shuffle (default: A B C D E)"
    )
    parser.add_argument(
        "--constraint", 
        type=str, 
        default="no_fixed",
        choices=["no_fixed", "allow_fixed"],
        help="Shuffling constraint (default: no_fixed)"
    )
    parser.add_argument(
        "--num_mappings", 
        type=int, 
        default=100,
        help="Number of mappings to generate (default: 100)"
    )
    parser.add_argument(
        "--ensure_position_diversity", 
        action="store_true",
        help="Ensure no entity appears in same position across mappings"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output file path for mappings"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducible generation"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Generate mappings
    mappings = generate_diverse_mappings(
        entities=args.entities,
        num_mappings=args.num_mappings,
        constraint=args.constraint,
        ensure_diversity=args.ensure_position_diversity
    )
    
    # Save to file
    output_path = Path(args.output)
    save_mappings_to_file(
        mappings=mappings,
        output_path=output_path,
        entities=args.entities,
        constraint=args.constraint,
        ensure_diversity=args.ensure_position_diversity
    )
    
    print("Mapping generation completed!")


if __name__ == "__main__":
    main()
