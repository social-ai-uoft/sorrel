#!/usr/bin/env python3
"""
Extract reward values defined in the treasurehunt game code.

This script extracts the actual reward values for different entities:
- Gem reward value (configurable, default: 10.0)
- Wall penalty (hardcoded: -1)
- Empty/Sand reward (0, no value)
"""

import argparse
import re
from pathlib import Path


def extract_reward_values_from_code():
    """Extract reward values by reading the source code.
    
    Returns:
        Dictionary with reward values for each entity type
    """
    rewards = {}
    
    # Read entities.py
    entities_file = Path(__file__).parent / "entities.py"
    if not entities_file.exists():
        print(f"Error: Could not find entities.py at {entities_file}")
        return rewards
    
    with open(entities_file, 'r') as f:
        entities_code = f.read()
    
    # Extract Wall value
    wall_match = re.search(r'self\.value\s*=\s*(-?\d+)', entities_code)
    if wall_match:
        wall_value = int(wall_match.group(1))
        rewards['Wall'] = {
            'value': wall_value,
            'description': 'Penalty for hitting a wall',
            'configurable': False,
            'hardcoded': wall_value,
            'source': 'entities.py: Wall.__init__() - self.value = -1'
        }
    
    # Extract Gem value (it's passed as parameter)
    gem_match = re.search(r'def __init__\(self, gem_value\):', entities_code)
    if gem_match:
        gem_value_match = re.search(r'self\.value\s*=\s*gem_value', entities_code)
        if gem_value_match:
            rewards['Gem'] = {
                'value': 'configurable (default: 10.0)',
                'description': 'Reward for collecting a gem',
                'configurable': True,
                'default': 10.0,
                'source': 'entities.py: Gem.__init__(gem_value) - self.value = gem_value'
            }
    
    # Check Sand
    sand_match = re.search(r'class Sand\(', entities_code)
    if sand_match:
        # Check if Sand has value attribute
        sand_section = entities_code[sand_match.end():entities_code.find('class Gem', sand_match.end())]
        if 'self.value' in sand_section:
            sand_value_match = re.search(r'self\.value\s*=\s*(-?\d+)', sand_section)
            if sand_value_match:
                sand_value = int(sand_value_match.group(1))
                rewards['Sand'] = {
                    'value': sand_value,
                    'description': 'Reward for stepping on sand',
                    'configurable': False,
                    'source': 'entities.py: Sand.__init__()'
                }
        else:
            rewards['Sand'] = {
                'value': 0,
                'description': 'No reward (sand has no value attribute)',
                'configurable': False,
                'source': 'entities.py: Sand.__init__() - no value attribute'
            }
    
    # Check EmptyEntity
    empty_match = re.search(r'class EmptyEntity\(', entities_code)
    if empty_match:
        empty_section = entities_code[empty_match.end():]
        if 'self.value' in empty_section:
            empty_value_match = re.search(r'self\.value\s*=\s*(-?\d+)', empty_section)
            if empty_value_match:
                empty_value = int(empty_value_match.group(1))
                rewards['EmptyEntity'] = {
                    'value': empty_value,
                    'description': 'Reward for stepping on empty space',
                    'configurable': False,
                    'source': 'entities.py: EmptyEntity.__init__()'
                }
        else:
            rewards['EmptyEntity'] = {
                'value': 0,
                'description': 'No reward (empty space has no value attribute)',
                'configurable': False,
                'source': 'entities.py: EmptyEntity.__init__() - no value attribute'
            }
    
    # Read main.py to get default gem_value
    main_file = Path(__file__).parent / "main.py"
    if main_file.exists():
        with open(main_file, 'r') as f:
            main_code = f.read()
        
        gem_value_match = re.search(r'--gem_value.*?default=([\d.]+)', main_code)
        if gem_value_match:
            default_gem_value = float(gem_value_match.group(1))
            if 'Gem' in rewards:
                rewards['Gem']['default'] = default_gem_value
                rewards['Gem']['value'] = f"configurable (default: {default_gem_value})"
    
    return rewards


def print_reward_summary(rewards, output_file=None):
    """Print a summary of reward values.
    
    Args:
        rewards: Dictionary of reward values
        output_file: Optional file to write output to
    """
    lines = []
    lines.append("="*80)
    lines.append("TREASUREHUNT GAME REWARD VALUES")
    lines.append("="*80)
    lines.append("")
    lines.append("This document lists all reward values defined in the treasurehunt game code.")
    lines.append("")
    
    for entity_name, reward_info in rewards.items():
        lines.append("-"*80)
        lines.append(f"Entity: {entity_name}")
        lines.append("-"*80)
        lines.append(f"  Reward Value: {reward_info['value']}")
        lines.append(f"  Description: {reward_info['description']}")
        if 'configurable' in reward_info:
            lines.append(f"  Configurable: {reward_info['configurable']}")
        if 'default' in reward_info:
            lines.append(f"  Default Value: {reward_info['default']}")
        if 'hardcoded' in reward_info:
            lines.append(f"  Hardcoded Value: {reward_info['hardcoded']}")
        lines.append(f"  Source: {reward_info['source']}")
        lines.append("")
    
    lines.append("="*80)
    lines.append("SUMMARY")
    lines.append("="*80)
    lines.append("")
    lines.append("Positive Rewards:")
    has_positive = False
    for entity_name, reward_info in rewards.items():
        if isinstance(reward_info['value'], (int, float)) and reward_info['value'] > 0:
            lines.append(f"  {entity_name}: +{reward_info['value']}")
            has_positive = True
        elif 'default' in reward_info and reward_info['default'] > 0:
            lines.append(f"  {entity_name}: +{reward_info['default']} (configurable, default)")
            has_positive = True
    if not has_positive:
        lines.append("  None (all positive rewards are configurable)")
    
    lines.append("")
    lines.append("Negative Rewards (Penalties):")
    has_negative = False
    for entity_name, reward_info in rewards.items():
        if isinstance(reward_info['value'], (int, float)) and reward_info['value'] < 0:
            lines.append(f"  {entity_name}: {reward_info['value']}")
            has_negative = True
    if not has_negative:
        lines.append("  None")
    
    lines.append("")
    lines.append("Zero/Negligible Rewards:")
    for entity_name, reward_info in rewards.items():
        if isinstance(reward_info['value'], (int, float)) and reward_info['value'] == 0:
            lines.append(f"  {entity_name}: {reward_info['value']}")
    
    lines.append("")
    lines.append("="*80)
    lines.append("CONFIGURATION")
    lines.append("="*80)
    lines.append("")
    lines.append("The gem reward value can be configured via:")
    lines.append("  - Command line: --gem_value <value> (default: 10.0)")
    lines.append("  - Config file: config.world.gem_value")
    lines.append("")
    lines.append("Wall penalty is hardcoded in entities.py and cannot be changed.")
    lines.append("")
    
    output = "\n".join(lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"✓ Reward values saved to: {output_file}")
    else:
        print(output)
    
    return output


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract reward values defined in the treasurehunt game code"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    # Extract reward values
    rewards = extract_reward_values_from_code()
    
    if not rewards:
        print("Error: Could not extract reward values from code")
        return
    
    # Print/save summary
    output_file = Path(args.output) if args.output else None
    print_reward_summary(rewards, output_file)


if __name__ == "__main__":
    main()
