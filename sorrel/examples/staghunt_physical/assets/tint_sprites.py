#!/usr/bin/env python3
"""Command-line script to generate tinted sprites for different agent kinds.

Usage:
    python tint_sprites.py --kind AgentKindA --multiplier 1.2 --kind AgentKindB --multiplier 0.8
    python tint_sprites.py -k AgentKindA -m 1.2 -k AgentKindB -m 0.8
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def tint_sprite(sprite_path: Path, multiplier: float) -> np.ndarray:
    """Tint a sprite by multiplying only the red (R) channel.
    
    - r_final = r_original * multiplier
    - g_final = g_original (unchanged)
    - b_final = b_original (unchanged)
    
    Args:
        sprite_path: Path to the original sprite image
        multiplier: Multiplier for red channel values
    
    Returns:
        Numpy array with tinted RGBA image
    """
    # Load the image
    img = Image.open(sprite_path).convert("RGBA")
    
    # Convert to numpy array for easy manipulation
    img_array = np.array(img, dtype=np.float32)
    
    # Extract RGB channels (first 3 channels)
    rgb = img_array[:, :, :3].copy()  # Make a copy to avoid modifying original
    alpha = img_array[:, :, 3:4] if img_array.shape[2] == 4 else None
    
    # Red channel: multiply and clip to [0, 255]
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] * multiplier, 0, 255)
    # Green and blue channels remain unchanged
    
    # Convert to uint8
    tinted_rgb = rgb.astype(np.uint8)
    
    # Reconstruct image
    if alpha is not None:
        tinted_array = np.concatenate([tinted_rgb, alpha.astype(np.uint8)], axis=2)
    else:
        tinted_array = tinted_rgb
    
    return tinted_array


def generate_tinted_sprites(
    base_dir: Path,
    kind: str,
    multiplier: float,
    base_sprites: dict[str, str]
) -> None:
    """Generate tinted sprites for a given agent kind.
    
    Args:
        base_dir: Directory containing base sprites
        kind: Agent kind name (e.g., "AgentKindA")
        multiplier: Red channel multiplier value
        base_sprites: Dictionary mapping orientation names to base sprite filenames
    """
    print(f"Generating tinted sprites for {kind} with multiplier {multiplier} (red channel only)...")
    
    for orientation_name, base_filename in base_sprites.items():
        base_path = base_dir / base_filename
        
        if not base_path.exists():
            print(f"  Warning: Base sprite {base_filename} not found, skipping...")
            continue
        
        # Generate tinted sprite
        tinted_array = tint_sprite(base_path, multiplier)
        
        # Determine output filename
        if orientation_name == "south":  # Default orientation (no suffix)
            output_filename = f"hero_{kind}.png"
        else:
            output_filename = f"hero_{kind}_{orientation_name}.png"
        
        output_path = base_dir / output_filename
        
        # Convert back to PIL Image and save
        tinted_img = Image.fromarray(tinted_array, mode="RGBA")
        tinted_img.save(output_path)
        print(f"  Created: {output_filename}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate tinted sprites for different agent kinds by multiplying red channel: r_final = r * multiplier, g and b unchanged.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single kind
  python tint_sprites.py --kind AgentKindA --multiplier 1.2
  
  # Multiple kinds
  python tint_sprites.py --kind AgentKindA --multiplier 1.2 --kind AgentKindB --multiplier 0.8
  
  # Using short flags
  python tint_sprites.py -k AgentKindA -m 1.2 -k AgentKindB -m 0.8
        """
    )
    
    parser.add_argument(
        "--kind", "-k",
        action="append",
        dest="kinds",
        required=True,
        help="Agent kind name (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--multiplier", "-m",
        action="append",
        type=float,
        dest="multipliers",
        required=True,
        help="Red channel multiplier value (can be specified multiple times, must match number of --kind arguments)"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command-line arguments."""
    if len(args.kinds) != len(args.multipliers):
        print("Error: Number of --kind arguments must match number of --multiplier arguments", file=sys.stderr)
        sys.exit(1)
    
    for kind in args.kinds:
        if not kind or not kind.strip():
            print(f"Error: Invalid kind name: '{kind}'", file=sys.stderr)
            sys.exit(1)
    
    for multiplier in args.multipliers:
        if multiplier <= 0:
            print(f"Error: Multiplier must be positive, got: {multiplier}", file=sys.stderr)
            sys.exit(1)
        if multiplier > 5.0:
            print(f"Warning: Multiplier {multiplier} is very large (>5.0), results may be oversaturated")


def main():
    """Main entry point."""
    args = parse_arguments()
    validate_arguments(args)
    
    # Get the directory where this script is located (assets directory)
    script_dir = Path(__file__).parent
    base_dir = script_dir
    
    # Define base sprites and their orientation names
    base_sprites = {
        "south": "hero.png",      # Default (south/down)
        "back": "hero-back.png",   # North (up)
        "right": "hero-right.png", # East
        "left": "hero-left.png",   # West
    }
    
    # Check that base sprites exist
    missing_sprites = [name for name, filename in base_sprites.items() 
                      if not (base_dir / filename).exists()]
    if missing_sprites:
        print(f"Error: Missing base sprites: {[base_sprites[name] for name in missing_sprites]}", file=sys.stderr)
        print(f"Please ensure all base sprites exist in: {base_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Generate tinted sprites for each kind-multiplier pair
    for kind, multiplier in zip(args.kinds, args.multipliers):
        generate_tinted_sprites(base_dir, kind, multiplier, base_sprites)
    
    print("\nDone! Tinted sprites have been generated.")


if __name__ == "__main__":
    main()

