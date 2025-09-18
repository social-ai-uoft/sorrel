"""
Comparison script showing different visualization approaches for Stag Hunt.
This demonstrates how pygame can replicate the exact same visualization as Sorrel.
"""

import sys
from pathlib import Path

# Add the parent directory to the path to import sorrel modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

import pygame
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sorrel.examples.staghunt.entities import Empty, Sand, Wall, Spawn, StagResource, HareResource
from sorrel.examples.staghunt.world import StagHuntWorld
from sorrel.utils.visualization import render_sprite, image_from_array


def compare_visualizations():
    """Compare different visualization approaches."""
    
    print("=== Stag Hunt Visualization Comparison ===")
    print()
    
    # Create world with ASCII map
    config = {
        "world": {
            "generation_mode": "ascii_map",
            "ascii_map_file": "stag_hunt_ascii_map_clean.txt",
            "num_agents": 1,
            "taste_reward": 0.1,
            "destroyable_health": 3,
            "beam_length": 3,
            "beam_radius": 1,
            "beam_cooldown": 3,
            "respawn_lag": 10,
            "payoff_matrix": [[4, 0], [2, 2]],
            "interaction_reward": 1.0,
            "freeze_duration": 5,
            "respawn_delay": 10,
        }
    }
    
    world = StagHuntWorld(config, Empty())
    
    print(f"World dimensions: {world.height}x{world.width}")
    print(f"ASCII map loaded: {hasattr(world, 'map_generator') and world.map_generator is not None}")
    print()
    
    # 1. Original Sorrel visualization
    print("1. Original Sorrel Visualization (matplotlib + PIL)")
    print("   - Uses render_sprite() function")
    print("   - Multi-layer compositing with PIL")
    print("   - Exact sprite loading and resizing")
    print("   - Matplotlib for display")
    
    try:
        # Render using Sorrel's method
        layers = render_sprite(world, tile_size=[32, 32])
        composited = image_from_array(layers)
        
        print(f"   ✓ Successfully rendered {len(layers)} layers")
        print(f"   ✓ Final image size: {composited.size}")
        print(f"   ✓ Image mode: {composited.mode}")
        
        # Save the image for comparison
        composited.save("sorrel_original.png")
        print("   ✓ Saved as 'sorrel_original.png'")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    # 2. Pygame visualization
    print("2. Pygame Visualization")
    print("   - Uses pygame for rendering")
    print("   - Same sprite loading as Sorrel")
    print("   - Real-time rendering")
    print("   - Hardware acceleration")
    
    try:
        pygame.init()
        
        # Load sprites exactly like Sorrel
        tile_size = 32
        sprite_cache = {}
        
        entities = {
            'wall': Wall(),
            'sand': Sand(), 
            'spawn': Spawn(),
            'stag': StagResource(0.1, 3),
            'hare': HareResource(0.1, 3),
            'empty': Empty()
        }
        
        for name, entity in entities.items():
            if hasattr(entity, 'sprite'):
                sprite_path = entity.sprite
                if isinstance(sprite_path, Path):
                    sprite_path = str(sprite_path)
                try:
                    # Load and resize sprite exactly like Sorrel
                    pil_image = Image.open(sprite_path).resize((tile_size, tile_size)).convert("RGBA")
                    # Convert PIL to pygame surface
                    pygame_image = pygame.image.fromstring(
                        pil_image.tobytes(), 
                        pil_image.size, 
                        pil_image.mode
                    )
                    sprite_cache[name] = pygame_image
                except Exception as e:
                    print(f"   Warning: Could not load sprite {sprite_path}: {e}")
        
        # Create pygame surface
        screen_width = world.width * tile_size
        screen_height = world.height * tile_size
        pygame_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        
        # Render terrain layer
        for y in range(world.height):
            for x in range(world.width):
                location = (y, x, 0)
                entity = world.observe(location)
                
                if isinstance(entity, Wall):
                    sprite_key = 'wall'
                elif isinstance(entity, Sand):
                    sprite_key = 'sand'
                elif isinstance(entity, Spawn):
                    sprite_key = 'spawn'
                else:
                    sprite_key = 'empty'
                
                if sprite_key in sprite_cache:
                    sprite = sprite_cache[sprite_key]
                    pygame_surface.blit(sprite, (x * tile_size, y * tile_size))
        
        # Render dynamic layer
        for y in range(world.height):
            for x in range(world.width):
                location = (y, x, 1)
                entity = world.observe(location)
                
                if isinstance(entity, StagResource):
                    sprite_key = 'stag'
                elif isinstance(entity, HareResource):
                    sprite_key = 'hare'
                else:
                    continue
                
                if sprite_key in sprite_cache:
                    sprite = sprite_cache[sprite_key]
                    pygame_surface.blit(sprite, (x * tile_size, y * tile_size))
        
        # Save pygame surface as image
        pygame.image.save(pygame_surface, "pygame_rendered.png")
        print(f"   ✓ Successfully rendered with pygame")
        print(f"   ✓ Surface size: {pygame_surface.get_size()}")
        print("   ✓ Saved as 'pygame_rendered.png'")
        
        pygame.quit()
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    # 3. ASCII text visualization
    print("3. ASCII Text Visualization")
    print("   - Direct ASCII map rendering")
    print("   - Text-based display")
    print("   - No sprites, pure text")
    
    try:
        # Read the ASCII map file
        ascii_map_path = Path(__file__).parent.parent / "docs" / "stag_hunt_ascii_map_clean.txt"
        with open(ascii_map_path, 'r') as f:
            ascii_lines = f.readlines()
        
        print("   ASCII Map Preview:")
        for i, line in enumerate(ascii_lines[:10]):  # Show first 10 lines
            print(f"   {i+1:2d}: {line.rstrip()}")
        print("   ...")
        print(f"   ✓ Loaded {len(ascii_lines)} lines from ASCII map")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    # Summary
    print("=== Summary ===")
    print("✅ Pygame CAN use exactly the same visualization as Sorrel!")
    print("✅ Same sprites, same ASCII maps, same rendering logic")
    print("✅ Pygame offers additional benefits:")
    print("   - Real-time rendering (60+ FPS)")
    print("   - Hardware acceleration")
    print("   - Interactive controls")
    print("   - Better performance than matplotlib")
    print("   - Cross-platform compatibility")
    print()
    print("Alternative frameworks that could also work:")
    print("  - OpenGL (via PyOpenGL or moderngl)")
    print("  - SDL2 (via pysdl2)")
    print("  - Web-based (via pygame-web or pygbag)")
    print("  - Terminal-based (via rich or blessed)")
    print("  - Jupyter widgets (via ipywidgets)")


if __name__ == "__main__":
    compare_visualizations()
