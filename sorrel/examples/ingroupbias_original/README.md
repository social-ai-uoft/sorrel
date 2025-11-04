# Ingroup Bias Game - Original Version

This directory contains the **original** ingroup bias game implementation before the latent factor enhancements.

## Contents

### Core Files
- `agents.py` - Original agent implementation
- `entities.py` - Original entity definitions (resources, walls, etc.)
- `env.py` - Original environment wrapper
- `world.py` - Original world implementation
- `main.py` - Original main script
- `sanity_checks.py` - Original sanity check tests
- `__init__.py` - Package initialization
- `ingroup_bias_game_desc.txt` - Game description

### Assets
- `assets/` - Original sprite files for agents, resources, and environment elements

### Configuration
- `configs/config.yaml` - Original configuration file

### Documentation
- `docs/Implementation plan for study.pdf` - Study implementation plan

## Usage

To run the original version:

```bash
cd sorrel/examples/ingroupbias_original
python main.py
```

## Differences from Enhanced Version

The enhanced version (`../ingroupbias/`) includes:
- Latent factor-based icon generation system
- Enhanced agents with group membership
- Enhanced resources with latent factors
- Enhanced environment with group-aware spawning
- Additional test scripts and generated icons

This original version preserves the baseline implementation for comparison and reference.
