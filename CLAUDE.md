# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sorrel** is a general-purpose multi-agent reinforcement learning (RL) framework in Python 3.12+. It acts as an "operating system" for RL processes, providing unified abstractions for environments, agents, models, observations, and actions.

## Commands

### Installation
```bash
poetry install                        # Basic install
poetry install --with dev,extras      # Full dev install (includes tensorboard, logging)
pre-commit install                    # Set up pre-commit hooks
```

### Testing
```bash
pytest                                # Run all tests (includes doctests)
pytest -v                             # Verbose output
pytest sorrel/path/to/test_file.py    # Run specific file
```

Tests use `--doctest-modules` by default (configured in `pyproject.toml`), so doctests embedded in source files are part of the test suite.

### Linting & Formatting
```bash
pre-commit run --all-files            # Run all hooks (black, isort, docformatter, pyupgrade, pyright)
```

### CLI
```bash
sorrel run <example-name> [--config <config-name>]   # Run a bundled example
sorrel show-logs <example-name>                       # Open TensorBoard for example logs
# Examples: cleanup, chess, cooking, iowa, staghunt, tag, taxi, treasurehunt
```

### Documentation
```bash
make html    # Build Sphinx docs
```

## Architecture

### Core Abstractions

**`Environment`** (`sorrel/environment.py`) — Top-level orchestrator. Manages epoch/turn cycles, logging, and visualization. Key methods: `setup_agents()`, `populate_environment()`, `run_experiment()`, `generate_memories()`.

**`World`** (`sorrel/worlds/`) — Container for entities on a grid or graph. `Gridworld` stores entities in a 3D numpy array `(height, width, layers)`. `NodeWorld` supports graph topologies.

**`Entity`** (`sorrel/entities/entity.py`) — Atomic game object placed in the world. Agents are special entities.

**`Agent`** (`sorrel/agents/agent.py`) — Abstract RL agent. Methods: `pov()` (observation), `get_action()`, `act()`, `is_done()`, `transition()`. Owns an `ObservationSpec`, `ActionSpec`, and `BaseModel`.

**`BaseModel`** (`sorrel/models/base_model.py`) — RL model abstraction. Methods: `take_action()`, `train_step()`, `epsilon_decay()`, `save()`. Manages a replay buffer. Subclasses live in `sorrel/models/pytorch/` and `sorrel/models/jax/`.

**`ObservationSpec` / `ActionSpec`** (`sorrel/observation/`, `sorrel/action/`) — Define how agents perceive the world and what actions are available. Actions are mapped as integers to human-readable strings.

**`Buffer`** (`sorrel/buffers.py`) — Experience replay buffer. Stores (state, action, reward, done) tuples; supports frame stacking via `n_frames`.

### Example Structure

Each example in `sorrel/examples/<name>/` follows this layout:
```
agents.py           # Custom Agent subclasses
entities.py         # Custom Entity types
env.py              # Experiment class (subclasses Environment)
world.py            # World layout and entity placement
main.py             # Hydra entry point
configs/            # Hydra YAML configs
data/               # Checkpoints, GIFs, TensorBoard logs (gitignored)
```

### Configuration

Examples use **Hydra** (`hydra-core`) for config management. Configs are YAML files under `examples/<name>/configs/`.

### Key Dependencies

- **PyTorch** and **JAX/Flax** — two supported model backends
- **Hydra** — config management for experiments
- **TensorBoard** — training logging (`sorrel/utils/logging.py`)
- **Pillow / matplotlib** — visualization and GIF rendering (`sorrel/utils/visualization.py`)
- **OpenAI SDK** — LLM-based agent support (`sorrel/models/llm.py`)
