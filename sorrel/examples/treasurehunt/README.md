# Treasure Hunt

A game where one or more agents explore to find valuable items (food, gems) and avoiding punishment items (bones). 

## Running the example

```bash
python sorrel/examples/treasurehunt/main.py
```

The script runs the experiment for the default number of epochs (1000) with a maximum of 100 turns per epoch. TensorBoard logs, model checkpoints, and animated GIFs are saved under `data/`.

## Project structure

```
treasurehunt/ 
├─ assets/                # Sprite images for game entities
├─ data/                  # Generated during runs: logs, checkpoints, GIFs
│   ├─ checkpoints/
│   ├─ gifs/
│   └─ logs/
├─ agents.py              # Agent implementation
├─ entities.py            # Entity definitions
├─ env.py                 # Environment wrapper and setup
├─ main.py                # Experiment entry point
└─ world.py               # Gridworld definition (TreasurehuntWorld)
```