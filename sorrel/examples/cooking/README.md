# Cooking

A simple multi-agent collaborative planning and goal achievement experiment inspired by the game *Overcooked*. 

## Running the example

```bash
python sorrel/examples/cooking/main.py
```

The script runs the experiment for the default number of epochs (5000) with a maximum of 100 turns per epoch. TensorBoard logs, model checkpoints, and animated GIFs are saved under `data/`.

## Project structure

```
cooking/ 
├─ assets/                # Sprite images for walls, sand, decks and the agent
├─ configs/                # Configuration files
├─ data/                  # Generated during runs: logs, checkpoints, GIFs
│   ├─ checkpoints/
│   ├─ gifs/
│   └─ logs/
├─ agents.py              # Agent implementation
├─ entities.py            # Cooking game entity definition
├─ env.py                 # Environment wrapper and setup
├─ main.py                # Experiment entry point
└─ world.py               # Gridworld implementation
```