# Cleanup

An adaptation of the Cleanup game from Melting Pot ([https://github.com/google-deepmind/meltingpot](google-deepmind/meltingpot)) in the Sorrel game engine.

## Running the example

```bash
python sorrel/examples/cleanup/main.py
```

The script runs the experiment for the default number of epochs (5000) with a maximum of 100 turns per epoch. TensorBoard logs, model checkpoints, and animated GIFs are saved under `data/`.

## Project structure

```
cleanup/ 
├─ assets/                # Sprite images for walls, sand, decks and the agent
├─ assets/                # Configuration files
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