# Tag

A game of tag between an agent who is It and one or more agents who is Not It. 

## Running the example

```bash
python sorrel/examples/tag/main.py
```

The script runs the experiment for the default number of epochs (5000) with a maximum of 100 turns per epoch. TensorBoard logs, model checkpoints, and animated GIFs are saved under `data/`.

## Project structure

```
tag/ 
├─ assets/                # Sprite images for agents and other entities
├─ data/                  # Generated during runs: logs, checkpoints, GIFs
│   ├─ checkpoints/
│   ├─ gifs/
│   └─ logs/
├─ agents.py              # Agent implementation
├─ env.py                 # Environment wrapper and setup
├─ main.py                # Experiment entry point
```