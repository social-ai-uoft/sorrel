# Stag Hunt in the Matrix

A gridworld game where two or more agents explore to find valuable items (stags, hares).

Stags require the cooperation of two agents to simultaneously attack in order to obtain them. Hares can be hunted by a single agent.

## Running the example

```bash
python sorrel/examples/staghunt/main.py
```

The script runs the experiment for the default number of epochs (1000) with a maximum of 100 turns per epoch. TensorBoard logs, model checkpoints, and animated GIFs are saved under `data/`.

## Project structure

```
staghunt/ 
├─ assets/                    # Sprite images for game entities
├─ data/                      # Generated during runs: logs, checkpoints, GIFs
│   ├─ checkpoints/
│   ├─ gifs/
│   └─ logs/
├─ agents.py                  # Agent implementation
├─ custom_observation_spec.py # Custom observation spec
├─ entities.py                # Entity definitions
├─ env.py                     # Environment wrapper and setup
├─ main.py                    # Experiment entry point
├─ metrics_collector.py       # Metrics collector
└─ world.py                   # Gridworld definition (StagHuntWorld)
```