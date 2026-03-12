# Iowa Gambling Task in the Matrix

This example implements a simplified matrix experiment inspired by the Iowa Gambling Task (IGT).

> Bechara, A., Damasio, A. R., Damasio, H., & Anderson, S. W. (1994). Insensitivity to future consequences following damage to human prefrontal cortex. *Cognition*, *50*, (1–3): 7–15. https://doi.org/10.1016/0010-0277(94)90018-3

The task is modelled as a gridworld in which an agent can move around and collect cards drawn from four decks (A–D). Decks A and B are “bad” decks (higher consistent rewards, but negative overall expected utility) while decks C and D are “good” decks (smaller consistent rewards, but positive overall expected utility).

## Running the example

```bash
python sorrel/examples/iowa/main.py
```

The script runs the experiment for the default number of epochs (5000) with a maximum of 100 turns per epoch. TensorBoard logs, model checkpoints, and animated GIFs are saved under `data/`.

## Project structure

```
iowa/
├─ assets/                # Sprite images for game entities
├─ data/                  # Generated during runs: logs, checkpoints, GIFs
│   ├─ checkpoints/
│   ├─ gifs/
│   └─ logs/
├─ agents.py              # Agent implementation
├─ entities.py            # Deck entity definitions
├─ env.py                 # Environment wrapper and setup
├─ main.py                # Experiment entry point
└─ world.py               # Gridworld definition (GamblingWorld)
```
