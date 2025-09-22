# Chess

This example demonstrates a minimal chess environment built on top of the Sorrel framework. It includes chess pieces, a chessboard, specifications for actions and observations within the chessboard, and an environment wrapper that runs an experiment.

Currently, due to the high complexity of the action space, only a random-action chess agent is implemented.

## Running the experiment

```bash
python sorrel/examples/chess/main.py
```

The script creates a minimal configuration that runs a few epochs of random play, recording the board state at each epoch. GIFs of the evolution are saved under `data/gifs/`.

## Project structure

The layout follows the conventions described in the top‑level `examples/README.md`:

```
chess/
├─ assets/              # Piece sprite images
├─ data/                # Generated during runs: stores model checkpoints, output logs, and gifs
├─ action_spec.py       # Action space definition
├─ agents.py            # Example agents
├─ entities.py          # Chess piece classes
├─ env.py               # Environment wrapper and setup
├─ main.py              # Experiment entry point
├─ observation_spec.py  # Observation specification
└─ world.py             # Chessboard implementation
```

> [!NOTE]
> Currently, pawn underpromotion is not implemented: all pawns reaching the 8th (for white) or 1st (for black) row are promoted to queens.