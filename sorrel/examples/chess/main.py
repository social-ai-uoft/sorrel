"""Entry point for the Chess example.

The script mirrors the structure of the other example ``main.py`` files. It
creates a ``ChessWorld`` (8x8 board) and a ``ChessEnvironment`` with a minimal
configuration, then runs a short experiment using the random agents.
"""

from sorrel.examples.chess.env import ChessEnvironment
from sorrel.examples.chess.world import ChessWorld

if __name__ == "__main__":
    # Minimal configuration – only the experiment parameters are required for
    # ``Environment.run_experiment``.  Model‑specific parameters are not used by
    # the random model implementation.
    config = {
        "experiment": {
            "epochs": 3,
            "max_turns": 50,
            "record_period": 1,
        },
        "model": {},
        "world": {},
    }

    # Construct the world
    world = ChessWorld()

    # Construct the environment and run the experiment.
    env = ChessEnvironment(world, config)
    env.run_experiment()
