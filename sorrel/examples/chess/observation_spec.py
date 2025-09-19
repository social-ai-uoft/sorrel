"""Observation specifications for the Chess example.

This module defines a custom observation spec that extends the generic
:class:`~sorrel.observation.observation_spec.OneHotObservationSpec` by
adding an additional channel indicating the colour of a piece.
"""

from __future__ import annotations

import numpy as np

from sorrel.observation.observation_spec import OneHotObservationSpec


class ChessOneHotObservationSpec(OneHotObservationSpec):
    """One-hot observation spec with an extra colour layer.

    The base class encodes each entity kind as a separate one-hot channel.
    This subclass appends a single additional channel that is ``1`` where the
    observed entity has ``colour == "white"``, ``-1`` where the
    observed entity has ``colour == "black"``, and ``0`` otherwise.  Empty
    squares (which have no ``colour`` attribute) are encoded as ``0``.
    """

    def observe(self, world, location: tuple | None = None) -> np.ndarray:  # type: ignore[override]
        # Get the standard one‑hot encoding from the superclass.
        base_obs = super().observe(world, location)
        # ``base_obs`` shape: (C, H, W) for full‑view.
        # Create colour layer with shape (1, H, W).
        colour_layer = np.zeros((1, *base_obs.shape[1:]), dtype=base_obs.dtype)
        # Iterate over all board coordinates (assuming a single layer gridworld).
        # ``world`` is a ``Gridworld``; we can use ``world.observe`` to get the entity.
        height, width = base_obs.shape[1], base_obs.shape[2]
        for i in range(height):
            for j in range(width):
                # location tuple includes layer index (0 for the only layer).
                entity = world.observe((i, j, 0))
                # Some entities (e.g., EmptySquare) do not have a ``colour`` attribute.
                if hasattr(entity, "colour") and getattr(entity, "colour") == "white":
                    colour_layer[0, i, j] = 1
                elif hasattr(entity, "colour") and getattr(entity, "colour") == "black":
                    colour_layer[0, i, j] = -1
        # Concatenate the colour channel to the existing channels.
        return np.concatenate([base_obs, colour_layer], axis=0)
