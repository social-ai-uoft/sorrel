"""Package for the Stag Hunt environment built on Sorrel.

This package exposes the key components required to instantiate and run
the stag hunt arena, namely the :class:`StagHuntWorld`,
:class:`StagHuntEnv`, :class:`StagHuntAgent` and the entity classes
defined in :mod:`staghunt.entities`.  See the module‑level
documentation within each file and the accompanying design specification
for further details.
"""

from sorrel.examples.staghunt.agents import StagHuntAgent  # noqa: F401
from sorrel.examples.staghunt.entities import (  # noqa: F401
    Empty,
    HareResource,
    Spawn,
    StagResource,
    Wall,
)
from sorrel.examples.staghunt.env import StagHuntEnv  # noqa: F401
from sorrel.examples.staghunt.world import StagHuntWorld  # noqa: F401
