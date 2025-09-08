"""Package for the Stag Hunt environment built on Sorrel.

This package exposes the key components required to instantiate and run
the stag hunt arena, namely the :class:`StagHuntWorld`,
:class:`StagHuntEnv`, :class:`StagHuntAgent` and the entity classes
defined in :mod:`staghunt.entities`.  See the module‑level
documentation within each file and the accompanying design specification
for further details.
"""

from .world import StagHuntWorld  # noqa: F401
from .env import StagHuntEnv  # noqa: F401
from .agents import StagHuntAgent  # noqa: F401
from .entities import Wall, Empty, Spawn, StagResource, HareResource  # noqa: F401