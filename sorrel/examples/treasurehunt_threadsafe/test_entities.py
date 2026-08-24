"""Tests for treasurehunt_threadsafe entities."""

from pathlib import Path

from sorrel.examples.treasurehunt_threadsafe.entities import Wall


def test_wall_uses_this_examples_own_sprite():
    """Regression test: a prior dedup switched Wall to import
    sorrel.entities.basic_entities.Wall directly, which orphaned this example's own
    assets/wall.png in favor of the shared generic sprite."""
    expected = Path(__file__).parent / "assets" / "wall.png"
    assert Wall().sprite == expected
