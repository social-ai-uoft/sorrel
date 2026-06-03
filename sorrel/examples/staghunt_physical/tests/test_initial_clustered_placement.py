"""Unit tests for initial clustered resource placement (Tier 1)."""

import numpy as np
import pytest

from sorrel.examples.staghunt_physical.entities import Empty, HareResource, StagResource
from sorrel.examples.staghunt_physical.env import StagHuntEnv
from sorrel.examples.staghunt_physical.world import StagHuntWorld


def create_test_config(**world_overrides) -> dict:
    world = {
        "height": 11,
        "width": 11,
        "num_agents": 2,
        "num_agents_to_spawn": 2,
        "resource_density": 0.15,
        "stag_probability": 0.5,
        "stag_reward": 10,
        "hare_reward": 3,
        "stag_health": 3,
        "hare_health": 3,
        "random_agent_spawning": True,
        "random_resource_respawn": True,
        "resource_cap_mode": "disabled",
        **world_overrides,
    }
    return {
        "world": world,
        "model": {
            "layer_size": 64,
            "epsilon": 0.1,
            "epsilon_min": 0.01,
            "n_frames": 1,
            "n_step": 1,
            "sync_freq": 100,
            "model_update_freq": 4,
            "batch_size": 32,
            "memory_size": 512,
            "LR": 0.00025,
            "TAU": 0.001,
            "GAMMA": 0.99,
            "n_quantiles": 12,
            "device": "cpu",
        },
        "experiment": {"epochs": 2, "max_turns": 5},
    }


def test_default_mode_is_uniform():
    config = create_test_config()
    world = StagHuntWorld(config, Empty())

    assert world.initial_resource_placement_mode == "uniform"
    assert not hasattr(world, "initial_num_patches")


def test_invalid_placement_mode_raises():
    config = create_test_config(initial_resource_placement={"mode": "invalid"})
    with pytest.raises(ValueError):
        StagHuntWorld(config, Empty())


def test_compute_clustered_radius_zero_one_cell():
    config = create_test_config(
        initial_resource_placement={
            "mode": "clustered",
            "num_patches": 1,
            "patch_radius": 0.0,
            "fill_probability": 1.0,
        }
    )
    world = StagHuntWorld(config, Empty())
    np.random.seed(0)
    points = world.compute_clustered_resource_spawn_points()

    assert len(points) == 1
    y, x, _ = points[0]
    assert points[0] not in world.agent_spawn_points
    assert 1 <= y < world.height - 1
    assert 1 <= x < world.width - 1


def test_compute_clustered_excludes_agent_spawns():
    config = create_test_config(
        initial_resource_placement={
            "mode": "clustered",
            "num_patches": 1,
            "patch_radius": 3.0,
            "fill_probability": 1.0,
        }
    )
    world = StagHuntWorld(config, Empty())
    world.agent_spawn_points = [(5, 5, world.dynamic_layer)]
    np.random.seed(0)
    points = world.compute_clustered_resource_spawn_points()

    assert (5, 5, world.dynamic_layer) not in points


def test_clustered_populate_places_resources():
    config = create_test_config(
        initial_resource_placement={
            "mode": "clustered",
            "num_patches": 3,
            "patch_radius": 2.0,
            "fill_probability": 1.0,
        },
        stag_probability=1.0,
    )
    world = StagHuntWorld(config, Empty())
    env = StagHuntEnv(world, config)
    env.setup_agents()
    np.random.seed(1)
    env.populate_environment()

    assert world.count_resources() > 0
    assert len(world.resource_spawn_points) > 0
    for loc in world.resource_spawn_points:
        assert loc not in world.agent_spawn_points
        entity = world.observe(loc)
        assert isinstance(entity, (StagResource, HareResource))


def test_uniform_populate_smoke():
    config = create_test_config(resource_density=0.5)
    world = StagHuntWorld(config, Empty())
    env = StagHuntEnv(world, config)
    env.setup_agents()
    np.random.seed(42)
    env.populate_environment()

    assert len(world.resource_spawn_points) > 0
    assert world.count_resources() > 0
