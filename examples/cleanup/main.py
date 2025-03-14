# ------------------------ #
# region: Imports          #
# general imports
import os
import sys
from datetime import datetime

import torch

# sorrel imports
from sorrel.action.action_spec import ActionSpec
from sorrel.config import Cfg, load_config
from sorrel.entities import Entity
from sorrel.models.pytorch import PyTorchIQN
from sorrel.utils.visualization import (animate, image_from_array,
                                            visual_field_sprite)
from examples.cleanup.agents import CleanupAgent, CleanupObservation
from examples.cleanup.env import Cleanup

# endregion                #
# ------------------------ #

# Experiment parameters
ENTITY_LIST = [
    "EmptyEntity",
    "Wall",
    "River",
    "Pollution",
    "AppleTree",
    "Apple",
    "CleanBeam",
    "ZapBeam",
    "CleanupAgent",
]
RECORD_PERIOD = 50  # how many epochs in each data recording period
EPSILON_DECAY = 0.0001


def setup(cfg: Cfg, **kwargs) -> Cleanup:
    """Set up the environment and everything within it."""

    agents = []
    # make the agents
    for _ in range(cfg.agent.agent.num):

        agent_vision_radius = cfg.agent.agent.obs.vision
        observation_spec = CleanupObservation(ENTITY_LIST, agent_vision_radius)
        action_spec = ActionSpec(["up", "down", "left", "right", "clean", "zap"])

        model = PyTorchIQN(
            input_size=observation_spec.input_size,
            action_space=action_spec.n_actions,
            seed=torch.random.seed(),
            num_frames=cfg.agent.agent.obs.num_frames,
            **cfg.model.iqn.parameters.to_dict(),
        )

        if "load_weights" in kwargs:
            model.load(file_path=kwargs.get("load_weights"))

        agents.append(CleanupAgent(
            observation_spec=observation_spec, 
            action_spec=action_spec, 
            model=model
        ))

    env = Cleanup(cfg, agents)
    return env


def run(env: Cleanup, **kwargs):
    """Run the experiment."""
    cfg: Cfg = env.cfg

    imgs = []
    total_score = 0
    total_loss = 0
    for epoch in range(cfg.experiment.epochs + 1):
        # Reset the environment at the start of each epoch
        env.reset()
        for agent in env.agents:
            agent.model.start_epoch_action(**locals())

        while not env.turn >= env.max_turns:
            if epoch % RECORD_PERIOD == 0:
                full_sprite = visual_field_sprite(env)
                imgs.append(image_from_array(full_sprite))

            env.take_turn()

        # At the end of each epoch, train as long as the batch size is large enough.
        if epoch > 10:
            for agent in env.agents:
                loss = agent.model.train_step()
                total_loss += loss

        total_score += env.game_score

        if epoch % RECORD_PERIOD == 0:
            avg_score = total_score / len(env.agents)
            print(
                f"Epoch: {epoch}; Epsilon: {env.agents[0].model.epsilon}; Losses this period: {total_loss}; Avg. score this period: {avg_score}"
            )
            animate(imgs, f"cleanup_epoch{epoch}", "./data/")
            # reset the data
            imgs = []
            total_score = 0
            total_loss = 0

        # update epsilon
        for agent in env.agents:
            new_epsilon = agent.model.epsilon - cfg.experiment.epsilon_decay
            agent.model.epsilon = max(new_epsilon, 0.01)

    if "save_weights" in kwargs:
        for i, agent in enumerate(env.agents):
            file_path = os.path.abspath(
                f'./checkpoints/{cfg.model.iqn.type}_{datetime.now().strftime("%Y%m%d-%H%m%s")}_{i}.pkl'
            )
            agent.model.save(file_path=file_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file",
        default="./configs/config.yaml",
    )
    parser.add_argument(
        "--load-weights", "-l", type=str, help="Path to pretrained model.", default=""
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()
    cfg = load_config(args)
    env = setup(
        cfg,
        # load_weights=args.load_weights,
    )
    run(env, save_weights=True)


if __name__ == "__main__":
    main()
