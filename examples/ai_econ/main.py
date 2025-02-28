# general imports
import argparse

# agentarium imports
from agentarium.config import load_config
from agentarium.utils.visualization import (animate, image_from_array,
                                            visual_field_sprite)
# imports from our example
from examples.ai_econ.env import EconEnv
from examples.ai_econ.utils import create_agents, create_models


def setup(cfg) -> EconEnv:
    """Set up all the whole environment and everything within."""

    # make the agents
    woodcutter_models, stonecutter_models, market_models = create_models(cfg)
    woodcutters, stonecutters, markets = create_agents(
        cfg, woodcutter_models, stonecutter_models, market_models
    )

    # make the environment
    return EconEnv(cfg, woodcutters, stonecutters, markets)


def run(env: EconEnv, cfg):
    """Run the experiment."""
    imgs = []
    total_seller_score = 0
    total_seller_loss = 0
    total_buyer_loss = 0
    for epoch in range(cfg.experiment.epochs + 1):
        # Reset the environment at the start of each epoch
        env.reset()
        for i in range(cfg.agent.seller.num):
            env.woodcutters[i].model.start_epoch_action(**locals())
            env.stonecutters[i].model.start_epoch_action(**locals())
        for i in range(cfg.agent.buyer.num):
            env.markets[i].model.start_epoch_action(**locals())

        while not env.turn >= env.max_turns:
            if epoch % cfg.experiment.record_period == 0:
                full_sprite = visual_field_sprite(env)
                imgs.append(image_from_array(full_sprite))

            env.take_turn()

        # At the end of each epoch, train as long as the batch size is large enough.
        if epoch > 10:
            for i in range(cfg.agent.seller.num):
                total_seller_loss += env.woodcutters[i].model.train_step()
                total_seller_loss += env.stonecutters[i].model.train_step()
            for i in range(cfg.agent.buyer.num):
                total_buyer_loss += env.markets[i].model.train_step()

        total_seller_score += env.seller_score
        current_seller_epsilon = env.woodcutters[0].model.epsilon

        if epoch % cfg.experiment.record_period == 0:
            avg_seller_score = total_seller_score / cfg.experiment.record_period
            print(
                f"Epoch: {epoch}; Epsilon: {current_seller_epsilon}; Losses this period: {total_seller_loss}; Avg. score this period: {avg_seller_score}"
            )
            animate(imgs, f"econ_epoch{epoch}", "./data/")
            # reset the data
            imgs = []
            total_score = 0
            total_loss = 0

        # update epsilon
        for i in range(cfg.agent.seller.num):
            new_epsilon = current_seller_epsilon - cfg.experiment.seller_epsilon_decay
            env.woodcutters[i].model.epsilon = max(new_epsilon, 0.01)
            env.stonecutters[i].model.epsilon = max(new_epsilon, 0.01)


if __name__ == "__main__":
    config = load_config(argparse.Namespace(config="./configs/config.yaml"))
    environment = setup(config)
    run(environment, config)
