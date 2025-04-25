# general imports
import argparse

# sorrel imports
from sorrel.config import load_config, Cfg
from sorrel.utils.visualization import (animate, image_from_array,
                                            visual_field_sprite)
# imports from our example
from examples.ai_econ.env import EconEnv
import examples.ai_econ.env 
from examples.ai_econ.utils import create_agents, create_models


def setup(cfg: Cfg) -> EconEnv:
    """Set up all the whole environment and everything within."""

    # make the agents
    woodcutter_models, stonecutter_models, market_models = create_models(cfg)
    woodcutters, stonecutters, markets = create_agents(
        cfg, woodcutter_models, stonecutter_models, market_models
    )

    # make the environment
    return EconEnv(cfg, woodcutters, stonecutters, markets)


def run(env: EconEnv, cfg: Cfg):
    """Run the experiment."""
    imgs = []
    total_seller_score = 0
    total_seller_loss = 0
    total_buyer_loss = 0
    percent_marker = int(0.5 * cfg.experiment.epochs)

    if cfg.experiment.log:
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        from os import path, mkdir
        log_dir = './data/tensorboard/'
        if not path.exists(log_dir):
            mkdir(log_dir)
        log_dir += f'{datetime.now().strftime("%Y%m%d-%H%M%S")}/'
        writer = SummaryWriter(
            log_dir=log_dir
        )

    for epoch in range(cfg.experiment.epochs + 1):
        print(f"[Epoch {epoch}/{cfg.experiment.epochs}] Running simulation...")
        env.reset() # <- hash tag this if you don't want to use original spawn (I will fix this later) 
        # env.new_place_agents(epoch, cfg.experiment.epochs) # <- toggle this for new agents spawn

        if epoch < percent_marker: 
            for i in range(cfg.agent.seller.num):
                env.woodcutters[i].wood_owned = 5
                env.stonecutters[i].stone_owned = 5

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

        if epoch > 10:
            for i in range(cfg.agent.seller.num):
                total_seller_loss += env.woodcutters[i].model.train_step()
                total_seller_loss += env.stonecutters[i].model.train_step()
            for i in range(cfg.agent.buyer.num):
                total_buyer_loss += env.markets[i].model.train_step()

        total_seller_score += env.seller_score
        current_seller_epsilon = env.woodcutters[0].model.epsilon

        if epoch % cfg.experiment.record_period == 0:
            animate(imgs, f"econ_epoch{epoch}", "./data/animations/")
            # reset the data
            imgs = []
        
        if cfg.experiment.log:
            writer.add_scalar('score', env.seller_score, epoch)
            writer.add_scalar('loss', total_seller_loss, epoch)
            writer.add_scalar('epsilon', current_seller_epsilon, epoch)
        total_seller_score = 0
        total_seller_loss = 0

        for i in range(cfg.agent.seller.num):
            new_epsilon = current_seller_epsilon - cfg.experiment.seller_epsilon_decay
            env.woodcutters[i].model.epsilon = max(new_epsilon, 0.01)
            env.stonecutters[i].model.epsilon = max(new_epsilon, 0.01)


if __name__ == "__main__":
    config = load_config(argparse.Namespace(config="./configs/config.yaml"))
    environment = setup(config)
    run(environment, config)
