# --------------- #
# region: Imports #
# --------------- #

# Import base packages
import os
import argparse
import random
import time

from IPython.display import clear_output
from matplotlib import pyplot as plt
from typing import Union
from datetime import datetime

# Import gem-specific packages
from agentarium.utils import ( 
    visual_field_sprite,
    random_seed, 
    set_seed, 
    fig2img, 
    nearest_2_power, 
    minmax,
)
from agentarium.buffers import GameReplayBuffer as Buffer
from agentarium.models.transformer import VisionTransformer as ViT

# Import RPG-specific packages
from examples.RPG.env import RPG
from examples.RPG.utils import (
    load_config,
    create_agents,
    create_entities,
    create_models,
    Cfg
)
from agentarium.logging_utils import GameLogger

# --------------- #
# endregion       #
# --------------- #

def eval_model(*flags,
               model_path: Union[str, os.PathLike],
               config_path: Union[str, os.PathLike],
               n_games: int = 1,
               ) -> list:

    '''
    Load model parameters from disk and evaluate N games with it.

    Parameters:
        *flags: An unpacked list of game objects to return. 
        Currently implemented: `memories`, `frames`, `scores`. 
        `jupyter-mode` modifies the pretty print function for ipynb output.
        model_path: The path to the model weights. \n
        config_path: The path to the configuration file. \n
        n_games: The number of games to play. \n

    Returns:
        A dictionary of game objects based on the flags passed in. 
    '''

    # Initialize the dictionary of game objects based on the flags passed in.
    game_objects = dict.fromkeys(flags)

    # Randomize seed
    seed = random_seed()
    
    # Read in configuration
    args = argparse.Namespace(config=config_path)
    cfg = load_config(args)

    # Create game objects
    models = create_models(cfg)
    agents = create_agents(cfg, models)
    entities = create_entities(cfg)
    env = RPG(cfg, agents, entities)

    # Load model checkpoint
    for agent in agents:
        agent.model.load(model_path)
        # Zero out model epsilon
        agent.model.set_epsilon(0)
        # Evaluation mode
        agent.model.eval()

    # Create experience buffer for the inverse model
    if 'memories' in flags:
        # Use the nearest power of 2 for the buffer size and batch size
        buffer_size = nearest_2_power(cfg.experiment.max_turns * n_games)
        # Batch size cannot be smaller than 64 or greater than 1024
        batch_size = minmax(buffer_size // 16, 64, 1024)
        buffer = Buffer(
            buffer_size=buffer_size,
            batch_size=batch_size, 
            device=cfg.model.iqn.device,
            gamma=cfg.model.iqn.parameters.GAMMA,
            timesteps=100,
        )
    # Create a list to add frames to
    if 'frames' in flags:
        game_objects['frames'] = []
    scores = GameLogger(max_epochs=n_games)

    # ----------------- #
    # region: Game loop #
    # ----------------- #
    for game in range(n_games):

        # Get the next seed
        set_seed(seed + game)

        # Game variables
        done, turn, points = 0, 0, 0
        frames = []
        ACTIONS = [
            'up', 'down', 'left', 'right'
        ]

        # Reset game objects
        env.reset()
        for agent in agents:
            agent.reset()

        if 'frames' in flags:
            img = visual_field_sprite(env.world, tile_size = env.tile_size)
            clear_output(wait = True)
            print(f'╔═════════════════════════════════════════════════════╗')
            print(f'║                 Initial game state...               ║')
            print(f'╚═════════════════════════════════════════════════════╝')
            plt.imshow(img.astype(int))
            plt.title(f'Initial state.')
            frame = fig2img(plt)
            frames.append(frame)
            plt.show()
        
        # ----------------- #
        # region: Turn loop #
        # ----------------- #
        while done == 0:
            
            # Increment turn counter
            turn += 1

            entities = env.get_entities_for_transition()
            # Entity transition
            for entity in entities:
                entity.transition(env)

            for agent in agents:

                # Agent transition
                state, action, reward, next_state, done_ = agent.transition(env)
                points += reward

                # If agent finished or max turns is reached, end the turn
                if turn >= cfg.experiment.max_turns or done_: 
                    done = 1 # End loop if maximum turns have been reached
                
                # Append the experience to the replay
                exp = (0, (state, action, reward, next_state, done))
                agent.episode_memory.append(exp)

                if 'memories' in flags:
                    buffer.add(state, action, reward, next_state, done)

                # Show the outcome of the turn
                if 'frames' in flags:
                    img = visual_field_sprite(env.world, tile_size=env.tile_size)
                    clear_output(wait = True)
                    scores.pretty_print(*flags, epoch=game, turn=turn, reward=points)
                    plt.imshow(img.astype(int))
                    plt.title(f'Game {seed}. Turn {turn}. Action: {ACTIONS[action]}. Reward: {reward}')
                    frame = fig2img(plt)
                    frames.append(frame)
                    plt.show()
                    
        
        # ----------------- #
        # endregion         #
        # ----------------- #

        # Collect information for the scores and the frames, and reset the batched action deque.
        if 'memories' in flags:
            buffer.game.clear()
        if 'scores' in flags:
            scores.record_turn(game, turn, 0., points)
        if 'frames' in flags:
            game_objects['frames'].append(frames)    

    # ----------------- #
    # endregion         #
    # ----------------- #

    # After the game is over, add the scores and the buffer to the game objects
    if 'memories' in flags:
        game_objects['memories'] = buffer
    if 'scores' in flags:
        game_objects['scores'] = scores

    # Remove `jupyter-mode` from the dictionary, as we don't need to return it.
    game_objects.pop('jupyter-mode', None)

    return game_objects

def train_transformer_model(cfg: Cfg, memories_path):

    # Load the stored memories
    memories = Buffer.load(memories_path)
    memories.batch_size = cfg.model.batch_size # Reduce batch size since the # of frames is quite long

    inverse_model = ViT(
        state_size=cfg.model.state_size,
        action_space=cfg.model.action_space,
        layer_size=cfg.model.layer_size,
        patch_size=cfg.model.patch_size,
        num_frames=cfg.model.num_frames,
        num_heads=cfg.model.num_heads,
        batch_size=cfg.model.batch_size,
        num_layers=cfg.model.num_layers,
        memory=memories,
        LR=cfg.model.LR,
        device=cfg.model.device,
        seed=random.randint(0,1000)
    )

    if cfg.log:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(
            log_dir=f'{cfg.root}/examples/RPG/runs/inverse/{datetime.now().strftime("%Y%m%d-%H%m%s")}/'
        )
    
    for epoch in range(cfg.experiment.epochs): # NOTE: with mps, 10000 epochs takes about 24 hours on the mac mini
        start = time.time()
        state_loss, action_loss = inverse_model.train_model()
        state_predictions, state_targets = inverse_model.plot_trajectory()

        if cfg.log:
            writer.add_scalar('Action loss', action_loss, epoch)
            writer.add_scalar('State loss', state_loss, epoch)
            writer.add_images('State targets', state_targets[:5], epoch) # First 5 images
            writer.add_images('State preds', state_predictions[:5], epoch)

        end = time.time()
        duration = end - start
        clear_output(wait = True)
        print(f'Epoch {epoch}. Total training duration: {duration}.')

        if epoch > 0 and epoch % 1000 == 0:
            # Save the model every 1000 epochs
            inverse_model.save(f'{cfg.root}/examples/RPG/models/checkpoints/transformer_{epoch}.pkl')       


        