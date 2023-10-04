from examples.RPG3.iRainbow_clean import iRainbowModel
from examples.RPG3.agents import Agent
from examples.RPG3.entities import Gem, EmptyObject, Wall
import argparse
import yaml
import os
import pdb
from PIL import Image, ImageDraw, ImageOps
from astropy.visualization import make_lupton_rgb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

DEVICES = ['cpu', 'cuda']
MODELS = {
    'iRainbowModel' : iRainbowModel
}
AGENTS = {
    'agent' : Agent
}
ENTITIES = {
    'Gem1' : Gem,
    'Gem2' : Gem,
    'Gem3' : Gem
}

def init_log(cfg):
    print('-' * 60)
    print(f'Starting experiment: {cfg.experiment.name}')
    print(f'Saving to: {cfg.save_dir}')
    print()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file")
    args = parser.parse_args()
    return args

def load_config(args):
    if args.config is None or not os.path.isfile(args.config):
        raise ValueError("Config file not found, please make sure you've included a path to the config file.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = Cfg(config)
    
    return config

def create_models(cfg, seed, device):
    models = []
    for model_name in vars(cfg.model):
        MODEL_TYPE = MODELS[vars(vars(cfg.model)[model_name])['type']]
        for _ in range(vars(vars(cfg.model)[model_name])['num']):
            model = MODEL_TYPE(cfg, seed, device) #vars(cfg.model)[model_name], seed, device)
            model.name = model_name
            models.append(
                model
            )

    return models

def create_agents(cfg, models):
    agents = []
    model_num = 0
    for agent_type in vars(cfg.agent):
        AGENT_TYPE = AGENTS[agent_type]
        for _ in range(vars(vars(cfg.agent)[agent_type])['num']):

            # fetch for model in models
            agent_model_name = vars(vars(cfg.agent)[agent_type])['model']
            for model in models:
                has_model = False
                if model.name == agent_model_name:
                    agent_model = model
                    has_model = True
                    models.remove(model)
                
                if has_model:
                    break

            if not has_model:
                raise ValueError(f"Model {agent_model_name} not found, please make sure it is defined in the config file.")
            agents.append(AGENT_TYPE(
                agent_model,
                cfg
            ))

        model_num += 1

    return agents

def create_entities(cfg):
    entities = []
    for entity_type in vars(cfg.entity):
        ENTITY_TYPE = ENTITIES[entity_type]

        # NOTE: Assumes only entities with num and num > 1 need to be initialized at the start
        if 'start_num' in vars(vars(cfg.entity)[entity_type]):
            for _ in range(vars(vars(cfg.entity)[entity_type])['start_num']):
                entities.append(ENTITY_TYPE(
                    cfg
                ))

    return entities


def update_memories(env, agent, done, end_update=True):
    exp = agent.episode_memory[-1]
    lastdone = exp[1][4]
    if done == 1:
        lastdone = 1
    if end_update == False:
        exp = exp[0], (exp[1][0], exp[1][1], agent.reward, exp[1][3], lastdone)
    if end_update == True:
        input2 = agent.pov(env)
        exp = exp[0], (exp[1][0], exp[1][1], agent.reward, input2, lastdone)
    agent.episode_memory[-1] = exp

def transfer_world_memories(agents, extra_reward = True):
    # transfer the events from agent memory to model replay
    for agent in agents:
        # this moves the specific form of the replay memory into the model class where it can be setup exactly for the model
        agent.model.transfer_memories(agent, extra_reward)

def make_pov_image(env, agent, wall_app=[0.0, 0.0, 0.0]):
    """
    Create an agent visual field of size (2k + 1, 2k + 1) pixels
    Layer = location[2] and layer in the else are added to this function
    """
    k = agent.visual_depth
    world = env.world
    location = agent.location

    if len(location) > 2:
        layer = location[2]
    else:
        layer = 0

    bounds = (location[0] - k, location[0] + k, location[1] - k, location[1] + k)
    # instantiate image
    image_r = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))
    image_g = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))
    image_b = np.random.random((bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1))

    for i in range(bounds[0], bounds[1] + 1):
        for j in range(bounds[2], bounds[3] + 1):
            # while outside the world array index...
            if i < 0 or j < 0 or i >= env.x - 1 or j >= env.y:
                # image has shape bounds[1] - bounds[0], bounds[3] - bounds[2]
                # visual appearance = wall
                image_r[i - bounds[0], j - bounds[2]] = wall_app[0]
                image_g[i - bounds[0], j - bounds[2]] = wall_app[1]
                image_b[i - bounds[0], j - bounds[2]] = wall_app[2]
            else:
                image_r[i - bounds[0], j - bounds[2]] = world[i][j][layer].appearance[0]
                image_g[i - bounds[0], j - bounds[2]] = world[i][j][layer].appearance[1]
                image_b[i - bounds[0], j - bounds[2]] = world[i][j][layer].appearance[2]

    #image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
    image = np.zeros((image_r.shape[0], image_r.shape[1], 3))
    image[:, :, 0] = image_r
    image[:, :, 1] = image_g
    image[:, :, 2] = image_b
    # display image
    # plt.imshow(image)
    # plt.show()
    return image

def create_world_image(env, layer=0, use_sprites=False):
        """
        Creates an RGB image of the whole world
        """
        world_shape = env.world.shape
        image_r = np.zeros((world_shape[0] * env.tile_size[0], world_shape[1] * env.tile_size[1]))
        image_g = np.zeros((world_shape[0] * env.tile_size[0], world_shape[1] * env.tile_size[1]))
        image_b = np.zeros((world_shape[0] * env.tile_size[0], world_shape[1] * env.tile_size[1]))

        for i in range(world_shape[0]):
            for j in range(world_shape[1]):

                if use_sprites:
                    tile_appearance = env.world[i, j, layer].sprite
                    tile_image = Image.open(tile_appearance).resize(env.tile_size).convert('RGBA')
                    tile_image_array = np.array(tile_image)

                    # Set transparent pixels to white
                    alpha = tile_image_array[:, :, 3]
                    tile_image_array[alpha == 0, :3] = 255

                    image_r[i * env.tile_size[0]: (i + 1) * env.tile_size[0], j * env.tile_size[1]: (j + 1) * env.tile_size[1]] = tile_image_array[:, :, 0]
                    image_g[i * env.tile_size[0]: (i + 1) * env.tile_size[0], j * env.tile_size[1]: (j + 1) * env.tile_size[1]] = tile_image_array[:, :, 1]
                    image_b[i * env.tile_size[0]: (i + 1) * env.tile_size[0], j * env.tile_size[1]: (j + 1) * env.tile_size[1]] = tile_image_array[:, :, 2]
                
                else:
                    tile_appearance = env.world[i, j, layer].appearance
                    image_r[i * env.tile_size[0]: (i + 1) * env.tile_size[0], j * env.tile_size[1]: (j + 1) * env.tile_size[1]] = tile_appearance[0]
                    image_g[i * env.tile_size[0]: (i + 1) * env.tile_size[0], j * env.tile_size[1]: (j + 1) * env.tile_size[1]] = tile_appearance[1]
                    image_b[i * env.tile_size[0]: (i + 1) * env.tile_size[0], j * env.tile_size[1]: (j + 1) * env.tile_size[1]] = tile_appearance[2]

        image = make_lupton_rgb(image_r, image_g, image_b, stretch=0.5)
        return image

def create_replays(**kwargs):
        cfg = kwargs['cfg']
        cur_epoch = kwargs['epoch']

        if cfg.replay.save == True and cur_epoch % cfg.replay.replay_frequency == 0:
            for i in range(cfg.replay.num_replays):
                # Create save directory
                save_dir = cfg.save_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)   
                
                replay_dir = save_dir + 'replay/'
                if not os.path.exists(replay_dir):
                    os.makedirs(replay_dir)   

                # Format filename
                filename = (
                    replay_dir
                    + cfg.experiment.name
                    + "_epoch"
                    + str(cur_epoch)
                    + "_num"
                    + str(i)
                    + ".gif"
                )

                # Create inputs for create_replay
                agents = kwargs['agents']
                entities = kwargs['entities']
                env = kwargs['env']

                create_replay(cfg, agents, entities, env, filename)

def create_replay(cfg, agents, entities, env, filename):

    fig = plt.figure()
    ims = []
    env.reset_world(agents, entities)
    turns = cfg.replay.turns
    done = 0
    turn = 0

    for agent in agents:
        agent.reset()

        # Save agents' epsilon to recover after the replay
        agent.temp_eps = agent.model.epsilon
        agent.model.epsilon = cfg.replay.epsilon

    while not done:
        if turn > turns:
            done = 1

        image = create_world_image(env, use_sprites=cfg.replay.use_sprites)
        # left_label = "Turn: " + str(turn)
        # cv2.putText(image, left_label, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        im = plt.imshow(image, animated=True)
        ims.append([im])

        for entity in entities:
            entity.transition(env)

        random.shuffle(agents)

        for agent in agents:
            agent.transition(env)

        turn += 1

    ani = animation.ArtistAnimation(fig, ims)
    ani.save(filename, writer="PillowWriter", fps=cfg.replay.fps)

    # Recover the agents' epsilon
    for agent in agents:
        agent.model.epsilon = agent.temp_eps




class Cfg:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Cfg(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Cfg(val) if isinstance(val, dict) else val)
