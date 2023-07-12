from examples.RPG.dualing_cnn_lstm_dqn import Model_CNN_LSTM_DQN
from examples.RPG.agents import Agent
from examples.RPG.entities import Coin, Bone, Food, Gem, Wall
import argparse
import yaml
import os
import pdb
from PIL import Image, ImageDraw, ImageOps
from astropy.visualization import make_lupton_rgb
import numpy as np
import matplotlib.pyplot as plt

# TODO:
# might not be able to initialize entities without populate fn
# see where entities given value and if it makes sense to connect to here

DEVICES = ['cpu', 'cuda']
MODELS = {
    'Model_CNN_LSTM_DQN' : Model_CNN_LSTM_DQN
}
AGENTS = {
    'agent' : Agent,
}
ENTITIES = {
    'Gem' : Gem,
    'Coin': Coin,
    'Bone': Bone,
    'Food': Food
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

    # if config.device not in DEVICES:
    #     raise ValueError(f"Device {config.device} not supported, please use one of {DEVICES}")

    # if config.model.name not in MODELS:
    #     raise ValueError(f"Model {config.model.name} not supported, please use one of {MODELS}")
    
    return config

def create_models(cfg):
    models = []
    for model_name in vars(cfg.model):
        MODEL_TYPE = MODELS[vars(vars(cfg.model)[model_name])['type']]
        for _ in range(vars(vars(cfg.model)[model_name])['num']):
            model = MODEL_TYPE(vars(cfg.model)[model_name])
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
            print(agent_model)
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
        for _ in range(vars(vars(cfg.entity)[entity_type])['num']):
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
        # this moves the specific form of the replay memory into the model class
        # where it can be setup exactly for the model
        agent.model.transfer_memories(agent, extra_reward)

def create_image(env, z = 0):
    """
    Create an RGB image for the given layer
    Each grid has 64 pixels, where each object takes up one pixel
    By default, an empty grid pixel is white
    The color of the object should be defined in the object class
    """
    layer = env.world[:, :, z] # Get the layer
    x, y, _ = layer.shape # Get the width and height of the layer
    print(x, y)

    image_size = (x * 65 - 1, y * 65 - 1) # Size of image, including the gridlines
    image = Image.new('RGB', image_size, 'black') # Create a new image with black color
    draw = ImageDraw.Draw(image)

    for i in range(x):
        for j in range(y):
            for idx, obj in enumerate(layer[i][j]):
                x_pixel = idx % 64 + i * 65 
                y_pixel = idx // 64 + j * 65 

                # Draw the pixel with the appearance attribute of the object
                draw.point((x_pixel, y_pixel), fill=obj.appearance)
    
    for i in range(x):
        for j in range(y):
            for x_pixel in range(i * 65 + 1, (i+1) * 65):
                for y_pixel in range(j * 65 + 1, (j+1) * 65):
                    draw.point((x_pixel, y_pixel), fill=(255, 255, 255)) # Fill it with white

    # Fill remaining pixels in each grid with white color
    for i in range(x):
        for j in range(y):
            for x_pixel in range(i * 65 + 1, (i+1) * 65 - 1): # for x_pixel in range(i * 65 + 1, (i+1) * 65): 
                for y_pixel in range(j * 65 + 1, (j+1) * 65 - 1):
                    if image.getpixel((x_pixel, y_pixel)) == (0, 0, 0): # If pixel is still black
                        draw.point((x_pixel, y_pixel), fill=(255, 255, 255)) # Fill it with white
    return image

def create_pov_image(env, z, agent):
    """
    Create an RGB image for the given layer centered around the agent
    Each grid has 64 pixels, where each object takes up one pixel
    By default, an empty grid pixel is white
    The color of the object should be defined in the object class
    """
    img = create_image(env, z)
    pov = transform_image(env, z, agent, img)
    return pov

def transform_image(env, z, agent, image):
    layer = env.world[:, :, z] # Get the layer
    x, y, _ = agent.location # Get the width and  height of the layer
    width, height, _ = layer.shape
    k = agent.vision # radius of the agent's vision

    left = max(0, (x-k)*65)
    top = max(0, (y-k)*65)
    right = min(width*65-1, (x+k+1)*65-1)
    bottom = min(height*65-1, (y+k+1)*65-1)

    # Crop the original image to the region of interest
    cropped = image.crop((left, top, right, bottom))

    # Calculate the number of pixels to add to each side
    left_pad = max(0, (x-k)*65 - left)
    top_pad = max(0, (y-k)*65 - top)
    right_pad = max(0, ((x+k+1)*65 - 1) - right)
    bottom_pad = max(0, ((y+k+1)*65 - 1) - bottom)

    # Add border of black grids if necessary
    bordered = ImageOps.expand(cropped, (left_pad, top_pad, right_pad, bottom_pad), fill='black')

    return bordered

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



class Cfg:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Cfg(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Cfg(val) if isinstance(val, dict) else val)
