from examples.partner_selection.agents import Agent, color_map
from examples.partner_selection.entities import Coin, Gem, Wall, Bone

from agentarium.models.iqn import iRainbowModel
import argparse
import yaml
import os
from PIL import Image
import shutil
from collections import deque
from agentarium.models.PPO import PPO
from examples.partner_selection.task_models import Classification
import random 
import itertools
import numpy as np
from scipy.special import softmax

DEVICES = ['cpu', 'cuda']
MODELS = {
    'iRainbowModel' : iRainbowModel,
    'Classification' : Classification
}
AGENTS = {
    'agent' : Agent,
}
ENTITIES = {
    'Gem' : Gem,
    'Coin': Coin,
    # 'Wall' : Wall,
    # 'EmptyObject': EmptyObject, 
    'Bone': Bone,
    # 'Food': Food
}

def init_log(cfg):
    print('-' * 60)
    print(f'Starting experiment: {cfg.experiment.name}')
    print()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file", default='./config.yaml')
    args = parser.parse_args()
    return args

def load_config(args):
    if args.config is None or not os.path.isfile(args.config):
        raise ValueError("Config file not found, please make sure you've included a path to the config file.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = Cfg(config)
    
    return config

def create_models(cfg):
    models = []
    for model_name in vars(cfg.model):
        MODEL_TYPE = MODELS[vars(vars(cfg.model)[model_name])['type']]
        for _ in range(vars(vars(cfg.model)[model_name])['num']):
            model = MODEL_TYPE(**vars(vars(vars(cfg.model)[model_name])['parameters']), device = 'cpu', seed = 1)
            model.name = model_name
            models.append(
                model
            )

    return models

def create_interaction_task_models(cfg):

    models = []
        
    MODEL_TYPE = MODELS[cfg.interaction_task.model.type]
    
    for _ in range(cfg.interaction_task.model.num):
        model = MODEL_TYPE(**vars(vars(cfg.interaction_task.model)['parameters']))
        model.name = cfg.interaction_task.model.type
        models.append(
            model
        )

    if len(models) != cfg.agent.agent.num:
        raise ValueError('Please make sure the number of models match the number of agents.')

    return models


def create_partner_selection_models_PPO(
        num_model, 
        cfg,
        device='cpu'
        ):
    """Create models for agents."""
    models = deque(maxlen=num_model)
    for _ in range(num_model):
        model = PPO(
            device=device, 
            state_dim=32,
            action_dim=6,
            # lr_actor=0.0001,
            # lr_critic=0.00005,
            lr_actor=0.00002,
            lr_critic=0.00001,
            gamma=0.99,
            K_epochs=10,
            eps_clip=0.2,
            # entropy_coefficient=0.01  
            entropy_coefficient=0.005
        )
        model.name = 'PPO'
        models.append(model)
    # convert to device
    for model in range(len(models)):
        models[model].model_main.to(device)
    return models


def create_agents(cfg, models):
    for ixs_m, model in enumerate(models):
        model.ixs = ixs_m
    agents = []
    model_num = 0
    if len(models) != cfg.agent.agent.num:
        raise ValueError('Please make sure the number of models match the number of agents.')
    for agent_type in vars(cfg.agent):
        AGENT_TYPE = AGENTS[agent_type]
        for ixs in range(vars(vars(cfg.agent)[agent_type])['num']):
            # fetch for model in models
            agent_model_name = vars(vars(cfg.agent)[agent_type])['model']
            for ixs_m, model in enumerate(models):
                has_model = False
                if model.name == agent_model_name:
                    if model.ixs == ixs:
                        agent_model = model
                        has_model = True
                        models.remove(model)
                        
                
                if has_model:
                    break

            if not has_model:
                raise ValueError(f"Model {agent_model_name} not found, please make sure it is defined in the config file.")
            agents.append(AGENT_TYPE(
                agent_model,
                cfg,
                ixs
            ))
        
        

        model_num += 1

    return agents

def create_entities(cfg):
    entities = []
    for entity_type in vars(cfg.entity):
        ENTITY_TYPE = ENTITIES[entity_type]
        # if entity_type not in ['Wall', 'EmptyObject']:
        #     ENTITY_TYPE = ENTITIES[entity_type]

        # NOTE: Assumes only entities with num and num > 1 need to be initialized at the start
        if 'start_num' in vars(vars(cfg.entity)[entity_type]):
            for _ in range(vars(vars(cfg.entity)[entity_type])['start_num']):
                entities.append(ENTITY_TYPE(
                    color_map(cfg.env.channels)[entity_type], cfg
                    # cfg.entity_type.apperance, cfg
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

class Cfg:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Cfg(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Cfg(val) if isinstance(val, dict) else val)





def create_gif_from_arrays(image_arrays, output_path, duration=100, loop=0):
    """
    Create a GIF from a sequence of images in NumPy array format.

    Args:
        image_arrays (list of np.ndarray): Sequence of images as NumPy arrays.
        output_path (str): Path to save the output GIF.
        duration (int): Duration of each frame in milliseconds (controls speed).
        loop (int): Number of times the GIF should loop (0 for infinite).
    """
    # Convert NumPy arrays to PIL Images
    if type(image_arrays[0]) != Image.Image:
        pil_images = [Image.fromarray(img) for img in image_arrays]
    else:
        pil_images = image_arrays
    
    # Save as GIF
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF saved at {output_path}")


def save_config_backup(config_file_path, backup_dir):
    """
    Saves a backup of the YAML configuration file with a timestamp and 'exp_name' as part of the name.

    Args:
        config_file_path (str): Path to the configuration YAML file.
        backup_dir (str): Directory where backup files will be saved.
    """
    # Ensure the backup directory exists
    os.makedirs(backup_dir, exist_ok=True)

    # Load the YAML configuration file
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract the 'exp_name' parameter from the config (defaults to 'experiment' if not found)
    exp_name = config.get('exp_name', 'experiment')

    # Construct the backup file name using 'exp_name' and the timestamp
    backup_file_name = f"{exp_name}_config_backup.yaml"
    backup_file_path = os.path.join(backup_dir, backup_file_name)

    # Create the destination directory if it doesn't exist
    destination_dir = os.path.dirname(backup_file_path)  # Extracts the directory part of the destination path
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        

    # Copy the config file to the backup location
    try:
        shutil.copy(config_file_path, backup_file_path)
        print(f"Backup created successfully at {backup_file_path}")
    except Exception as e:
        print(f"Error during backup: {e}")

    print(f"Config file backed up to: {backup_file_path}")


def find_agent_with_ixs(agents, ixs):
    for agent in agents:
        if agent.ixs == ixs:
            return agent

def agents_sampling(agents):
    """
    Sample two agents as potential partner choices and one agent as the focal agent.
    """
    # sample all needed agents
    selected_agents = random.sample(agents, 3)
    selected_agents_indices = [agent.ixs for agent in selected_agents]
    # pick the focal agent
    focal_agent_ixs = random.sample([i for i in range(len(selected_agents_indices))])[0]
    focal_agent = selected_agents[focal_agent_ixs]
    partner_choices = [selected_agents[i] for i in range(len(selected_agents)) if i != focal_agent_ixs]
    return focal_agent, partner_choices


def create_agent_appearances(num_agents):
    """
    Create the appearances for num_agents agents, used in partner selection process.

    Args:
    num_agents - number of agents
    """
    appearances = 255. * np.eye(num_agents)
    return appearances


def generate_preferences(X):
    """Generate a list of length X where the sum of elements is 1."""
    values = np.random.rand(X)  # Generate X random numbers
    values = np.array([0., 1.0]) #TODO: temporary
    return values  # Normalize so sum is 1


def generate_variability(X, max=0.5):
    """Generate a list of nums representing the variability of preferences."""
    values = np.random.rand(X) * max
    values = np.array([1. for _ in range(X)]) * max # 1, 0, 1 # working: 1, 0.1, 1
    return values.tolist()


def get_agents_by_ixs(agents, target_ixs):
    """
    Filters agents whose 'ixs' value is in the target_ixs list.

    :param agents: List of agent dictionaries (or objects with an 'ixs' attribute).
    :param target_ixs: List or set of desired ixs values.
    :return: List of matching agents.
    """
    target_ixs = set(target_ixs)  # Convert to set for faster lookup
    return [agent for agent in agents if agent.ixs in target_ixs]