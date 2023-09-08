# --------------- #
# region: Imports #
# --------------- #
import yaml
import os
import argparse

from examples.ft.models.iqn import iRainbowModel
from examples.ft.agents import Agent
from examples.ft.entities import Truck, Object
from examples.ft.utils import color_map
# --------------- #
# endregion       #
# --------------- #

# List of objects to generate

MODELS = {
    'iRainbowModel' : iRainbowModel
}

AGENTS = {
    'agent': Agent
}

ENTITIES = {
    'truck': Truck
}
    
class Cfg:
    '''
    Configuration class for parsing the YAML configuration file.
    '''
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [Cfg(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, Cfg(val) if isinstance(val, dict) else val)

# --------------------------- #
# region: config functions    #
# --------------------------- #
def init_log(cfg: Cfg) -> None:
    print('=' * 40)
    print(f'Starting experiment: {cfg.experiment.name}')
    print(f'Saving to: {cfg.save_dir}')
    print('=' * 40)

def parse_args() -> argparse.Namespace:
    '''
    Helper function for preparsing the arguments.

    Returns:
        The argparse.Namespace object containing the args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file")
    args = parser.parse_args()
    return args

def load_config(args: argparse.Namespace) -> Cfg:
    '''
    Load the parsed arguments into the Cfg class.

    Parameters:
        args: The argparse.Namespace object containing the args

    Returns:
        A Cfg class object with the configurations for the experiment
    '''
    if args.config is None or not os.path.isfile(args.config):
        raise ValueError("Config file not found, please make sure you've included a path to the config file.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = Cfg(config)
    
    return config
# --------------------------- #
# endregion: config functions #
# --------------------------- #

# --------------------------- #
# region: object creators     #
# --------------------------- #
def create_models(cfg: Cfg) -> list:
    '''
    Create a list of models used for the agents.

    Returns:
        A list of models of the specified type 
    '''
    models = []
    for model_name in vars(cfg.model):
        MODEL_TYPE = MODELS[vars(vars(cfg.model)[model_name])['type']]
        for _ in range(vars(vars(cfg.model)[model_name])['num']):
            model = MODEL_TYPE(**vars(cfg.model.iqn.parameters), device=cfg.model.iqn.device, seed = 1)
            model.name = model_name
            models.append(
                model
            )

    return models

def create_agents(
        cfg: Cfg, 
        models: list
    ) -> list[Agent]:
    '''
    Create a list of agents used for the task

    Parameters:
        models: A list of models that govern the agents' behaviour.

    Returns:
        A list of agents of the specified type
    '''
    agents = []
    model_num = 0
    colors = color_map(cfg.env.channels)
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
                color = colors['agent'],
                model = agent_model,
                cfg = vars(cfg.agent)[agent_type]
            ))

        model_num += 1

    return agents

def create_entities(cfg: Cfg) -> list[Object]:
    '''
    Create a list of entities used for the task.

    Returns:
        A list of entities of the specified type
    '''
    entities = []
    colors = color_map(cfg.env.channels)
    for entity_type in vars(cfg.entity):
        ENTITY_TYPE = ENTITIES[entity_type]
        for entity in vars(cfg.entity)[entity_type]:
            color = colors[vars(entity)['cuisine']]
            entities.append(
                ENTITY_TYPE(
                    color=color,
                    cfg = entity
                )
            )
    return entities
# --------------------------- #
# endregion: object creators  #
# --------------------------- #