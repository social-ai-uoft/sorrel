from examples.RPG.agents import Agent, color_map
from examples.RPG.entities import Coin, Bone, Food, Gem, Wall

from agentarium.models.iqn import iRainbowModel
import argparse
import yaml
import os

DEVICES = ['cpu', 'cuda']
MODELS = {
    'iRainbowModel' : iRainbowModel
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
                    color_map(cfg.env.channels)[entity_type], cfg
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