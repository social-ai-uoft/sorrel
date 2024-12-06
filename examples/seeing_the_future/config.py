# --------------- #
# region: Imports #
# --------------- #

# Import gem packages
from agentarium.config import (
    Cfg
)
from agentarium.models.iqn import iRainbowModel
from examples.trucks.agents import Agent
from examples.trucks.entities import Truck, Entity
from examples.trucks.utils import color_map

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

# --------------------------- #
# region: object creators     #
# --------------------------- #
def create_models(cfg: Cfg) -> list[iRainbowModel]:
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
                appearance = colors['Agent'],
                model = agent_model,
                cfg = vars(cfg.agent)[agent_type]
            ))

        model_num += 1

    return agents

def create_entities(cfg: Cfg) -> list[Entity]:
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
            appearance = colors[vars(entity)['cuisine']]
            entities.append(
                ENTITY_TYPE(
                    appearance=appearance,
                    cfg = entity
                )
            )
    return entities
# --------------------------- #
# endregion: object creators  #
# --------------------------- #