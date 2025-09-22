# --------------- #
# region: Imports #
# --------------- #

from typing import Optional

from sorrel.entities import Entity
from sorrel.examples.deprecated.state_punishment.agents import Agent, color_map

# Import gem packages
from sorrel.models.pytorch.iqn import iRainbowModel

# --------------- #
# endregion       #
# --------------- #

# List of objects to generate

MODELS = {"iRainbowModel": iRainbowModel}

AGENTS = {"agent": Agent}

ENTITIES = {"truck": Entity}


# --------------------------- #
# region: object creators     #
# --------------------------- #
def create_models(cfg) -> list[iRainbowModel]:
    """Create a list of models used for the agents.

    Returns:
        A list of models of the specified type
    """
    models = []
    for model_name in vars(cfg.model):
        MODEL_TYPE = MODELS[vars(vars(cfg.model)[model_name])["type"]]
        for _ in range(vars(vars(cfg.model)[model_name])["num"]):
            model = MODEL_TYPE(
                **vars(cfg.model.iqn.parameters), device=cfg.model.iqn.device, seed=1
            )
            model.name = model_name
            models.append(model)

    return models


def create_agents(cfg, models: list) -> list:
    """Create a list of agents used for the task.

    Parameters:
        models: A list of models that govern the agents' behaviour.

    Returns:
        A list of agents of the specified type
    """
    agents = []
    model_num = 0
    colors = color_map(cfg.env.channels)
    agent_model: Optional[iRainbowModel] = None
    for agent_type in vars(cfg.agent):
        has_model = None
        AGENT_TYPE = AGENTS[agent_type]
        for i in range(vars(vars(cfg.agent)[agent_type])["num"]):

            # fetch for model in models
            agent_model_name = vars(vars(cfg.agent)[agent_type])["model"]
            for model in models:
                has_model = False
                if model.name == agent_model_name:
                    agent_model = model
                    has_model = True
                    models.remove(model)

                if has_model:
                    break

            if not has_model:
                raise ValueError(
                    f"Model {agent_model_name} not found, please make sure it is defined in the config file."
                )
            agents.append(
                AGENT_TYPE(model=agent_model, cfg=vars(cfg.agent)[agent_type], ixs=i)
            )

        model_num += 1

    return agents


def create_entities(cfg) -> list[Entity]:
    """Create a list of entities used for the task.

    Returns:
        A list of entities of the specified type
    """
    entities = []
    colors = color_map(cfg.env.channels)
    for entity_type in vars(cfg.entity):
        ENTITY_TYPE = ENTITIES[entity_type]
        for entity in vars(cfg.entity)[entity_type]:
            appearance = colors[vars(entity)["cuisine"]]
            entities.append(ENTITY_TYPE)
    return entities


# --------------------------- #
# endregion: object creators  #
# --------------------------- #
