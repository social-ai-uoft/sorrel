import torch
from omegaconf import DictConfig, OmegaConf

from sorrel.action.action_spec import ActionSpec
from sorrel.models.base_model import BaseModel
from sorrel.models.pytorch.ppo import PyTorchPPO, RolloutBuffer
from sorrel.observation.observation_spec import ObservationSpec
from sorrel.examples.ai_econ.agents import Buyer, EconEnvObsSpec, Seller


def create_models(config: DictConfig) -> tuple[list[PyTorchPPO], list[PyTorchPPO], list[PyTorchPPO]]:
    """
    Create models given a configuration.

    Returns a tuple of 3 lists of models, for woodcutters/stonecutters/markets respectively.
    """
    woodcutter_models, stonecutter_models = [], []
    market_models = []
    for _ in range(config.agent.seller.num):
        model_name = config.agent.seller.model
        model_type = config.model.seller[model_name].type
        if model_type == "PyTorchPPO":  
            
            woodcutter_models.append(
                PyTorchPPO(  
                    input_size=(
                        len(config.agent.seller.obs.entity_list)
                        * (2 * config.agent.seller.obs.vision_radius + 1)
                        * (2 * config.agent.seller.obs.vision_radius + 1)
                        + 2,
                    ),
                    action_space=8,  
                    max_turns=config.experiment.max_turns, 
                    seed=torch.random.seed(),
                    **config.model.seller[model_name].parameters  
                )
            )

            stonecutter_models.append(
                PyTorchPPO(
                    input_size=(
                        len(config.agent.seller.obs.entity_list)
                        * (2 * config.agent.seller.obs.vision_radius + 1)
                        * (2 * config.agent.seller.obs.vision_radius + 1)
                        + 2,
                    ),
                    action_space=8,
                    max_turns=config.experiment.max_turns,
                    seed=torch.random.seed(),
                    **config.model.seller[model_name].parameters
                )
            )

    for _ in range(config.agent.buyer.num):
        model_name = config.agent.buyer.model
        model_type = config.model.buyer[model_name].type 
        if model_type == "PyTorchPPO": 
            
            
            market_models.append(
                PyTorchPPO(
                    input_size=(len(config.agent.buyer.obs.entity_list) * 3 * 3,),
                    action_space=7,  
                    max_turns=config.experiment.max_turns,  
                    seed=torch.random.seed(),
                    **config.model.buyer[model_name].parameters
                )
            )

    return woodcutter_models, stonecutter_models, market_models


def create_agents(config: DictConfig, woodcutter_models, stonecutter_models, market_models) -> tuple[list[Seller], list[Seller], list[Buyer]]:
    """
    Creates the agents for this environment.
    Appearances are placeholders for now; 0 for woodcutters, 1 for stonecutters, and 2 for markets.

    Returns a tuple of 3 lists of agents, woodcutters/stonecutters/markets respectively.
    Minority agents are positioned at the start of the woodcutters or stonecutters lists.
    """
    woodcutters, stonecutters = [], []
    markets = []
    for i in range(0, config.agent.seller.num):
        woodcutters.append(
            Seller(
                config,
                appearance=0,
                is_majority=(
                    i >= config.agent.seller.num * config.agent.seller.majority_percentage
                ),
                is_woodcutter=True,
                observation_spec=EconEnvObsSpec(
                    config.agent.seller.obs.entity_list,
                    vision_radius=config.agent.seller.obs.vision_radius,
                    full_view=False, 
                ),
                action_spec=ActionSpec([0, 1, 2, 3, 4, 5, 6]),
                model=woodcutter_models[i],
            )
        )
        stonecutters.append(
            Seller(
                config,
                appearance=1,
                is_majority=(
                    i >= config.agent.seller.num * config.agent.seller.majority_percentage
                ),
                is_woodcutter=False,
                observation_spec=EconEnvObsSpec(
                    config.agent.seller.obs.entity_list,
                    vision_radius=config.agent.seller.obs.vision_radius,
                    full_view=False,
                ),
                action_spec=ActionSpec([0, 1, 2, 3, 4, 5, 6, 7]),
                model=stonecutter_models[i],
            )
        )

    for i in range(0, config.agent.buyer.num):
        markets.append(
            Buyer(
                config,
                appearance=2,
                observation_spec=ObservationSpec(config.agent.buyer.obs.entity_list, vision_radius=1, full_view=False),
                model=market_models[i], #Before: model=BaseModel(),
            )
        )

    return woodcutters, stonecutters, markets
