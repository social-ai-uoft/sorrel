import torch

from sorrel.action.action_spec import ActionSpec
from sorrel.models.base_model import SorrelModel
from sorrel.models.pytorch.iqn import iRainbowModel
from sorrel.observation.observation_spec import ObservationSpec
from examples.ai_econ.agents import Buyer, EconEnvObsSpec, Seller


def create_models(cfg):
    """
    Create models given a configuration.

    Returns a tuple of 3 lists of models, for woodcutters/stonecutters/markets respectively.
    """
    woodcutter_models, stonecutter_models = [], []
    market_models = []
    for _ in range(cfg.agent.seller.num):
        model_name = cfg.agent.seller.model
        model_type = vars(cfg.model.seller)[model_name].type
        if model_type == "iRainbowModel":
            woodcutter_models.append(
                iRainbowModel(
                    **vars(cfg.model.seller)[model_name].parameters.to_dict(),
                    seed=torch.random.seed(),
                    input_size=(
                        1,
                        (
                            len(cfg.agent.seller.obs.entity_list)
                            * (2 * cfg.agent.seller.obs.vision_radius + 1)
                            * (2 * cfg.agent.seller.obs.vision_radius + 1)
                            + 2
                        ),
                    )
                )
            )
            stonecutter_models.append(
                iRainbowModel(
                    **vars(cfg.model.seller)[model_name].parameters.to_dict(),
                    seed=torch.random.seed(),
                    input_size=(
                        1,
                        (
                            len(cfg.agent.seller.obs.entity_list)
                            * (2 * cfg.agent.seller.obs.vision_radius + 1)
                            * (2 * cfg.agent.seller.obs.vision_radius + 1)
                            + 2
                        ),
                    )
                )
            )

    for _ in range(cfg.agent.buyer.num):
        model_name = cfg.agent.buyer.model
        model_type = vars(cfg.model.buyer)[model_name]
        if model_type == "iRainbowModel":
            market_models.append(
                iRainbowModel(
                    **vars(cfg.model.seller)[model_name].parameters.to_dict()
                )
            )

    return woodcutter_models, stonecutter_models, market_models


def create_agents(cfg, woodcutter_models, stonecutter_models, market_models):
    """
    Creates the agents for this environment.
    Appearances are placeholders for now; 0 for woodcutters, 1 for stonecutters, and 2 for markets.

    Returns a tuple of 3 lists of agents, woodcutters/stonecutters/markets respectively.
    Minority agents are positioned at the start of the woodcutters or stonecutters lists.
    """
    woodcutters, stonecutters = [], []
    markets = []
    for i in range(0, cfg.agent.seller.num):
        woodcutters.append(
            Seller(
                cfg,
                appearance=0,
                is_majority=(
                    i >= cfg.agent.seller.num * cfg.agent.seller.majority_percentage
                ),
                is_woodcutter=True,
                observation_spec=EconEnvObsSpec(
                    cfg.agent.seller.obs.entity_list,
                    vision_radius=cfg.agent.seller.obs.vision_radius,
                ),
                action_spec=ActionSpec([0, 1, 2, 3, 4, 5, 6]),
                model=woodcutter_models[i],
            )
        )
        stonecutters.append(
            Seller(
                cfg,
                appearance=1,
                is_majority=(
                    i >= cfg.agent.seller.num * cfg.agent.seller.majority_percentage
                ),
                is_woodcutter=False,
                observation_spec=EconEnvObsSpec(
                    cfg.agent.seller.obs.entity_list,
                    vision_radius=cfg.agent.seller.obs.vision_radius,
                ),
                action_spec=ActionSpec([0, 1, 2, 3, 4, 5, 6]),
                model=stonecutter_models[i],
            )
        )

    for i in range(0, cfg.agent.buyer.num):
        markets.append(
            Buyer(
                cfg,
                appearance=2,
                observation_spec=ObservationSpec(cfg.agent.buyer.obs.entity_list, vision_radius=1),
                model=SorrelModel(0, 0, 0, 0),
            )
        )

    return woodcutters, stonecutters, markets
