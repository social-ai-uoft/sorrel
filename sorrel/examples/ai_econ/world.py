# """The environment for treasurehunt, a simple example for the purpose of a tutorial."""

# # begin imports

# from omegaconf import DictConfig, OmegaConf

# from sorrel.worlds import Gridworld

# # end imports


# # begin econworld
# class EconWorld(Gridworld):
#     """AI Economist world."""

#     def __init__(self, config: dict | DictConfig, default_entity):
#         layers = 2
#         if type(config) != DictConfig:
#             config = OmegaConf.create(config)
#         super().__init__(
#             config.world.height, config.world.width, layers, default_entity
#         )

#         self.config = config


# # end econworld
