import random
import numpy as np

from sorrel.worlds import Gridworld
from sorrel.location import Location

from sorrel.examples.ai_econ.agents import Seller, Buyer



class EconWorld(Gridworld):
    """
    AI Economist environment.
    """

    def __init__(self, config, default_entity):
        layers = 4
        super().__init__(config.env.height, config.env.width, layers, default_entity)

        self.config = config
        self.woodcutters: list[Seller] = []
        self.stonecutters: list[Seller] = []
        self.markets: list[Buyer] = []

        # TODO: based on the size of the environment, have a hard limit on the number of agents

        self.max_turns = config.experiment.max_turns
        self.seller_score = 0
        self.buyer_score = 0
        self.total_reward = 0