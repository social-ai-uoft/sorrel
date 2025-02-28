"""Agent classes for the AI Economist task. Includes the resource collectors and the market deciders."""

import numpy as np

from agentarium.agents import Agent
from agentarium.environments import GridworldEnv
from agentarium.observation.observation import ObservationSpec


class EconEnvObsSpec(ObservationSpec):

    def observe(
        self,
        env: GridworldEnv,
        location: tuple | None = None,
        resources: list[int] | None = None,
    ):
        visual_field = super().observe(env, location).flatten()
        if resources is not None:
            resources = np.array(resources)
        else:
            resources = np.array([0, 0])
        return np.concatenate((visual_field, resources))


class Seller(Agent):
    """A resource gatherer in the AI Economist environment."""

    def __init__(
        self,
        cfg,
        appearance,
        is_woodcutter,
        is_majority,
        observation_spec: ObservationSpec,
        model,
    ):
        # the actions are: move north, move south, move west, move east, extract resource, sell wood, sell stone
        action_space = [0, 1, 2, 3, 4, 5, 6]
        super().__init__(observation_spec, model, action_space)

        self.appearance = appearance  # the "id" of the agent
        self.is_woodcutter = (
            is_woodcutter  # is the agent part of the wood cutter group?
        )
        self.is_majority = (
            is_majority  # is the agent part of the marjority of its group?
        )

        if (self.is_woodcutter and self.is_majority) or (
            not self.is_woodcutter and not self.is_majority
        ):
            self.wood_success_rate = cfg.agent.seller.skilled_success_rate
            self.stone_success_rate = cfg.agent.seller.unskilled_success_rate
        else:
            self.wood_success_rate = cfg.agent.seller.unskilled_success_rate
            self.stone_success_rate = cfg.agent.seller.skilled_success_rate

        self.sell_reward = cfg.agent.seller.sell_reward

        self.wood_owned = 0
        self.stone_owned = 0

        if self.is_woodcutter:
            self.sprite = "./assets/hero.png"
        else:
            self.sprite = "./assets/hero-g.png"

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        state = np.zeros_like(np.prod(self.model.input_size))
        action = 0
        reward = 0.0
        done = False
        for _ in range(self.model.num_frames):
            self.add_memory(state, action, reward, done)

    def pov(self, env: GridworldEnv) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(
            env, self.location, [self.wood_owned, self.stone_owned]
        )

        # flatten the image to get the state
        return image.reshape(1, -1)

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        prev_states = self.model.memory.current_state(
            stacked_frames=self.model.num_frames - 1
        )
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action

    def act(self, env: GridworldEnv, action: int) -> float:
        """Act on the environment, returning the reward."""

        # if the agent chooses to move
        if 0 <= action <= 3:
            if action == 0:  # move north
                if self.is_woodcutter:
                    self.sprite = "./assets/hero-back.png"
                else:
                    self.sprite = "./assets/hero-back-g.png"
                new_location = (
                    self.location[0] - 1,
                    self.location[1],
                    self.location[2],
                )
            if action == 1:  # move south
                if self.is_woodcutter:
                    self.sprite = "./assets/hero.png"
                else:
                    self.sprite = "./assets/hero-g.png"
                new_location = (
                    self.location[0] + 1,
                    self.location[1],
                    self.location[2],
                )
            if action == 2:  # move west
                if self.is_woodcutter:
                    self.sprite = "./assets/hero-left.png"
                else:
                    self.sprite = "./assets/hero-left-g.png"
                new_location = (
                    self.location[0],
                    self.location[1] - 1,
                    self.location[2],
                )
            if action == 3:  # move east
                if self.is_woodcutter:
                    self.sprite = "./assets/hero-right.png"
                else:
                    self.sprite = "./assets/hero-right-g.png"
                new_location = (
                    self.location[0],
                    self.location[1] + 1,
                    self.location[2],
                )

            # try moving to new_location
            env.move(self, new_location)

        # if the agent chooses to extract
        if action == 4:
            # get the node that is directly below the agent
            node_below = env.observe(
                (self.location[0], self.location[1], self.location[2] - 1)
            )
            if node_below.kind == "WoodNode" and node_below.num_resources > 0:
                if np.random.random() < self.wood_success_rate:
                    self.wood_owned += 1
                    node_below.num_resources -= 1
                    return 0
            elif node_below.kind == "StoneNode" and node_below.num_resources > 0:
                if np.random.random() < self.stone_success_rate:
                    self.stone_owned += 1
                    node_below.num_resources -= 1
                    return 0
            return 0

        # if the agent chooses to sell wood (attempts to sell 1 unit for now)
        # for now, reward the agent for a successful trade so long as there is a market in range (visual range for now)
        if action == 5:
            if self.wood_owned < 1:
                return 0  # not enough resources on hand

            r = self.observation_spec.vision_radius
            for H in range(
                max(self.location[0] - r, 0), min(self.location[0] + r), env.height
            ):
                for W in range(
                    max(self.location[1] - r, 0), min(self.location[1] + r), env.width
                ):
                    if env.observe((H, W, self.location[2])).kind == "Buyer":
                        self.wood_owned -= 1
                        # TODO: reflect this sale on the other end somehow??
                        env.seller_score += self.sell_reward
                        return self.sell_reward
            return 0  # sell was unsuccessful

        # if the agent chooses to sell stone (attempts to sell 1 unit for now)
        if action == 6:
            if self.stone_owned < 1:
                return 0  # not enough resources on hand

            r = self.observation_spec.vision_radius
            for H in range(
                max(self.location[0] - r, 0), min(self.location[0] + r), env.height
            ):
                for W in range(
                    max(self.location[1] - r, 0), min(self.location[1] + r), env.width
                ):
                    if env.observe((H, W, self.location[2])).kind == "Buyer":
                        self.stone_owned -= 1
                        # TODO: reflect this sale on the other end somehow??
                        env.seller_score += self.sell_reward
                        return self.sell_reward
            return 0  # sell was unsuccessful

        # action invalid (we should never reach this code)
        return 0

    def is_done(self, env: GridworldEnv) -> bool:
        """Returns whether this Agent is done."""
        return env.turn >= env.max_turns


class Buyer(Agent):
    """A market (resource buyer) in the AI Economist environment."""

    def __init__(self, cfg, appearance, observation_spec: ObservationSpec, model):
        # the actions are (for now): buy wood, buy stone
        action_space = [0, 1]
        super().__init__(observation_spec, model, action_space)

        self.appearance = appearance  # the "id" of the agent

        self.buy_reward = cfg.agent.buyer.buy_reward

        self.wood_owned = 0
        self.stone_owned = 0

        self.sprite = "./assets/bank.png"

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        pass

    def pov(self, env: GridworldEnv) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        pass

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        pass

    def act(self, env: GridworldEnv, action: int) -> float:
        pass

    def is_done(self, env: GridworldEnv) -> bool:
        """Returns whether this Agent is done."""
        pass

    def add_memory(
        self, state: np.ndarray, action: int, reward: float, done: bool
    ) -> None:
        pass

    def transition(self, env: GridworldEnv) -> None:
        pass
