"""Agent classes for the AI Economist task. Includes the resource collectors and the market deciders."""

from typing import TYPE_CHECKING
from omegaconf import DictConfig
import numpy as np

from sorrel.action.action_spec import ActionSpec
from sorrel.agents import Agent
from sorrel.entities import Entity
from sorrel.models.base_model import BaseModel
from sorrel.observation.observation_spec import ObservationSpec, OneHotObservationSpec

from sorrel.examples.ai_econ.entities import EmptyEntity

#from examples.treasurehunt import env

if TYPE_CHECKING:
    from sorrel.examples.ai_econ.world import EconWorld


class EconEnvObsSpec(OneHotObservationSpec):

    def observe(
        self,
        env: "EconWorld",
        location: tuple | None = None,
        resources: list[int] | None = None,
    ):
        visual_field = super().observe(env, location).flatten()
        if resources is not None:
            resources = np.array(resources)
        else:
            resources = np.array([0, 0])
        return np.concatenate((visual_field, resources))
    
class Beam(Entity["EconWorld"]):
    """Generic beam class for agent beams."""

    def __init__(self):
        super().__init__()
        self.turn_counter = 0
        self.has_transitions = True

    def transition(self, env: "EconWorld"):
        # Beams persist for one full turn, then disappear.
        if self.turn_counter >= 1:
            env.add(self.location, EmptyEntity())
        else:
            self.turn_counter += 1

class SellWoodBeam(Beam):
    def __init__(self):
        super().__init__()
        self.sprite = f"./assets/beam.png"


class SellStoneBeam(Beam):
    def __init__(self):
        super().__init__()
        self.sprite = f"./assets/zap.png"

class MarketNavBeam(Beam):
    def __init__(self):
        super().__init__()
        self.sprite = f"./assets/Pure_violet.webp" # Different color from selling beams for visual clarity
        self.kind = "SellWoodBeam"  


class Seller(Agent):
    """A resource gatherer in the AI Economist environment."""

    def __init__(
        self,
        config: DictConfig,
        appearance: list,
        is_woodcutter: bool,
        is_majority: bool,
        observation_spec: EconEnvObsSpec,
        action_spec: ActionSpec,
        model: BaseModel,
    ):
        # the actions are: move north, south, west, east, extract resource, sell wood, sell stone
        super().__init__(observation_spec, action_spec=action_spec, model=model)

        self.config = config

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
            self.wood_success_rate = config.agent.seller.skilled_success_rate
            self.stone_success_rate = config.agent.seller.unskilled_success_rate
        else:
            self.wood_success_rate = config.agent.seller.unskilled_success_rate
            self.stone_success_rate = config.agent.seller.skilled_success_rate

        self.sell_reward = config.agent.seller.sell_reward

        self.wood_owned = 0
        self.stone_owned = 0

        if self.is_woodcutter:
            self.sprite = "./assets/hero.png"
        else:
            self.sprite = "./assets/hero-g.png"

            # To note whether we're close to a market (to use regular actions)
            self.near_market = False
            self.market_locations_cache = None
            self.last_market_check = 0


    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        state = np.zeros_like(np.prod(self.model.input_size))
        action = (0, 0.0)
        reward = 0.0
        done = False
        for _ in range(self.config.agent.seller.obs.num_frames):  
            super().add_memory(state, action, reward, done)  
            #self.add_memory(state, action, reward, done)

        self.near_market = False
        self.market_locations_cache = None
        self.last_market_check = 0
        self.last_action_was_7 = False

    def pov(self, env: "EconWorld") -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation_spec.observe(
            env, self.location, [self.wood_owned, self.stone_owned]
        )

        # flatten the image to get the state
        return image.reshape(1, -1)


    def compute_taxicab_distance(self, location, targets: list[tuple]) -> np.array:
        
        distances = []
        # Get taxicab distance for each target in the list
        for target in targets:
            distance = sum([abs(x - y) for x, y in zip(location, target)])
            distances.append(distance)

        return np.array(distances)

    def find_market_locations(self, env: "EconWorld") -> list[tuple]:
        """Find all market locations in the environment"""
        # Use old market locations if available
        if self.market_locations_cache is not None and env.turn - self.last_market_check < 10:
            return self.market_locations_cache
            
        market_locations = []
        
        # Find center of the grid with markets for reference 
        center_row = env.height // 2
        center_col = env.width // 2
        
        # The 4 markets are placed at these offsets from center
        market_offsets = [(-4, -4), (-4, 4), (4, -4), (4, 4)]
        
        for offset in market_offsets:
            # Calculate market position and use same layer as agent
            market_pos = (center_row + offset[0], center_col + offset[1], self.location[2])
            market_locations.append(market_pos)
        
        # Cache the market locations and update check time
        self.market_locations_cache = market_locations
        self.last_market_check = env.turn
        
        return market_locations

    def can_see_market(self, env: "EconWorld") -> bool:
        """Check if the agent can see any market within its vision radius"""
        r = self.observation_spec.vision_radius
        
        for H in range(
            max(self.location[0] - r, 0), min(self.location[0] + r + 1, env.height)
        ):
            for W in range(
                max(self.location[1] - r, 0), min(self.location[1] + r + 1, env.width)
            ):
                entity = env.observe((H, W, self.location[2]))
                if entity.kind == "Buyer":
                    return True
        
        return False
    
    def movement(self, action):

        if action == 0:  # move north
            return (self.location[0] - 1, self.location[1], self.location[2])
        elif action == 1:  # move south
            return (self.location[0] + 1, self.location[1], self.location[2])
        elif action == 2:  # move west
            return (self.location[0], self.location[1] - 1, self.location[2])
        elif action == 3:  # move east
            return (self.location[0], self.location[1] + 1, self.location[2])
        else:
            return self.location

    def chase_market(self, env: "EconWorld") -> int:
        """Chase the nearest market location using manhattan distance"""

        # Get market locations
        markets = self.find_market_locations(env)
        
        # Compute taxicab distance to all markets
        distances = self.compute_taxicab_distance(self.location, markets)
        
        # Choose market with minimum distance
        min_locs = np.where(distances == distances.min())[0]
        chosen_market_idx = np.random.choice(min_locs)
        chosen_market = markets[chosen_market_idx]
        
        # Compute possible paths
        ACTIONS = [0, 1, 2, 3]  # north, south, west, east
        TOO_FAR = 999999999
        attempted_paths = [self.movement(action) for action in ACTIONS]
        
        # Calculate distances from each possible new position to the target market
        path_distances = []
        for path in attempted_paths:
            if self.is_valid_location(env, path):
                # Calculate distance from this path to the market
                distance = sum([abs(x - y) for x, y in zip(path, chosen_market)])
                path_distances.append(distance)
            else:
                path_distances.append(TOO_FAR)
        
        path_distances = np.array(path_distances)
        
        # Choose an action that minimizes the taxicab distance
        candidate_actions = np.where(path_distances == path_distances.min())[0]
        chosen_action = np.random.choice(candidate_actions)
        
        ##### If the chosen action is 7, we need to return a tuple for PPO compatibility (ADJUST THIS HWEHWEHWEHWEH)

        #if hasattr(self, 'model') and self.model == 'ppo' and isinstance(chosen_action, (int, np.integer)):
            #return (tuple([chosen_action]), 0)

        return int(chosen_action)

    def is_valid_location(self, env: "EconWorld", location: tuple) -> bool:
        """Check if a location is valid (no walls or other obstacles)"""
        # Check if the location is within bounds
        if (location[0] < 0 or location[0] >= env.height or
            location[1] < 0 or location[1] >= env.width):
            return False
        
        entity = env.observe(location)
            
        # Check if the location is passable (this will now catch walls)
        if hasattr(entity, 'passable') and not entity.passable:
            return False
            
        # Check if there's an agent already at this location
        for agent in env.woodcutters + env.stonecutters:
            if agent.location == location and agent != self:
                return False
                
        return True


    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        try:
            # Try with stacked_frames parameter first
            prev_states = self.model.memory.current_state(
                stacked_frames=self.config.agent.seller.obs.num_frames - 1
            )
        except TypeError:
            # If PPO doesn't support stacked_frames, just get current state
            prev_states = self.model.memory.current_state()
    
        stacked_states = np.vstack((prev_states, state))
        model_input = stacked_states.reshape(1, -1)
        
        # Store environment reference for future use if we don't have it yet
        if not hasattr(self, 'current_env'):
            if hasattr(self.model, 'env'):
                self.current_env = self.model.env

        # Get environment reference if available
        env = getattr(self, 'current_env', None)
            
        # If we have access to the environment, check for markets
        if env is not None:
            # Update the market visibility status every few turns
            # Access turn from the experiment, not the world
            if hasattr(self, 'current_env') and hasattr(self.current_env, 'turn'):
                if self.current_env.turn % 5 == 0:
                    self.near_market = self.can_see_market(env)
        
        # Check if homing is enabled for markets before allowing action 7
        homing_enabled = False
        if env is not None and hasattr(env, 'markets') and env.markets:
            # Enable the homing action if homing = True and the agent cannot see a market
            homing_enabled = all(
                hasattr(market, 'homing') and market.homing
                for market in env.markets
            ) and not self.near_market 
        
        raw_action = self.model.take_action(model_input)
        
        if isinstance(raw_action, tuple):
            raw_action = raw_action[0]  # Get just the action part

        # If homing is false or we're near a market, disable action 7 
        if raw_action == 7 and not homing_enabled:
            action = np.random.randint(0, 7)  # Choose an action other than 7
        else:
            action = raw_action
                
        return action



    def act(self, env: "EconWorld", action: int) -> float:
        """Act on the environment, returning the reward."""

        # Store environment reference for future use
        if not hasattr(self, 'current_env') or self.current_env is None:
            self.current_env = env

        # Update market visibility before beginning action
        self.near_market = self.can_see_market(env)

        # If agent can see market, disable action 7
        if self.near_market and action == 7:
            action = np.random.randint(0, 7)  # Choose any action except 7
        
        # First time action 7 is chosen, enter navigation state
        elif action == 7:
            # Check if all markets have homing enabled
            homing_enabled = False
            if hasattr(env, 'markets') and env.markets:
                homing_enabled = all(
                    hasattr(market, 'homing') and market.homing
                    for market in env.markets
                )
                    
            # If homing is disabled, replace with random movement action
            if not homing_enabled:
                action = np.random.randint(0, 7)  # Choose any action except 7
            else:
                # Create a visual beam to indicate market navigation
                beam_loc = (self.location[0], self.location[1], 3)
                env.add(beam_loc, MarketNavBeam())
                
                # Use the chase_market function to get the best movement action
                movement_action = self.chase_market(env)
                
                # Execute the movement action instead
                action = movement_action
                
                # Note that action 7 was chosen this turn so future actions will auto-navigate
                self.last_action_was_7 = True
        
        # Continue automated navigation if we had previously chosen action 7 and haven't found market yet
        elif hasattr(self, 'last_action_was_7') and self.last_action_was_7 and not self.near_market:
            # Use the chase_market function to get the best movement action
            action = self.chase_market(env)
        else:
            # Reset navigation state if we see a market
            if hasattr(self, 'last_action_was_7'):
                self.last_action_was_7 = False

        # MOVEMENT (0–3)
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

        # EXTRACT RESOURCE (4)
        if action == 4:
            node_below = env.observe(
                (self.location[0], self.location[1], self.location[2] - 1)
            )
            if node_below.kind == "WoodNode" and node_below.num_resources > 0:
                if np.random.random() < self.wood_success_rate:
                    self.wood_owned += 1
                    node_below.num_resources -= 1
                    return 1 #### Changed this (From 0 to 1)
            elif node_below.kind == "StoneNode" and node_below.num_resources > 0:
                if np.random.random() < self.stone_success_rate:
                    self.stone_owned += 1
                    node_below.num_resources -= 1
                    return 1 ################# Changed this (From 0 to 1)
            return 0

        # SELL WOOD (5)
        if action == 5:
            beam_loc = (self.location[0], self.location[1], 3)
            env.add(beam_loc, SellWoodBeam())

            if self.wood_owned < 1:
                return 0  # not enough resources

            r = self.observation_spec.vision_radius
            for H in range(
                max(self.location[0] - r, 0), min(self.location[0] + r, env.height)
            ):
                for W in range(
                    max(self.location[1] - r, 0), min(self.location[1] + r, env.width)
                ):
                    if env.observe((H, W, self.location[2])).kind == "Buyer":
                        self.wood_owned -= 1
                        env.seller_score += self.sell_reward
                        return self.sell_reward
            return 0

        # SELL STONE (6)
        if action == 6:
            beam_loc = (self.location[0], self.location[1], 3)
            env.add(beam_loc, SellStoneBeam())

            if self.stone_owned < 1:
                return 0  # not enough resources

            r = self.observation_spec.vision_radius
            for H in range(
                max(self.location[0] - r, 0), min(self.location[0] + r, env.height),
            ):
                for W in range(
                    max(self.location[1] - r, 0), min(self.location[1] + r, env.width),
                ):
                    if env.observe((H, W, self.location[2])).kind == "Buyer":
                        self.stone_owned -= 1
                        env.seller_score += self.sell_reward
                        return self.sell_reward
            return 0

        # Fallback
        return 0


    def is_done(self, env: "EconWorld") -> bool:
        """Returns whether this Agent is done."""
        return False
        #return env.turn >= env.max_turns (the old one)

    def add_memory(self, state: np.ndarray, action, reward: float, done: bool) -> None:
        """Override add_memory to handle PPO's tuple requirement."""
        # PPO expects (action, log_prob) tuple, but framework passes single action
        if not isinstance(action, tuple):
            # Create a dummy log_prob for the action
            if hasattr(action, 'item'):
                action_val = action.item()
            else:
                action_val = action
            action = (action_val, 0.0)  # (action, dummy_log_prob)
        
        # Now call parent's add_memory with the proper tuple
        super().add_memory(state, action, reward, done)

class Buyer(Agent):
    """A market (resource buyer) in the AI Economist environment."""

    def __init__(self, config, appearance, observation_spec: ObservationSpec, model):
        # the actions are (for now): buy wood, buy stone
        action_spec = ActionSpec([0, 1])
        super().__init__(observation_spec, action_spec, model)

        self.appearance = appearance  # the "id" of the agent

        self.buy_reward = config.agent.buyer.buy_reward

        self.wood_owned = 0
        self.stone_owned = 0

        self.homing = False # Toggle for manhattan distance navigation

        self.sprite = "./assets/bank.png"

    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        pass

    def pov(self, env: "EconWorld") -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        pass

    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        pass

    def act(self, env: "EconWorld", action: int) -> float:
        pass

    def is_done(self, env: "EconWorld") -> bool:
        """Returns whether this Agent is done."""
        pass

    def add_memory(
        self, state: np.ndarray, action: int, reward: float, done: bool
    ) -> None:
        pass

    def transition(self, env: "EconWorld") -> None:
        pass
