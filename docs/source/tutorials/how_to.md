# How to Create a Custom Experiment

We will explore how you can create your own custom experiment with a tutorial through a simple example, Treasurehunt. 
In this example, the evironment contains agents with full vision who can only move up, down, left or right, 
as well as gems that have a random chance of spawning on empty spaces. 
The agents' level of success will be measured by the game score, which is determined by how many gems that they pick up.


## Overview
The file structure for our experiment will be as follows (under the ``examples/`` directory):

```
treasurehunt
├── assets
│ └── <animation sprites>
├── data
│ └── <data records>
├── agents.py
├── entities.py
├── env.py
└── main.py
```

```{eval-rst}
.. note::
   Unlike the rest of the files in this example, the data folder may not already exist in the repository because it is ignored by git. 
   If you want to run the provided :code:`examples\treasurehunt\main.py`, you may need to create the data folder. 
```

We will create a custom environment named `Treasurehunt`, custom entities `EmptyEntity`, `Wall`, and `Gem`, and a custom agent `TreasurehuntAgent`. 
We will then write a `main.py` script that carries out the experiment, and render parts of the experiment as gifs.

Let's get started!

## The Entities
In ``entities.py``, we will create the 3 entities that we require: `EmptyEntity`, `Wall`, and `Gem`. 
All the custom entities will extend the base `Entity` class provided by Agentarium; see {class}`agentarium.primitives.Entity` 
for its attributes (including their default values) and methods.

We begin by making the necessary imports:
```python
import numpy as np

from agentarium.primitives import Entity
```

Then, we create the classes `Wall` and `Gem`, with custom constructors that overwrite default parent attribute values and include sprites used for animation later on. 
These sprites should be placed in a ``./assets/`` folder. Both `Wall` and `Gem` do not transition.

```python
class Wall(Entity):
    """An entity that represents a wall in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        self.value = -1  # Walls penalize contact
        self.sprite = (
            "~/Documents/GitHub/agentarium/examples/treasurehunt/assets/wall.png"
        )


class Gem(Entity):
    """An entity that represents a gem in the treasurehunt environment."""

    def __init__(self, gem_value):
        super().__init__()
        self.passable = True  # Agents can move onto Gems
        self.value = gem_value
        self.sprite = (
            "~/Documents/GitHub/agentarium/examples/treasurehunt/assets/gem.png"
        )
```

We then create `EmptyEntity`, which requires a custom transition method. 
Here we note that the transition method requires information such as spawn probability and gem value which must be provided through the environment. 
Therefore, we expect them to be attributes of our custom `Treasurehunt` environment.

```python
class EmptyEntity(Entity):
    """An entity that represents an empty space in the treasurehunt environment."""

    def __init__(self):
        super().__init__()
        self.passable = True  # Agents can enter EmptySpaces
        self.sprite = (
            "~/Documents/GitHub/agentarium/examples/treasurehunt/assets/sand.png"
        )

    def transition(self, env):
        """
        EmptySpaces can randomly spawn into Gems based on the item spawn probabilities dictated in the environmnet.
        """
        if (
            np.random.random() < env.spawn_prob
        ):  # NOTE: If this rate is too high, the environment gets overrun
            env.add(self.location, Gem(env.gem_value))
```

## The Environment
In ``env.py``, we will create the environment of our experiment: `Treasurehunt`. 
It will extend the base `GridworldEnv` class provided by Agentarium; 
see {class}`agentarium.primitives.GridworldEnv` for its attributes and methods.

We write the import statements:
```python
# Import base packages
import numpy as np

# Import primitive types
from agentarium.primitives import GridworldEnv
# Import experiment specific classes
from examples.treasurehunt.entities import EmptyEntity, Gem, Wall
```

We create the constructor first. In addition to the attributes from `GridworldEnv`, we add the attributes `self.gem_value` 
and `self.spawn_prob` as noted above. We also add the attributes `self.max_turns`, `self.agents`, and `self.game_score` to
allow access to and modification of parts of the environment at the experiment level later.
```python
class Treasurehunt(GridworldEnv):
    """
    Treasurehunt environment.
    """

    def __init__(self, height, width, gem_value, spawn_prob, max_turns, agents):
        layers = 2
        default_entity = EmptyEntity()
        super().__init__(height, width, layers, default_entity)

        self.gem_value = gem_value
        self.spawn_prob = spawn_prob
        self.agents = agents
        self.max_turns = max_turns

        self.game_score = 0
        self.populate()
```

We delegate the task of actually filling in the entities and constructing `self.world` to the method `populate()`:
```python
    def populate(self):
        """
        Populate the treasurehunt world by creating walls, then randomly spawning 1 gem and 1 agent.
        Note that every space is already filled with EmptyEntity as part of super().__init__().
        """
        valid_spawn_locations = []
        
        for index in np.ndindex(self.world.shape):
            y, x, z = index
            if y in [0, self.height - 1] or x in [0, self.width - 1]:
                # Add walls around the edge of the world (when indices are first or last)
                self.add(index, Wall())
            else:
                # valid spawn location
                valid_spawn_locations.append(index)

        # spawn the agents
        agent_locations = np.random.choice(
            np.array(valid_spawn_locations), size=len(self.agents), replace=False
        )
        for loc, agent in zip(agent_locations, self.agents):
            loc = tuple(loc)
            self.add(loc, agent)
```

We override the parent's `reset()` method with our slightly modified version, 
since we already keep track of the agents that need to be reset (unlike in `GridworldEnv`) and have an additional attribute that needs resetting. 
(See {func}`agentarium.primitives.GridworldEnv.reset` and {func}`agentarium.primitives.GridworldEnv.create_world` for details.)
```python
    def reset(self):
        """Reset the environment and all its agents."""
        self.create_world()
        self.game_score = 0
        self.populate()
        for agent in self.agents:
            agent.reset(self)
```


## The Agent
In ``agents.py``, we will create the agent for our experiment: `TreasurehuntAgent`. 
It will extend the base `Agent` class provided by Agentarium; 
see {class}`agentarium.primitives.Agent` for its attributes and methods. 

We make our imports:
````python
import numpy as np

from agentarium.primitives import Agent, GridworldEnv
````

We make our custom constructor:
```python
class TreasurehuntAgent(Agent):
    """
    A treasurehunt agent that uses the iqn model.
    """

    def __init__(self, observation, model):
        action_space = [0, 1, 2, 3]  # up, down, left, right
        super().__init__(observation, model, action_space)

        self.sprite = (
            "~/Documents/GitHub/agentarium/examples/treasurehunt/assets/hero.png"
        )
```
We will use `Observation` for `TreasurehuntAgent`'s observation, and `PyTorchIQN` for `TreasurehuntAgent`'s model.
We do not create them in this file (they will be passed into `TreasurehuntAgent` externally), 
but we will use the functionality that they provide by accessing the attributes of this class.


Note that unlike the other base classes we've worked on top of so far, `Agent` is an abstract class, and every custom agent that extends it must implement the methods 
`reset()`, `pov()`, `get_action()`, `act()`, and `is_done()`. Let's go through them one by one. 

To implement {func}`agentarium.primitives.Agent.reset`, we add a number of all zero SARD's to the agent's model's memory that is equal to the number of frames that it can access.
The "zero state" is obtained by getting the shape of the state observed by this agent through `np.prod(self.model.input_size)`, 
and then creating an all zeros array with the same shape.
```python
    def reset(self) -> None:
        """Resets the agent by fill in blank images for the memory buffer."""
        state = np.zeros_like(np.prod(self.model.input_size))
        action = 0
        reward = 0.0
        done = False
        for i in range(self.model.num_frames):
            self.add_memory(state, action, reward, done)
```

To implement {func}`agentarium.primitives.Agent.pov`, we get the observed image using the provided `observe()` function from the Observation class 
(in Channels x Height x Width), and then returning the flattened image. 
```python
    def pov(self, env: GridworldEnv) -> np.ndarray:
        """Returns the state observed by the agent, from the flattened visual field."""
        image = self.observation.observe(env, self.location)
        # flatten the image to get the state
        return image.reshape(1, -1)
```

To implement {func}`agentarium.primitives.Agent.get_action`, we stack the current state with the previous states in the model's memory buffer, 
and pass the stacked frames (as a horizontal vector) into the model to obtain the action chosen. 
```python
    def get_action(self, state: np.ndarray) -> int:
        """Gets the action from the model, using the stacked states."""
        prev_states = self.model.memory.current_state(
            stacked_frames=self.model.num_frames - 1
        )
        stacked_states = np.vstack((prev_states, state))

        model_input = stacked_states.reshape(1, -1)
        action = self.model.take_action(model_input)
        return action
```

To implement {func}`agentarium.primitives.Agent.act`, we calculate the new locaiton based on the action taken, 
record the reward obtained based on the entity at the new location, then try to move the agent to the new location using the provided {func}`GridworldEnv.move()`. 
```python
    def act(self, env: GridworldEnv, action: int) -> float:
        """Act on the environment, returning the reward."""

        new_location = tuple()
        if action == 0:  # UP
            new_location = (self.location[0] - 1, self.location[1], self.location[2])
        if action == 1:  # DOWN
            new_location = (self.location[0] + 1, self.location[1], self.location[2])
        if action == 2:  # LEFT
            new_location = (self.location[0], self.location[1] - 1, self.location[2])
        if action == 3:  # RIGHT
            new_location = (self.location[0], self.location[1] + 1, self.location[2])

        # get reward obtained from object at new_location
        target_object = env.observe(new_location)
        reward = target_object.value
        env.game_score += reward

        # try moving to new_location
        env.move(self, new_location)

        return reward
```

Finally, we implement {func}`agentarium.primitives.Agent.act` by checking if the current turn (tracked by default in {attr}`GridworldEnv.turn`) 
exceeds the maximum number of turns. 
```python
    def is_done(self, env: GridworldEnv) -> bool:
        """Returns whether this Agent is done."""
        return env.turn >= env.max_turns
```

Now, we are all done with our custom classes. Time to set up the actual experiment!


## The Experiment Script: `main.py`
First, we make our imports as usual:
```python
# general imports
import torch

# agentarium imports
from agentarium.models.pytorch import PyTorchIQN
from agentarium.observation.observation import Observation
from agentarium.utils.visualization import (animate, image_from_array,
                                            visual_field_sprite)
# imports from our example
from examples.treasurehunt.agents import TreasurehuntAgent
from examples.treasurehunt.env import Treasurehunt
```

Then, we will define our experiment parameters as global constants:
```python
# experiment parameters
EPOCHS = 1000
MAX_TURNS = 100
EPSILON_DECAY = 0.001
ENTITY_LIST = ["EmptyEntity", "Wall", "Gem", "TreasurehuntAgent"]
RECORD_PERIOD = 50  # how many epochs in each data recording period
```
These parameters, as well as the world configuration and model hyperparameters later, can be extracted from this script for faster and easier adjustments using configuration files. 
Here is a [quick tutorial](/configuration_files.md).

We will first construct the observations, the models, the agents, and the environment. 
The entities will not need to be constructed explicitly as they will be generated by the environment.
```python
def setup() -> Treasurehunt:
    """Set up all the whole environment and everything within."""
    # world configurations
    height = 7
    width = 7
    gem_value = 10
    spawn_prob = 0.002

    # make the agents
    agent_num = 2
    agents = []
    for _ in range(agent_num):
        observation = Observation(ENTITY_LIST)

        model = PyTorchIQN(
            input_size=(len(ENTITY_LIST), height, width),
            action_space=4,
            layer_size=250,
            epsilon=0.7,
            device="cpu",
            seed=torch.random.seed(),
            num_frames=5,
            n_step=3,
            sync_freq=200,
            model_update_freq=4,
            BATCH_SIZE=64,
            memory_size=1024,
            LR=0.00025,
            TAU=0.001,
            GAMMA=0.99,
            N=12,
        )

        agents.append(TreasurehuntAgent(observation, model))

    # make the environment
    env = Treasurehunt(height, width, gem_value, spawn_prob, MAX_TURNS, agents)
    return env
```

Then, we will run the experiment. Most of the work here is done by calling {func}`GridworldEnv.take_turn()`, 
which transitions every entity in the environment, then every agent, then increments the turn count by one. 
In addition to printing information about each recording period on the terminal, 
we also use functions from {mod}`agentarium.utils.visualization` to record states as images and animate them into a gif.
```python
def run(env: Treasurehunt):
    """Run the experiment."""
    imgs = []
    total_score = 0
    total_loss = 0
    for epoch in range(EPOCHS):
        # Reset the environment at the start of each epoch
        env.reset()
        for agent in env.agents:
            agent.model.start_epoch_action(**locals())

        while not env.turn >= env.max_turns:
            if epoch % RECORD_PERIOD == 0:
                full_sprite = visual_field_sprite(env)
                imgs.append(image_from_array(full_sprite))

            env.take_turn()

        # At the end of each epoch, train as long as the batch size is large enough.
        if epoch > 10:
            for agent in env.agents:
                loss = agent.model.train_model()
                total_loss += loss

        # Special action: update epsilon
        for agent in env.agents:
            new_epsilon = agent.model.epsilon - EPSILON_DECAY
            agent.model.epsilon = max(new_epsilon, 0.01)

        total_score += env.game_score

        if epoch % RECORD_PERIOD == 0:
            avg_score = total_score / RECORD_PERIOD
            print(
                f"Epoch: {epoch}; Epsilon: {env.agents[0].model.epsilon}; Losses this period: {total_loss}; Avg. score this period: {avg_score}"
            )
            animate(imgs, f"treasurehunt_epoch{epoch}", "../data/")

            # reset the data
            imgs = []
            total_score = 0
            total_loss = 0
```

Finally, write the main block:
```python
if __name__ == "__main__":
    env = setup()
    run(env)
```

And we're done! You can run this script from command line, and see the animations in `treasurehunt\data`.
