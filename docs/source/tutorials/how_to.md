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

We will create a custom environment named `Treasurehunt`, custom entities `EmptyEntity`, `Wall`, `Sand`, and `Gem`, and a custom agent `TreasurehuntAgent`.
The environment will have two layers: `TreasurehuntAgent` and `EmptyEntity` will be on the top layer, and `Sand` will be on the bottom layer.
We will then write a `main.py` script that carries out the experiment, and render parts of the experiment as gifs.

Let's get started!

## The Entities
In ``entities.py``, we will create the 3 entities that we require: `EmptyEntity`, `Wall`, and `Gem`. 
All the custom entities will extend the base `Entity` class provided by Sorrel; see {class}`sorrel.entities.Entity` 
for its attributes (including their default values) and methods.

We begin by making the necessary imports:
```{literalinclude} /../../examples/treasurehunt/entities.py
:start-after: begin imports
:end-before: end imports
```

Then, we create the classes `Wall`, `Sand`, and `Gem`, with custom constructors that overwrite default parent attribute values and include sprites used for animation later on. 
These sprites should be placed in a ``./assets/`` folder. All of these entities do not transition.
```{literalinclude} /../../examples/treasurehunt/entities.py
:pyobject: Wall
```
```{literalinclude} /../../examples/treasurehunt/entities.py
:pyobject: Sand
```
```{literalinclude} /../../examples/treasurehunt/entities.py
:pyobject: Gem
```

We then create `EmptyEntity`, which requires a custom transition method.
Here we note that the transition method requires information such as spawn probability and gem value which must be provided through the environment. 
Therefore, we expect them to be attributes of our custom `Treasurehunt` environment.
```{literalinclude} /../../examples/treasurehunt/entities.py
:pyobject: EmptyEntity
```

## The Environment
In ``env.py``, we will create the environment of our experiment: `Treasurehunt`. 
It will extend the base `GridworldEnv` class provided by Sorrel; 
see {py:obj}`sorrel.environments.GridworldEnv` for its attributes and methods.

We write the import statements:
```{literalinclude} /../../examples/treasurehunt/env.py
:start-after: begin imports
:end-before: end imports
```

We create the constructor first. In addition to the attributes from `GridworldEnv`, we add the attributes `self.gem_value` 
and `self.spawn_prob` as noted above. We also add the attributes `self.max_turns`, `self.agents`, and `self.game_score` 
so that we can access these attributes of the environment at the experiment level later.
```{literalinclude} /../../examples/treasurehunt/env.py
:lines: 15-19
```
```{literalinclude} /../../examples/treasurehunt/env.py
:pyobject: Treasurehunt.__init__
```

We delegate the task of actually filling in the entities and constructing `self.world` to the method `populate()`:
```{literalinclude} /../../examples/treasurehunt/env.py
:pyobject: Treasurehunt.populate
```

```{eval-rst}
.. note::
   We had to work around :code:`np.random.choice` a little in order to use it. 
   We have specifically avoided using `random.choices` because we would then need to seed np.random and random separately 
   for reproducible results. It's generally a good idea to choose one random generator and only use that across the scope of your example.
```

We will also write a `reset()` method to reset the environment at the end of every game, using {func}`sorrel.environments.GridworldEnv.create_world`:
```{literalinclude} /../../examples/treasurehunt/env.py
:pyobject: Treasurehunt.reset
```

## The Agent
In ``agents.py``, we will create the agent for our experiment: `TreasurehuntAgent`. 
It will extend the base `Agent` class provided by Sorrel; 
see {class}`sorrel.agents.Agent` for its attributes and methods. 

We make our imports:
```{literalinclude} /../../examples/treasurehunt/agents.py
:start-after: begin imports
:end-before: end imports
```

We make our custom constructor:
```{literalinclude} /../../examples/treasurehunt/agents.py
:lines: 13-17
```
```{literalinclude} /../../examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.__init__
```

We will use {class}`sorrel.observation.obvservation_spec.OneHotObservationSpec` for `TreasurehuntAgent`'s observation, and {class}`sorrel.models.pytorch.PyTorchIQN` for `TreasurehuntAgent`'s model.
We do not create them in this file (they will be passed into `TreasurehuntAgent`'s constructor externally), 
but we will use the functionality that they provide by accessing the attributes of this class.

Note that unlike the other base classes we've worked on top of so far, `Agent` is an abstract class, and every custom agent that extends it must implement the methods 
`reset()`, `pov()`, `get_action()`, `act()`, and `is_done()`. Let's go through them one by one. 

To implement {func}`sorrel.agents.Agent.reset`, we add a number of all zero SARD's to the agent's model's memory that is equal to the number of frames that it can access.
The "zero state" is obtained by getting the shape of the state observed by this agent through [self.model.input_size](#sorrel.models.base_model.SorrelModel.input_size), 
and then creating an all zeros array with the same shape.
```{literalinclude} /../../examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.reset
```

To implement {func}`sorrel.agents.Agent.pov`, we get the observed image (in Channels x Height x Width) 
using the provided [OneHotObservationSpec.observe()](#sorrel.observation.observation_spec.OneHotObservationSpec.observe) function, and then returning the flattened image. 
```{literalinclude} /../../examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.pov
```

To implement {func}`sorrel.agents.Agent.get_action`, we stack the current state with the previous states in the model's memory buffer, 
and pass the stacked frames (as a horizontal vector) into the model to obtain the action chosen. (See [SorrelModel.take_action](#sorrel.models.base_model.SorrelModel.take_action))
```{literalinclude} /../../examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.get_action
```

To implement {func}`sorrel.agents.Agent.act`, we calculate the new location based on the action taken, 
record the reward obtained based on the entity at the new location, then try to move the agent to the new location using the provided [GridworldEnv.move()](#sorrel.environments.gridworld.GridworldEnv.move). 
```{literalinclude} /../../examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.act
```

Finally, we implement {func}`sorrel.agents.Agent.is_done` by checking if the current turn (tracked by default in [GridworldEnv.turn](#sorrel.environments.gridworld.GridworldEnv.turn)) 
exceeds the maximum number of turns. 
```{literalinclude} /../../examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.is_done
```

Now, we are all done with our custom classes. Time to set up the actual experiment!

## The Experiment Script: `main.py`
First, we make our imports as usual:
```{literalinclude} /../../examples/treasurehunt/main.py
:start-after: begin imports
:end-before: end imports
```

Then, we will define our experiment parameters as global constants:
```{literalinclude} /../../examples/treasurehunt/main.py
:start-after: begin parameters
:end-before: end parameters
```
These parameters, as well as the world configuration and model hyperparameters later, can be extracted from this script for faster and easier adjustments using configuration files. 
Here is a [quick tutorial](./configuration_files.md).

We will first create the observation specification, the models, the agents, and the environment. 
The entities will not need to be created explicitly as they will be generated by the environment.
```{literalinclude} /../../examples/treasurehunt/main.py
:pyobject: setup
```

Then, we will run the experiment. Most of the work here is done by calling [GridworldEnv.take_turn()](#sorrel.environments.gridworld.GridworldEnv.take_turn), 
which transitions every entity in the environment, then every agent, then increments the turn count by one. 
In addition to printing information about each recording period on the terminal, 
we also use functions from {mod}`sorrel.utils.visualization` to record states as images and animate them into a gif.
```{literalinclude} /../../examples/treasurehunt/main.py
:pyobject: run
```
```{note}
Since the sprite paths for the entities are relative, the animation will only generate properly if you are running the script from its file location. If you are encountering errors of the form ```FileNotFoundError: [Errno 2] No such file or directory``` thrown by `animate()`, double check that you are working in the same directory as this `main.py`.
```

Finally, write the main block:
```{literalinclude} /../../examples/treasurehunt/main.py
:start-after: begin main
:end-before: end main
```

And we're done! You can run this script from command line, and see the animations in `treasurehunt\data`.
