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

We will create a custom environment named `Treasurehunt`, custom entities `EmptyEntity`, `Wall`, `Sand`, and `Gem`, and a custom agent `TreasurehuntAgent`.
The environment will have two layers: `TreasurehuntAgent` and `EmptyEntity` will be on the top layer, and `Sand` will be on the bottom layer.
We will then write a `main.py` script that implements the custom experiment `TreasurehuntExperiment`, which will allow us to run and record the experiment.

Let's get started!

## The Entities
In ``entities.py``, we will create the 3 entities that we require: `EmptyEntity`, `Wall`, and `Gem`. 
All the custom entities will extend the base `Entity` class provided by Sorrel; see {class}`sorrel.entities.Entity` 
for its attributes (including their default values) and methods. Note that the `Entity` class uses a Generic type; we will specify the type as `Treasurehunt` in our custom entities as that is the environment that our entities are compatible with.

We begin by making the necessary imports:
```{literalinclude} /../../sorrel/examples/treasurehunt/entities.py
:start-after: begin imports
:end-before: end imports
```

Then, we create the classes `Wall`, `Sand`, and `Gem`, with custom constructors that overwrite default parent attribute values and include sprites used for animation later on.
These sprites should be placed in a ``./assets/`` folder. All of these entities do not transition.
```{literalinclude} /../../sorrel/examples/treasurehunt/entities.py
:pyobject: Wall
```
```{literalinclude} /../../sorrel/examples/treasurehunt/entities.py
:pyobject: Sand
```
```{literalinclude} /../../sorrel/examples/treasurehunt/entities.py
:pyobject: Gem
```
```{note}
We use `Path(__file__)` to ensure that the animation sprite paths are always relative to the path to this `entities.py` file, no matter where one may be running this code from.
```

We then create `EmptyEntity`, which requires a custom transition method.
Here we note that the transition method requires information such as spawn probability and gem value which must be provided through the environment. 
Therefore, we expect them to be attributes of our custom `Treasurehunt` environment.
```{literalinclude} /../../sorrel/examples/treasurehunt/entities.py
:pyobject: EmptyEntity
```

## The Environment
In ``env.py``, we will create the environment of our experiment: `Treasurehunt`. 
It will extend the base `GridworldEnv` class provided by Sorrel; 
see {py:obj}`sorrel.environments.GridworldEnv` for its attributes and methods.

We write the import statements:
```{literalinclude} /../../sorrel/examples/treasurehunt/env.py
:start-after: begin imports
:end-before: end imports
```

We create the constructor. In addition to the attributes from `GridworldEnv`, we add the attributes `self.gem_value` 
and `self.spawn_prob` as noted above. We also add the attributes `self.max_turns` so that it can be accessed by the agents to determine if they are Done after an action.
```{literalinclude} /../../sorrel/examples/treasurehunt/env.py
:start-after: begin treasurehunt
:end-before: end treasurehunt
```

Note that the environment is very barebones. The task of actually filling in the entities and constructing the world is delegated to our custom experiment class, as we will see in a moment.

## The Agent
In ``agents.py``, we will create the agent for our experiment: `TreasurehuntAgent`. 
It will extend the base `Agent` class provided by Sorrel; 
see {class}`sorrel.agents.Agent` for its attributes and methods. Note that `Agent` is a generic subclass of `Entity` so just like with our custom entities, we need to specify the type of environment that the custom agent is compatible with when inheriting from `Agent`.

We make our imports:
```{literalinclude} /../../sorrel/examples/treasurehunt/agents.py
:start-after: begin imports
:end-before: end imports
```

We make our custom constructor:
```{literalinclude} /../../sorrel/examples/treasurehunt/agents.py
:start-after: begin treasurehunt agent
:end-before: end constructor
```

We will use {py:obj}`sorrel.observation.obvservation_spec.OneHotObservationSpec` for `TreasurehuntAgent`'s observation, {py:obj}`sorrel.action.action_spec.ActionSpec` for `TreasurehuntAgent`'s actions, and `sorrel.models.pytorch.PyTorchIQN` for `TreasurehuntAgent`'s model.
We do not create them in this file (they will be passed into `TreasurehuntAgent`'s constructor externally), 
but we will use the functionality that they provide by accessing the attributes of this class.

Note that unlike the other base classes we've worked on top of so far, `Agent` is an abstract class, and every custom agent that extends it must implement the methods 
`reset()`, `pov()`, `get_action()`, `act()`, and `is_done()`. Let's go through them one by one. 

To implement {func}`sorrel.agents.Agent.reset`, we call the agent's model's reset function. This is [required of all sorrel models that inherit the base model class](#sorrel.models.base_model.BaseModel.reset).
```{literalinclude} /../../sorrel/examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.reset
```

To implement {func}`sorrel.agents.Agent.pov`, we get the observed image (in Channels x Height x Width) 
using the provided [OneHotObservationSpec.observe()](#sorrel.observation.observation_spec.OneHotObservationSpec.observe) function, and then returning the flattened image. 
```{literalinclude} /../../sorrel/examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.pov
```

To implement {func}`sorrel.agents.Agent.get_action`, we stack the current state with the previous states in the model's memory buffer, 
and pass the stacked frames (as a horizontal vector) into the model to obtain the action chosen. (See [BaseModel.take_action](#sorrel.models.base_model.BaseModel.take_action))
```{literalinclude} /../../sorrel/examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.get_action
```

To implement {func}`sorrel.agents.Agent.act`, we calculate the new location based on the action taken, 
record the reward obtained based on the entity at the new location, then try to move the agent to the new location using the provided [GridworldEnv.move()](#sorrel.environments.GridworldEnv.move). 
```{literalinclude} /../../sorrel/examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.act
```

Finally, we implement {func}`sorrel.agents.Agent.is_done` by checking if the current turn (tracked by default in [GridworldEnv.turn](#sorrel.environments.GridworldEnv.turn)) 
exceeds the maximum number of turns. 
```{literalinclude} /../../sorrel/examples/treasurehunt/agents.py
:pyobject: TreasurehuntAgent.is_done
```

Now, we are all done with our custom classes. Time to set up the actual experiment!

## The Experiment Script: `main.py`

First, we make our imports as usual:
```{literalinclude} /../../sorrel/examples/treasurehunt/main.py
:start-after: begin imports
:end-before: end imports
```

We will now write our custom experiment class by inheriting the {class}`sorrel.experiment.Experiment` class that has an already implemented [run()](#sorrel.experiment.Experiment.run) method which will run the experiment for us. Much like the custom entities and agents, we need to specify the environment this custom experiment is using when inheriting from the generic experiment.

```{literalinclude} /../../sorrel/examples/treasurehunt/main.py
:start-after: begin treasurehunt experiment
:end-before: end constructor
```

Note that the experiment takes in a `config` that can be accessed at `self.config` which stores the configurations used for this experiment. Certain config values are required when using the default methods: see [the documentation](#sorrel.experiment.Experiment) for more details.

Like `Agent`, `Experiment` requires us to implement two abstract methods.

The first is {func}`sorrel.experiment.Experiment.setup_agents`, where we create the agents used in this specific experiment and save them in the attribute `self.agents`:

```{literalinclude} /../../sorrel/examples/treasurehunt/main.py
:pyobject: TreasurehuntExperiment.setup_agents
```

The second is {func}`sorrel.experiment.Experiment.populate_environment`, where we create all entities and populate `self.env` with the entities as well as the agents.
```{literalinclude} /../../sorrel/examples/treasurehunt/main.py
:pyobject: TreasurehuntExperiment.populate_environment
```

```{eval-rst}
.. note::
   We had to work around :code:`np.random.choice` a little in order to use it. 
   We have specifically avoided using `random.choices` because we would then need to seed np.random and random separately 
   for reproducible results. It's generally a good idea to choose one random generator and only use that across the scope of your example.
```

Lastly, we will run the experiment. 
Most of the work is done by calling the [Experiment.run()](#sorrel.experiment.Experiment.run) method. 

```{literalinclude} /../../sorrel/examples/treasurehunt/main.py
:start-after: begin main
:end-before: end main
```

Here, we use a dictionary to store our configs for the experiment and pass in constants for the environment parameters. In general, we recommend using configuration files for a more clean and centralized approach: here's a [quick tutorial](./configuration_files.md).

And we're done! You can run this script from command line, and see the animations in `treasurehunt\data`.
