# Configuration Files

## Introduction
When running experiments, it can be beneficial to be able to modify different parameters quickly. 
For this, we provide the functionality to read ``yaml`` configuration files through a `Cfg` class.
It is recommended that the constructors of your customized primitive classes use a `Cfg` object to initialize attributes, 
especially attributes that may need to be tweaked between different experiments.

For example, if you expect to tweak the number of apples in your environment frequently, 
you might create a config ``yaml`` file that contains the following lines:
```python
# ...

env:
    # ...
    num_apples: 5

# ...
```
And write the constructor like so:
```python
class MyGridworldEnv(GridworldEnv):

    num_apples: int
    num_bananas: int
    
    def __init__(self, config: Cfg, num_bananas: int):
        # ...
        self.num_apples = config.env.num_apples
        self.num_bananas = num_bananas
```
This way, if you wish to modify `num_apples` between experiments, you can simply modify the `num_apples` value in your config file, 
instead of having to modify the code itself as is the case for `num_bananas`.

## Making and Using Config Files
Start by creating a ``config`` directory in your experiment folder, and creating a file in the new directory called ``config.yaml``.

Here is a template that contains some frequently used parameters to get you started.
```yaml
experiment:
  name:
  epochs:
  max_turns:

env:
  height:
  width:
  layers:
  default_object:

model:
  model_type:

agent:
  agent_type:
    num:
    model:
    appearance:

root: '~/Documents/GitHub/sorrel'
log:
```

To load the config file into a `Cfg` object named `cfg`, do:

```python
cfg = load_config(argparse.Namespace(config='../configs/config.yaml'))
```

Afterwards, all configuration parameters will be stored in `cfg` as attributes.
You can then access using syntax like `cfg.agent.agent_type.num`.

```{eval-rst}
For more information, including type hints for common config parameters, see documentation for the :py:class:`Cfg` class.
```
