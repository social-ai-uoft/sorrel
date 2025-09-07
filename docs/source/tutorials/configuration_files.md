# Configuration Files

## Introduction
When running experiments, it can be beneficial to be able to modify different parameters quickly. 
For this, we provide the functionality to read ``yaml`` configuration files through [Hydra](https://hydra.cc/).
It is recommended that the constructors of your customized primitive classes use a `config` to initialize attributes, 
especially attributes that may need to be tweaked between different experiments.


## Basic Usage Example

Suppose that you expect to tweak the number of apples in your environment frequently.
You might create a config ``yaml`` file that contains the following lines:
```python
# ...

env:
    # ...
    num_apples: 5

# ...
```

And write the constructor like so:
```python
class MyGridworld(Gridworld):

    num_apples: int
    num_bananas: int
    
    def __init__(self, config: DictConfig, num_bananas: int):
        # ...
        self.num_apples = config.env.num_apples
        self.num_bananas = num_bananas
```

Finally, tell Hydra which config file to use in the main function where you run your experiment:
```python

@hydra.main(version_base=None, config_path="<path to configs folder>", config_name="<config file name>")
def main(cfg: DictConfig):
  env = MyGridworld(config=cfg, num_bananas=2)
  # ...
```

This way, if you wish to modify `num_apples` between experiments, you can simply modify the `num_apples` value in your config file, 
instead of having to modify the code itself as is the case for `num_bananas`.

You can also override the value of `num_apples` from commandline like so: ``python my_app.py ++env.num_apples=10``

You can use the configs using attribute style access (`cfg.env.num_apples`) or dictionary style access (`cfg["env"]["num_apples"]`). For more detailed tutorials, see the [Hydra Documentation](https://hydra.cc/docs/tutorials/intro/).

## A Sample Config File

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

root:
log:
```