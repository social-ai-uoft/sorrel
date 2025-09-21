# Examples

This folder contains example experiments using the Sorrel codebase.

## Project Structure

For consistency with other examples, we recommend that projects use the following basic structure:

```
example/
├─ assets/            # Custom images for rendering the sprites
├─ configs/           # Configuration files (optional)      
├─ data/              
│  ├─ checkpoints/    # Saved model checkpoints
│  ├─ gifs/           # Saved animated GIFs
│  └─ logs/           # Tensorboard logs
├─ agents.py          # Custom agents for the environment
├─ entities.py        # Custom entities for the environment
├─ env.py             # Environment setup
├─ main.py            # Running the experiment
└─ world.py           # World layout for agents and entities
```

Additional extensions and custom components, such as action/observation specifications, utilities, can be added as needed.

## Running examples

To run an example experiment, it should be sufficient to run `main.py` without any additional arguments. Typically, this will setup and run the experiment, and will include a method for importing a predefined configuration file (e.g., via Hydra) or will create a configuration dictionary for use within the example.