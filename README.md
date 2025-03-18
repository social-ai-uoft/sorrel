<img style="width: 100px" src="https://github.com/social-ai-uoft/sorrel/blob/main/media/sorrel-icon.png" />

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/social-ai-uoft/sorrel/main.svg)](https://results.pre-commit.ci/latest/github/social-ai-uoft/sorrel/main) ![pytest status](https://github.com/social-ai-uoft/sorrel/workflows/PyTest/badge.svg)

# Sorrel

Sorrel is a general-purpose reinforcement learning engine that enables researchers, developers, and students to 
develop and deploy reinforcement learning algorithms on new and pre-existing environments.

Sorrel can be regarded as an “operating system” for the RL process, in the sense that it unifies and abstracts 
foundational components from environment simulation all the way to policy search, exploration, and function 
approximation.

Our hope is that Sorrel will foster new research ideas, applications, and tools for a unified RL approach. We 
believe Sorrel will accelerate the rate of progress of the RL research space, as well as allow us to experiment 
with our own solutions to both novel and long-standing RL problems.

> [!NOTE]
> Sorrel is extremely experimental and subject to change!

## Development

> [!IMPORTANT]
> We recommend you follow these instructions in a fresh conda/virtual environment to keep packages isolated
> from other environments and/or Python versions. Python 3.11+ is required.

To create a [virtual environment](https://docs.python.org/3/library/venv.html),
navigate to your project directory in the terminal and run:
```
python -m venv ./venv
```
To activate the virtual environment, run 

```
./venv/Scripts/activate  # windows

source ./venv/bin/activate  # mac
```

to deactivate the virtual environment, simply run ```deactivate``` in the terminal.

### Getting started

Sorrel uses the [poetry](https://python-poetry.org/) package manager to manage its dependencies. Start by 
running ```poetry --version``` in your terminal to make sure you have poetry installed.

> [!IMPORTANT]
> If you do not have poetry, use ```pipx install poetry``` (not pip) to make sure you do not have poetry installed in the 
> same environment that Sorrel is using. 
> See the [poetry](https://python-poetry.org/) documentation for more information and 
> installation instructions.

With poetry available, to install Sorrel as an user, run the following command:
```
poetry install
```
in the folder containing the ``pyproject.toml`` file.

If you wish to install additional dependencies, such as tensorboard for logging needs, 
you can include the extra dependencies by running the following instead:
```
poetry install --with extras
```
in the folder containing the ``pyproject.toml`` file.

To install Sorrel in development mode, include the optional dependency groups like so:
```
poetry install --with dev,extras
```
in the folder containing the ``pyproject.toml`` file.

> [!NOTE]
> If cannot run your script from terminal due to errors like ```module 'sorrel' not found```, check if your virtual environment is activated. 
> If it is, and you are still getting the same error, try running ```poetry run python your_script.py```.

### Workflow Tools
We use a number of tools to simplify the development workflow for contributors. Among these tools include code 
formatters (such as [black](https://github.com/python/black)) to report style errors and (try to) 
automatically format your code wherever possible, along with testing frameworks (such as 
[pytest](https://pypi.python.org/pypi/pytest)) to automatically test the code.

We have included a [pre-commit](https://pre-commit.com/) configuration to automatically run all CI tasks whenever you 
attempt to commit to the Sorrel repository. 

> [!WARNING]
> We highly recommend you use pre-commit as pull requests will NOT be
> merged unless you pass ALL CI checks (including the pre-commit CI check).

To set up pre-commit, start by confirming that it's installed by running:
```
pre-commit --version
```
If this fails, you'll need to install pre-commit by running ``pip install pre-commit``. Then, run
```
pre-commit install
```
in the folder containing ``.pre-commit-config.yaml`` file.

Afterward, the Git hooks will be run automatically at every new commit.

You may also run these hooks manually with ```pre-commit run --all-files```. If needed, you can skip the hooks (not 
recommended) with ```git commit --no-verify -m <commit message>```.

> [!NOTE]
> You may have to run ```pre-commit run --all-files``` manually a couple of times to make it pass when you commit,
> as each formatting tool will first format the code and fail the first time but should pass the second time.

### Writing documentation

To contribute to the documentation, you may add, delete, or edit files in the ``.\docs\source`` folder. This project 
uses [Sphinx](https://www.sphinx-doc.org/) to auto-build files in ``.rst``, ``.md``, or ``.ipynb`` format into 
html.

Note that since Sorrel is private (for now), we cannot host the documentation online yet. To view your documentation
changes locally, navigate to the ``.\docs`` folder and run the following command:

Windows:
```
.\make html
```

Other systems:
```
make html
```

Then, view the updated documentations by opening ``.\docs\build\html\index.html`` with the browser of your choice.

> [!NOTE]
> If the changes you made were not reflected across all the pages, you may have to run ```.\make clean```
> (on Windows) or ```make clean``` (on other systems) before the above command to ensure the documentation is built from
> scratch. It is generally good practice to do so anyway but may take slightly longer.

## Citing the project

TODO: Insert paper citation here

## Maintainers
Sorrel is currently maintained by Yibing Ju (aka [@bing-j](https://github.com/galacticglum)) and [Rebekah Gelpi](https://rgelpi.github.io) (aka [@rgelpi](https://github.com/galacticglum)).

> [!NOTE]
> **We do not do technical support, nor consulting** and don't answer personal questions per email. If you have any questions, concerns, or suggestions, please post them on the [GitHub issues page](https://github.com/social-ai-uoft/sorrel/issues) or the [GitHub discussion page](https://github.com/social-ai-uoft/sorrel/discussions).

## Contributing

Sorrel is open source and is developed by a community of researchers, developers, and students. We welcome contributions from all levels of the community. To get started, please read the [contributing guide](CONTRIBUTING.md).

## Acknowledgments

TODO: Insert acknowledgements here
