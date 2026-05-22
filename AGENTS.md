# Repository Guidelines

## Project Structure & Module Organization
- `sorrel/`: core Python package (agents, environments, utilities, examples). This is the primary source tree.
- `sorrel/examples/`: runnable example environments and notebooks (e.g., `sorrel/examples/treasurehunt/`).
- `docs/`: Sphinx documentation sources and build tooling.
- `media/`: images and assets used by docs/README.
- `outputs/`: generated artifacts from runs (treat as ephemeral).

## Formatting & Style Requirements (Strict)

All contributions **must conform to the repository’s existing formatting and style rules**. Changes that deviate from these conventions must be revised before submission.

- Follow **Black-compatible formatting** (line length, spacing, and structure).
- Use **4-space indentation** everywhere (no tabs).
- Naming conventions:
  - `snake_case` for variables and functions
  - `PascalCase` for classes
  - `UPPER_SNAKE_CASE` for constants
  - lowercase, short module names
- Imports must be **isort-compatible** (Black profile), grouped and ordered consistently.
- Docstrings must follow standard Python conventions and be compatible with `docformatter`.
- Type annotations must be valid under `pyright` expectations.

**Do not run, modify, or reconfigure pre-commit hooks.**  
Ensure all edits are already compliant with:
`black`, `isort (black profile)`, `docformatter`, `pyupgrade`, and `pyright`.

If unsure about formatting, **mirror the style of nearby files** in `sorrel/agents/` or existing examples. Do not introduce new formatting patterns.

Formatting-only or cleanup-only changes are **not allowed** unless explicitly requested.

## Testing Guidelines
- To test changes in `sorrel/`, create a **new experiment folder** inside `sorrel/examples/` (you may mimic an existing simple environment such as `treasurehunt` or `cleanup`).
- Run the main file of the new experiment with the `sorrel` conda environment activated.
- **Do not modify existing example environments** for testing.

## Commit & Pull Request Guidelines
- Do not make commits directly.
- Implement the requested changes and perform required tests only.

## Documentation Notes
- Do not edit documentation files.

## Environment
- Conda binary (absolute path): `/opt/miniconda3/bin/conda`
- Conda environment name: `sorrel`
- Do NOT assume an activated shell.
- Do NOT use `conda activate`.

## Command execution rules
- Never run `python`, `pip`, `pytest`, `jupyter`, `ipython`, or any Python-related tool directly.
- Always execute Python-related commands using:

  `/opt/miniconda3/bin/conda run -n sorrel <command>`

- If a command would fail outside the Conda environment, it MUST be run using the above pattern.

## Examples
- `/opt/miniconda3/bin/conda run -n sorrel python script.py`
- `/opt/miniconda3/bin/conda run -n sorrel pytest`
- `/opt/miniconda3/bin/conda run -n sorrel pip install -r requirements.txt`
- `/opt/miniconda3/bin/conda run -n sorrel jupyter lab`

## Prohibited patterns
- Do NOT run `python` directly.
- Do NOT run `pip` directly.
- Do NOT rely on shell initialization files.
- Do NOT attempt to persist shell state across commands.

## Failure handling
- If a command fails due to environment issues, retry it using:
  `/opt/miniconda3/bin/conda run -n sorrel <command>`
- Do not attempt alternative execution methods that bypass the Conda environment.

## Assumptions
- The Conda environment `sorrel` already exists.
- Required dependencies are installed inside the `sorrel` environment unless explicitly instructed otherwise.