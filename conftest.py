from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config: object) -> bool:
    path = Path(collection_path)
    if path.suffix != ".py" or path.parent.name != "analysis":
        return False
    # sorrel/examples/<example>/analysis/*.py are notebooks/scripts, not tests.
    return path.parent.parent.parent.name == "examples"
