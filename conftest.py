from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config: object) -> bool:
    path = Path(collection_path)
    return (
        path.suffix == ".py"
        and path.name.startswith("analyze_")
        and path.parent.name == "analysis"
        and path.parent.parent.name == "staghunt_physical"
    )
