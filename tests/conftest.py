from pathlib import Path

import pytest


@pytest.fixture
def tasks_dir() -> Path:
    return Path(__file__).parent.parent / "tasks"


@pytest.fixture
def base_config_path(tasks_dir) -> Path:
    return tasks_dir / "base.yaml"


@pytest.fixture
def vulnllm_config_path(tasks_dir) -> Path:
    return tasks_dir / "vulnllm.yaml"
