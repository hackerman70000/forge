from forge.config import TaskConfig, load_task_config
from forge.data import load_dataset_from_config

__all__ = ["TaskConfig", "Trainer", "load_dataset_from_config", "load_task_config"]


def __getattr__(name: str):
    if name == "Trainer":
        from forge.train import Trainer

        return Trainer
    msg = f"module 'forge' has no attribute {name!r}"
    raise AttributeError(msg)
