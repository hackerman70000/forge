from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class TaskSection(BaseModel):
    name: str
    type: Literal["classification", "generation"] = "classification"


class DataSection(BaseModel):
    format: Literal["parquet", "jsonl", "csv", "hf_dataset"] = "hf_dataset"
    source: str
    input_columns: list[str] = Field(default_factory=lambda: ["input"])
    target_column: str = "target"
    labels: dict[str, str] | None = None
    split: str | None = None
    split_column: str | None = None
    test_size: float = 0.1
    max_samples_per_label: int | None = None
    seed: int = 3407

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        if not 0 < v < 1:
            msg = f"test_size must be between 0 and 1 exclusive, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("labels", mode="before")
    @classmethod
    def normalize_label_keys(cls, v: dict | None) -> dict[str, str] | None:
        if v is None:
            return None
        return {str(k): str(val) for k, val in v.items()}


class PromptSection(BaseModel):
    system: str = ""
    template: str = "{input}"
    response_template: str | None = None


class LoRASection(BaseModel):
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainingSection(BaseModel):
    model: str = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
    output_dir: str | None = None
    load_in_4bit: bool = True
    chat_template: str | None = None
    gradient_checkpointing: str = "unsloth"
    lora: LoRASection = Field(default_factory=LoRASection)
    epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    optimizer: str = "adamw_torch_fused"
    lr_scheduler: str = "linear"
    neftune_noise_alpha: float | None = None
    label_smoothing: float = 0.0
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 50
    save_total_limit: int = 3
    early_stopping_patience: int = 3
    seed: int = 3407


class ExportSection(BaseModel):
    lora: bool = True
    hub_repo: str | None = None
    hub_private: bool = True


class TaskConfig(BaseModel):
    inherits: str | None = None
    task: TaskSection
    data: DataSection
    prompt: PromptSection = Field(default_factory=PromptSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    export: ExportSection = Field(default_factory=ExportSection)


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _resolve_raw(path: Path, visited: set[Path] | None = None) -> dict:
    if visited is None:
        visited = set()

    resolved = path.resolve()
    if resolved in visited:
        msg = f"Circular inheritance detected: {path}"
        raise ValueError(msg)
    visited.add(resolved)

    raw = _load_yaml(path)
    inherits = raw.pop("inherits", None)

    if inherits:
        base_path = path.parent / inherits
        if not base_path.exists():
            msg = f"Inherited config not found: {base_path}"
            raise FileNotFoundError(msg)
        base_raw = _resolve_raw(base_path, visited)
        raw = _deep_merge(base_raw, raw)

    return raw


def load_task_config(path: Path) -> TaskConfig:
    raw = _resolve_raw(path)
    return TaskConfig(**raw)
