import pytest
import yaml

from forge.config import TaskConfig, _deep_merge, load_task_config


class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"training": {"epochs": 3, "lr": 0.001}}
        override = {"training": {"epochs": 5}}
        result = _deep_merge(base, override)
        assert result == {"training": {"epochs": 5, "lr": 0.001}}

    def test_deep_nested(self):
        base = {"training": {"lora": {"r": 32, "alpha": 64}}}
        override = {"training": {"lora": {"r": 16}}}
        result = _deep_merge(base, override)
        assert result["training"]["lora"]["r"] == 16
        assert result["training"]["lora"]["alpha"] == 64


class TestLoadTaskConfig:
    def test_load_base_requires_task_and_data(self, base_config_path):
        with pytest.raises(ValueError, match="validation error"):
            load_task_config(base_config_path)

    def test_load_with_inheritance(self, vulnllm_config_path):
        config = load_task_config(vulnllm_config_path)
        assert config.task.name == "vulnllm-detection"
        assert config.task.type == "classification"
        assert config.data.source == "UCSB-SURFI/VulnLLM-R-Train-Data"
        assert config.data.labels == {"0": "not_vulnerable", "1": "vulnerable"}
        assert config.training.lora.r == 32
        assert config.training.epochs == 3

    def test_inheritance_override(self, vulnllm_config_path):
        config = load_task_config(vulnllm_config_path)
        assert config.training.model == "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

    def test_missing_inherits_file(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "inherits": "nonexistent.yaml",
                    "task": {"name": "test"},
                    "data": {"source": "test", "format": "jsonl"},
                }
            )
        )
        with pytest.raises(FileNotFoundError):
            load_task_config(config_file)

    def test_minimal_config(self, tmp_path):
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "task": {"name": "test-task"},
                    "data": {"source": "some/dataset", "format": "jsonl"},
                }
            )
        )
        config = load_task_config(config_file)
        assert config.task.name == "test-task"
        assert config.prompt.system == ""
        assert config.training.epochs == 3


class TestTaskConfig:
    def test_defaults(self):
        config = TaskConfig(
            task={"name": "test", "type": "classification"},
            data={"source": "test/data", "format": "jsonl"},
        )
        assert config.training.learning_rate == 2e-4
        assert config.training.lora.r == 32
        assert config.export.lora is True

    def test_classification_labels(self):
        config = TaskConfig(
            task={"name": "test", "type": "classification"},
            data={
                "source": "test/data",
                "format": "parquet",
                "labels": {"0": "negative", "1": "positive"},
            },
        )
        assert config.data.labels["0"] == "negative"
        assert config.data.labels["1"] == "positive"
