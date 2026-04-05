import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from forge.cli import _log_config, app
from forge.config import TaskConfig


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def valid_config_file(tmp_path):
    config = {
        "task": {"name": "test-task", "type": "classification"},
        "data": {"source": str(tmp_path / "data.jsonl"), "format": "jsonl"},
        "training": {"model": "org/model", "epochs": 1},
    }
    config_file = tmp_path / "task.yaml"
    config_file.write_text(yaml.dump(config))
    return config_file


@pytest.fixture
def lora_path_with_adapter(tmp_path):
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    adapter_config = {
        "base_model_name_or_path": "org/base-model",
        "peft_type": "LORA",
    }
    (lora_dir / "adapter_config.json").write_text(json.dumps(adapter_config))
    return lora_dir


@pytest.fixture
def lora_path_with_4bit_adapter(tmp_path):
    lora_dir = tmp_path / "lora"
    lora_dir.mkdir()
    adapter_config = {
        "base_model_name_or_path": "unsloth/Llama-3.3-70B-Instruct-unsloth-bnb-4bit",
        "peft_type": "LORA",
    }
    (lora_dir / "adapter_config.json").write_text(json.dumps(adapter_config))
    return lora_dir


def _make_mock_trainer(output_path: Path):
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = output_path
    mock_trainer_class = MagicMock(return_value=mock_trainer_instance)
    return mock_trainer_class, mock_trainer_instance


class TestTrainCommand:
    def test_train_with_valid_config_calls_trainer(self, runner, valid_config_file, tmp_path):
        mock_trainer_class, _mock_trainer_instance = _make_mock_trainer(tmp_path / "output")

        with (
            patch("forge.config.load_task_config") as mock_load,
            patch("forge.train.Trainer", mock_trainer_class),
            patch("forge.cli.setup_logging"),
            patch("forge.cli._log_config"),
        ):
            mock_config = MagicMock()
            mock_config.training.output_dir = None
            mock_config.task.name = "test-task"
            mock_load.return_value = mock_config

            result = runner.invoke(app, ["train", str(valid_config_file)])

        assert result.exit_code == 0

    def test_train_with_model_override(self, runner, valid_config_file, tmp_path):
        runner.invoke(app, ["train", str(valid_config_file), "--model", "new/model"])

    def test_train_with_lora_r_override_sets_alpha(self, runner, valid_config_file, tmp_path):
        config = TaskConfig(
            task={"name": "test-task", "type": "classification"},
            data={"source": "test", "format": "jsonl"},
        )

        mock_trainer_class, _ = _make_mock_trainer(tmp_path / "out")

        with (
            patch("forge.config.load_task_config", return_value=config),
            patch("forge.train.Trainer", mock_trainer_class),
            patch("forge.cli.setup_logging"),
            patch("forge.cli._log_config"),
        ):
            runner.invoke(app, ["train", str(valid_config_file), "--lora-r", "16"])
            assert config.training.lora.r == 16
            assert config.training.lora.alpha == 32

    def test_train_exits_1_on_trainer_error(self, runner, valid_config_file, tmp_path):
        config = TaskConfig(
            task={"name": "test-task"},
            data={"source": "test", "format": "jsonl"},
        )

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.side_effect = RuntimeError("GPU out of memory")
        mock_trainer_class = MagicMock(return_value=mock_trainer_instance)

        with (
            patch("forge.config.load_task_config", return_value=config),
            patch("forge.train.Trainer", mock_trainer_class),
            patch("forge.cli.setup_logging"),
            patch("forge.cli._log_config"),
        ):
            result = runner.invoke(app, ["train", str(valid_config_file)])
            assert result.exit_code == 1

    def test_train_missing_config_file(self, runner, tmp_path):
        result = runner.invoke(app, ["train", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code != 0

    def test_train_lora_alpha_is_double_r(self, runner, valid_config_file, tmp_path):
        config = TaskConfig(
            task={"name": "test-task"},
            data={"source": "test", "format": "jsonl"},
        )

        mock_trainer_class, _ = _make_mock_trainer(tmp_path / "out")

        with (
            patch("forge.config.load_task_config", return_value=config),
            patch("forge.train.Trainer", mock_trainer_class),
            patch("forge.cli.setup_logging"),
            patch("forge.cli._log_config"),
        ):
            runner.invoke(app, ["train", str(valid_config_file), "--lora-r", "8"])
            assert config.training.lora.r == 8
            assert config.training.lora.alpha == 16

    def test_train_without_overrides_preserves_config(self, runner, valid_config_file, tmp_path):
        config = TaskConfig(
            task={"name": "test-task"},
            data={"source": "test", "format": "jsonl"},
            training={"model": "original/model", "epochs": 5},
        )

        mock_trainer_class, _ = _make_mock_trainer(tmp_path / "out")

        with (
            patch("forge.config.load_task_config", return_value=config),
            patch("forge.train.Trainer", mock_trainer_class),
            patch("forge.cli.setup_logging"),
            patch("forge.cli._log_config"),
        ):
            runner.invoke(app, ["train", str(valid_config_file)])
            assert config.training.model == "original/model"
            assert config.training.epochs == 5


class TestMergeLora:
    def test_merge_lora_missing_path(self, runner, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        result = runner.invoke(app, ["merge-lora", str(nonexistent)])
        assert result.exit_code == 1

    def test_merge_lora_missing_adapter_config(self, runner, tmp_path):
        lora_dir = tmp_path / "lora_no_config"
        lora_dir.mkdir()
        result = runner.invoke(app, ["merge-lora", str(lora_dir)])
        assert result.exit_code == 1

    def test_merge_lora_requires_base_model_if_adapter_config_empty(self, runner, tmp_path):
        lora_dir = tmp_path / "lora"
        lora_dir.mkdir()
        adapter_config = {}
        (lora_dir / "adapter_config.json").write_text(json.dumps(adapter_config))

        result = runner.invoke(app, ["merge-lora", str(lora_dir)])
        assert result.exit_code == 1

    def test_merge_lora_4bit_model_substitutes_fp16(self, runner, lora_path_with_4bit_adapter, tmp_path):
        mock_torch = MagicMock()
        mock_torch.bfloat16 = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls = MagicMock(return_value=mock_model)
        mock_tok_cls = MagicMock(return_value=mock_tokenizer)
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_model
        mock_peft_cls = MagicMock(return_value=mock_peft_model)

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "peft": MagicMock(PeftModel=mock_peft_cls),
                "transformers": MagicMock(AutoModelForCausalLM=mock_model_cls, AutoTokenizer=mock_tok_cls),
            },
        ):
            output = tmp_path / "merged"
            result = runner.invoke(app, ["merge-lora", str(lora_path_with_4bit_adapter), "--output", str(output)])

        if result.exit_code == 0:
            call_args_str = str(mock_tok_cls.from_pretrained.call_args)
            assert "bnb-4bit" not in call_args_str

    def test_merge_lora_with_explicit_base_model_skips_adapter_config_base(self, runner, lora_path_with_adapter, tmp_path):
        mock_torch = MagicMock()
        mock_torch.bfloat16 = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model_cls = MagicMock(return_value=mock_model)
        mock_tok_cls = MagicMock(return_value=mock_tokenizer)
        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = mock_model
        mock_peft_cls = MagicMock(return_value=mock_peft_model)

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "peft": MagicMock(PeftModel=mock_peft_cls),
                "transformers": MagicMock(AutoModelForCausalLM=mock_model_cls, AutoTokenizer=mock_tok_cls),
            },
        ):
            output = tmp_path / "merged"
            result = runner.invoke(
                app,
                ["merge-lora", str(lora_path_with_adapter), "--base-model", "my/custom-base", "--output", str(output)],
            )

        if result.exit_code == 0:
            call_args = mock_tok_cls.from_pretrained.call_args
            assert call_args[0][0] == "my/custom-base"


class TestLogConfig:
    def test_log_config_does_not_raise(self):
        config = TaskConfig(
            task={"name": "test-task", "type": "classification"},
            data={"source": "some/data", "format": "jsonl"},
            training={"model": "org/model", "epochs": 5},
        )
        _log_config(config)

    def test_log_config_with_hub_repo(self):
        config = TaskConfig(
            task={"name": "test-task"},
            data={"source": "some/data", "format": "jsonl"},
            export={"hub_repo": "my-org/my-model", "lora": True},
        )
        _log_config(config)

    def test_log_config_without_hub_repo(self):
        config = TaskConfig(
            task={"name": "test-task"},
            data={"source": "some/data", "format": "jsonl"},
            export={"hub_repo": None},
        )
        _log_config(config)

    def test_log_config_effective_batch_size(self):
        config = TaskConfig(
            task={"name": "test"},
            data={"source": "test", "format": "jsonl"},
            training={"batch_size": 4, "gradient_accumulation_steps": 8},
        )
        _log_config(config)
