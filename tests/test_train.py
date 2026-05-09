import csv
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from forge.config import TaskConfig


def _make_mock_torch(cuda_available: bool = False) -> MagicMock:
    mock = MagicMock()
    mock.cuda.is_available.return_value = cuda_available
    mock.cuda.memory_allocated.return_value = 4_000_000_000 if cuda_available else 0
    mock.cuda.is_bf16_supported.return_value = False
    mock.cuda.empty_cache = MagicMock()
    return mock


@pytest.fixture
def basic_config():
    return TaskConfig(
        task={"name": "my-task", "type": "classification"},
        data={
            "source": "test",
            "format": "jsonl",
            "input_columns": ["text"],
            "target_column": "label",
        },
        prompt={"template": "{text}"},
        training={
            "model": "some-org/some-model",
            "output_dir": None,
        },
    )


@pytest.fixture(autouse=True)
def inject_torch_stub():
    if "torch" not in sys.modules:
        mock_torch = _make_mock_torch()
        sys.modules["torch"] = mock_torch
        import forge.train

        forge.train.torch = mock_torch
        try:
            yield mock_torch
        finally:
            sys.modules.pop("torch", None)
            if hasattr(forge.train, "torch") and forge.train.torch is mock_torch:
                del forge.train.torch
    else:
        yield sys.modules["torch"]


class TestMetricsCallback:
    def test_csv_file_created_on_train_begin(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_control = MagicMock()

        callback.on_train_begin(mock_args, mock_state, mock_control)

        assert (tmp_path / "training_metrics.csv").exists()
        callback.on_train_end(mock_args, mock_state, mock_control)

    def test_csv_header_written(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_control = MagicMock()

        callback.on_train_begin(mock_args, mock_state, mock_control)
        callback.on_train_end(mock_args, mock_state, mock_control)

        with (tmp_path / "training_metrics.csv").open() as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == ["step", "epoch", "loss", "learning_rate", "grad_norm", "gpu_memory_gb", "tokens_per_second", "timestamp"]

    def test_on_log_writes_row_when_loss_present(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        args = MagicMock()
        state = MagicMock()
        state.global_step = 10
        state.epoch = 1.0
        control = MagicMock()

        callback.on_train_begin(args, state, control)

        logs = {"loss": 0.5, "learning_rate": 2e-4, "grad_norm": 0.123}
        callback.on_log(args, state, control, logs=logs)

        callback.on_train_end(args, state, control)

        with (tmp_path / "training_metrics.csv").open() as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)

        assert row[0] == "10"
        assert float(row[2]) == pytest.approx(0.5)

    def test_on_log_skips_row_when_no_loss(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        args = MagicMock()
        state = MagicMock()
        state.global_step = 5
        state.epoch = 0.5
        control = MagicMock()

        callback.on_train_begin(args, state, control)

        logs = {"learning_rate": 2e-4}
        callback.on_log(args, state, control, logs=logs)
        callback.on_train_end(args, state, control)

        with (tmp_path / "training_metrics.csv").open() as f:
            rows = list(csv.reader(f))

        assert len(rows) == 1

    def test_on_log_empty_logs_skips(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()

        callback.on_train_begin(args, state, control)
        callback.on_log(args, state, control, logs={})
        callback.on_train_end(args, state, control)

        with (tmp_path / "training_metrics.csv").open() as f:
            rows = list(csv.reader(f))

        assert len(rows) == 1

    def test_on_log_none_logs_skips(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()

        callback.on_train_begin(args, state, control)
        callback.on_log(args, state, control, logs=None)
        callback.on_train_end(args, state, control)

    def test_on_train_end_closes_file(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()

        callback.on_train_begin(args, state, control)
        assert callback.csv_file is not None
        callback.on_train_end(args, state, control)
        assert callback.csv_file.closed

    def test_on_train_end_without_begin_is_safe(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()

        callback.on_train_end(args, state, control)

    def test_close_is_idempotent(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()

        callback.on_train_begin(args, state, control)
        callback.close()
        callback.close()
        assert callback.csv_file.closed

    def test_close_without_begin_is_safe(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        callback.close()

    def test_eval_loss_is_used_when_loss_absent(self, tmp_path):
        from forge.train import MetricsCallback

        callback = MetricsCallback(tmp_path)
        args = MagicMock()
        state = MagicMock()
        state.global_step = 50
        state.epoch = 1.0
        control = MagicMock()

        callback.on_train_begin(args, state, control)
        logs = {"eval_loss": 0.25}
        callback.on_log(args, state, control, logs=logs)
        callback.on_train_end(args, state, control)

        with (tmp_path / "training_metrics.csv").open() as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)

        assert float(row[2]) == pytest.approx(0.25)


class TestLogMemory:
    def test_log_memory_no_gpu(self):
        import forge.train

        mock_torch = _make_mock_torch(cuda_available=False)
        forge.train.torch = mock_torch

        mock_psutil = MagicMock()
        mock_psutil.Process.return_value.memory_info.return_value.rss = 2_000_000_000

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            from forge.train import _log_memory

            _log_memory("test_stage")

        mock_psutil.Process.assert_called_once()
        mock_torch.cuda.memory_allocated.assert_not_called()

    def test_log_memory_with_gpu(self):
        import forge.train

        mock_torch = _make_mock_torch(cuda_available=True)
        forge.train.torch = mock_torch

        mock_psutil = MagicMock()
        mock_psutil.Process.return_value.memory_info.return_value.rss = 8_000_000_000

        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            from forge.train import _log_memory

            _log_memory("with_gpu")

        mock_torch.cuda.memory_allocated.assert_called_once()


class TestCleanupMemory:
    def test_cleanup_no_gpu(self):
        import forge.train

        mock_torch = _make_mock_torch(cuda_available=False)
        forge.train.torch = mock_torch

        with patch("forge.train.gc") as mock_gc:
            from forge.train import _cleanup_memory

            _cleanup_memory()
            mock_gc.collect.assert_called_once()
            mock_torch.cuda.empty_cache.assert_not_called()

    def test_cleanup_with_gpu(self):
        import forge.train

        mock_torch = _make_mock_torch(cuda_available=True)
        forge.train.torch = mock_torch

        with patch("forge.train.gc") as mock_gc:
            from forge.train import _cleanup_memory

            _cleanup_memory()
            mock_gc.collect.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()


class TestTrainerResolveOutputDir:
    def test_with_explicit_output_dir(self, tmp_path, basic_config):
        basic_config.training.output_dir = str(tmp_path / "output")

        from forge.train import Trainer

        trainer = Trainer(basic_config)
        output = trainer._resolve_output_dir()
        assert output == Path(basic_config.training.output_dir)
        assert output.exists()

    def test_auto_output_dir_strips_bnb_suffix(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = TaskConfig(
            task={"name": "test-task"},
            data={"source": "test", "format": "jsonl"},
            training={"model": "unsloth/Llama-3.3-70B-Instruct-bnb-4bit", "output_dir": None},
        )
        from forge.train import Trainer

        trainer = Trainer(config)
        output = trainer._resolve_output_dir()
        path_str = str(output)
        assert "test-task" in path_str
        assert "Llama-3.3-70B-Instruct" in path_str
        assert "bnb-4bit" not in path_str
        assert output.exists()

    def test_auto_output_dir_created(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = TaskConfig(
            task={"name": "my-task"},
            data={"source": "test", "format": "jsonl"},
            training={"model": "org/SomeModel", "output_dir": None},
        )
        from forge.train import Trainer

        trainer = Trainer(config)
        output = trainer._resolve_output_dir()
        assert output.exists()
        assert output.is_dir()

    def test_output_dir_nested_creation(self, tmp_path, basic_config):
        deep_path = tmp_path / "a" / "b" / "c"
        basic_config.training.output_dir = str(deep_path)

        from forge.train import Trainer

        trainer = Trainer(basic_config)
        output = trainer._resolve_output_dir()
        assert output.exists()


class TestTemplateMarkers:
    def test_chatml_markers_present(self):
        from forge.train import Trainer

        assert "chatml" in Trainer.TEMPLATE_MARKERS
        chatml = Trainer.TEMPLATE_MARKERS["chatml"]
        assert "instruction_part" in chatml
        assert "response_part" in chatml
        assert chatml["instruction_part"] == "<|im_start|>user\n"
        assert chatml["response_part"] == "<|im_start|>assistant\n"

    def test_llama_31_markers_present(self):
        from forge.train import Trainer

        assert "llama-3.1" in Trainer.TEMPLATE_MARKERS
        llama = Trainer.TEMPLATE_MARKERS["llama-3.1"]
        assert llama["instruction_part"] == "<|start_header_id|>user<|end_header_id|>\n\n"
        assert llama["response_part"] == "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def test_mistral_markers_present(self):
        from forge.train import Trainer

        assert "mistral" in Trainer.TEMPLATE_MARKERS
        mistral = Trainer.TEMPLATE_MARKERS["mistral"]
        assert mistral["instruction_part"] == "[INST]"
        assert mistral["response_part"] == "[/INST]"

    def test_all_markers_have_required_keys(self):
        from forge.train import Trainer

        for template_name, markers in Trainer.TEMPLATE_MARKERS.items():
            assert "instruction_part" in markers, f"Missing instruction_part for {template_name}"
            assert "response_part" in markers, f"Missing response_part for {template_name}"


class TestTrainerSetupMocked:
    def test_setup_calls_fast_language_model(self, basic_config):
        mock_unsloth = types.ModuleType("unsloth")
        mock_flm_class = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_flm_class.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm_class.get_peft_model.return_value = mock_model

        param1 = MagicMock()
        param1.numel.return_value = 100
        param1.requires_grad = True
        param2 = MagicMock()
        param2.numel.return_value = 900
        param2.requires_grad = False
        mock_model.parameters.return_value = [param1, param2]
        mock_unsloth.FastLanguageModel = mock_flm_class

        mock_chat_templates = types.ModuleType("unsloth.chat_templates")
        mock_chat_templates.get_chat_template = MagicMock(return_value=mock_tokenizer)
        mock_chat_templates.train_on_responses_only = MagicMock()

        with (
            patch.dict(sys.modules, {"unsloth": mock_unsloth, "unsloth.chat_templates": mock_chat_templates}),
            patch("forge.train._cleanup_memory"),
            patch("forge.train._log_memory"),
        ):
            from forge.train import Trainer

            trainer = Trainer(basic_config)
            trainer.setup()

            mock_flm_class.from_pretrained.assert_called_once()
            mock_flm_class.get_peft_model.assert_called_once()
            assert trainer.model is mock_model
            assert trainer.tokenizer is mock_tokenizer


class TestTrainerFormatDataset:
    def test_format_dataset_returns_text_dataset(self, basic_config, monkeypatch):
        mock_unsloth = types.ModuleType("unsloth")
        monkeypatch.setitem(sys.modules, "unsloth", mock_unsloth)

        from forge.train import Trainer

        trainer = Trainer(basic_config)
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.apply_chat_template.side_effect = [
            "formatted-a",
            "formatted-b",
        ]

        dataset = Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"}],
                    [{"role": "user", "content": "b"}, {"role": "assistant", "content": "y"}],
                ]
            }
        )

        with patch("forge.train._log_memory"), patch("forge.train._cleanup_memory"):
            result = trainer._format_dataset(dataset, "train")

        assert result.column_names == ["text"]
        assert result["text"] == ["formatted-a", "formatted-b"]
        assert trainer.tokenizer.apply_chat_template.call_count == 2

    def test_format_dataset_disables_thinking_when_configured(self, basic_config, monkeypatch):
        mock_unsloth = types.ModuleType("unsloth")
        monkeypatch.setitem(sys.modules, "unsloth", mock_unsloth)

        from forge.train import Trainer

        basic_config.prompt.enable_thinking = False
        trainer = Trainer(basic_config)
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.apply_chat_template.return_value = "formatted"

        dataset = Dataset.from_dict(
            {
                "messages": [
                    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"}],
                ]
            }
        )

        with patch("forge.train._log_memory"), patch("forge.train._cleanup_memory"):
            trainer._format_dataset(dataset, "train")

        trainer.tokenizer.apply_chat_template.assert_called_once_with(
            dataset[0]["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    def test_format_dataset_handles_empty_dataset(self, basic_config, monkeypatch):
        mock_unsloth = types.ModuleType("unsloth")
        monkeypatch.setitem(sys.modules, "unsloth", mock_unsloth)

        from forge.train import Trainer

        trainer = Trainer(basic_config)
        trainer.tokenizer = MagicMock()

        dataset = Dataset.from_dict({"messages": []})

        with patch("forge.train._log_memory"), patch("forge.train._cleanup_memory"):
            result = trainer._format_dataset(dataset, "train")

        assert result.column_names == ["text"]
        assert result["text"] == []
        trainer.tokenizer.apply_chat_template.assert_not_called()

    def test_setup_no_division_by_zero_when_zero_params(self, basic_config):
        mock_unsloth = types.ModuleType("unsloth")
        mock_flm_class = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_flm_class.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_flm_class.get_peft_model.return_value = mock_model
        mock_model.parameters.return_value = []
        mock_unsloth.FastLanguageModel = mock_flm_class

        mock_chat_templates = types.ModuleType("unsloth.chat_templates")
        mock_chat_templates.get_chat_template = MagicMock(return_value=mock_tokenizer)

        with (
            patch.dict(sys.modules, {"unsloth": mock_unsloth, "unsloth.chat_templates": mock_chat_templates}),
            patch("forge.train._cleanup_memory"),
            patch("forge.train._log_memory"),
        ):
            from forge.train import Trainer

            trainer = Trainer(basic_config)
            trainer.setup()
