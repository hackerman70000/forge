import csv
import json
from unittest.mock import patch

import pytest
from datasets import Dataset

from forge.config import TaskConfig
from forge.data import (
    _balance_by_label,
    _build_messages,
    _filter_by_split,
    _load_single_dataset,
    convert_to_chat_format,
    load_dataset_from_config,
)


@pytest.fixture
def basic_config():
    return TaskConfig(
        task={"name": "test", "type": "classification"},
        data={
            "source": "test_source",
            "format": "jsonl",
            "input_columns": ["text"],
            "target_column": "label",
        },
        prompt={"template": "{text}"},
    )


@pytest.fixture
def labeled_config():
    return TaskConfig(
        task={"name": "test", "type": "classification"},
        data={
            "source": "test_source",
            "format": "jsonl",
            "input_columns": ["text"],
            "target_column": "label",
            "labels": {"0": "negative", "1": "positive"},
        },
        prompt={"system": "Classify.", "template": "{text}"},
    )


class TestLoadRawDataset:
    def test_load_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        records = [{"text": f"sample {i}", "label": i % 2} for i in range(5)]
        with jsonl_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ds = _load_single_dataset(str(jsonl_file), "jsonl", None)
        assert len(ds) == 5
        assert "text" in ds.column_names
        assert "label" in ds.column_names

    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "data.csv"
        with csv_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            for i in range(4):
                writer.writerow([f"text {i}", i % 2])

        ds = _load_single_dataset(str(csv_file), "csv", None)
        assert len(ds) == 4
        assert "text" in ds.column_names

    def test_load_parquet(self, tmp_path):
        pytest.importorskip("pyarrow")
        parquet_file = tmp_path / "data.parquet"
        ds = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 0]})
        ds.to_parquet(str(parquet_file))

        loaded = _load_single_dataset(str(parquet_file), "parquet", None)
        assert len(loaded) == 3

    def test_load_hf_dataset_mocked(self):
        mock_ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})

        with patch("forge.data.load_dataset", return_value=mock_ds) as mock_load:
            result = _load_single_dataset("some/dataset", "hf_dataset", None)
            mock_load.assert_called_once_with("some/dataset", split="train", keep_in_memory=False)
            assert len(result) == 2

    def test_load_hf_dataset_with_split(self):
        mock_ds = Dataset.from_dict({"text": ["a"], "label": [0]})

        with patch("forge.data.load_dataset", return_value=mock_ds) as mock_load:
            _load_single_dataset("some/dataset", "hf_dataset", "validation")
            mock_load.assert_called_once_with("some/dataset", split="validation", keep_in_memory=False)


class TestFilterBySplit:
    def test_no_split_column_returns_unchanged(self, basic_config):
        ds = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 0]})
        basic_config.data.split_column = None
        result = _filter_by_split(ds, basic_config)
        assert len(result) == 3

    def test_no_split_value_returns_unchanged(self, basic_config):
        ds = Dataset.from_dict({"text": ["a", "b"], "split": ["train", "test"], "label": [0, 1]})
        basic_config.data.split_column = "split"
        basic_config.data.split = None
        result = _filter_by_split(ds, basic_config)
        assert len(result) == 2

    def test_filters_by_split_column(self, basic_config):
        ds = Dataset.from_dict(
            {
                "text": ["a", "b", "c", "d"],
                "split": ["train", "test", "train", "test"],
                "label": [0, 1, 0, 1],
            }
        )
        basic_config.data.split_column = "split"
        basic_config.data.split = "train"
        result = _filter_by_split(ds, basic_config)
        assert len(result) == 2
        assert all(x == "train" for x in result["split"])

    def test_missing_split_column_returns_unchanged(self, basic_config):
        ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
        basic_config.data.split_column = "nonexistent_col"
        basic_config.data.split = "train"
        result = _filter_by_split(ds, basic_config)
        assert len(result) == 2


class TestBalanceByLabel:
    def test_all_samples_in_one_label(self):
        config = TaskConfig(
            task={"name": "test"},
            data={
                "source": "test",
                "format": "jsonl",
                "target_column": "label",
                "max_samples_per_label": 3,
            },
        )
        ds = Dataset.from_dict({"text": ["a", "b", "c", "d", "e"], "label": [1, 1, 1, 1, 1]})
        result = _balance_by_label(ds, config)
        assert len(result) == 3

    def test_max_samples_larger_than_dataset(self):
        config = TaskConfig(
            task={"name": "test"},
            data={
                "source": "test",
                "format": "jsonl",
                "target_column": "label",
                "max_samples_per_label": 1000,
            },
        )
        ds = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 0]})
        result = _balance_by_label(ds, config)
        assert len(result) == 3

    def test_empty_dataset(self):
        config = TaskConfig(
            task={"name": "test"},
            data={
                "source": "test",
                "format": "jsonl",
                "target_column": "label",
                "max_samples_per_label": 5,
            },
        )
        ds = Dataset.from_dict({"text": [], "label": []})
        result = _balance_by_label(ds, config)
        assert len(result) == 0

    def test_preserves_order_of_selected_indices(self):
        config = TaskConfig(
            task={"name": "test"},
            data={
                "source": "test",
                "format": "jsonl",
                "target_column": "label",
                "max_samples_per_label": 2,
            },
        )
        ds = Dataset.from_dict(
            {
                "text": [f"t{i}" for i in range(10)],
                "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            }
        )
        result = _balance_by_label(ds, config)
        assert len(result) == 4


class TestBuildMessages:
    def test_template_with_missing_variable_renders_empty(self, labeled_config):
        labeled_config.prompt.template = "{text} and {missing_col}"
        sample = {"text": "hello", "label": 1}
        messages = _build_messages(sample, labeled_config)
        assert "hello and " in messages[1]["content"]
        assert "{missing_col}" not in messages[1]["content"]

    def test_template_with_special_braces(self):
        config = TaskConfig(
            task={"name": "test"},
            data={"source": "test", "format": "jsonl", "target_column": "label"},
            prompt={"template": "Code: ```\n{code}\n```"},
        )
        sample = {"code": "int x = 0;", "label": "safe"}
        messages = _build_messages(sample, config)
        assert "int x = 0;" in messages[0]["content"]

    def test_template_with_newlines_in_content(self, basic_config):
        sample = {"text": "line1\nline2\nline3", "label": "0"}
        messages = _build_messages(sample, basic_config)
        assert "line1\nline2\nline3" in messages[0]["content"]

    def test_integer_target_with_labels_mapping(self, labeled_config):
        sample = {"text": "good", "label": 1}
        messages = _build_messages(sample, labeled_config)
        assert messages[-1]["content"] == "positive"

    def test_string_target_with_labels_mapping(self, labeled_config):
        sample = {"text": "bad", "label": "0"}
        messages = _build_messages(sample, labeled_config)
        assert messages[-1]["content"] == "negative"

    def test_float_target_without_labels_mapping(self, basic_config):
        sample = {"text": "x", "label": 0.75}
        messages = _build_messages(sample, basic_config)
        assert messages[-1]["content"] == "0.75"

    def test_label_not_in_mapping_falls_back_to_str(self, labeled_config):
        sample = {"text": "x", "label": 99}
        messages = _build_messages(sample, labeled_config)
        assert messages[-1]["content"] == "99"

    def test_extra_columns_available_in_template(self):
        config = TaskConfig(
            task={"name": "test"},
            data={
                "source": "test",
                "format": "jsonl",
                "input_columns": ["text"],
                "target_column": "label",
            },
            prompt={"template": "{text} - extra: {extra_col}"},
        )
        sample = {"text": "hello", "label": 0, "extra_col": "bonus"}
        messages = _build_messages(sample, config)
        assert "bonus" in messages[0]["content"]

    def test_messages_structure_without_system(self, basic_config):
        basic_config.prompt.system = ""
        sample = {"text": "test", "label": "answer"}
        messages = _build_messages(sample, basic_config)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_messages_structure_with_system(self, labeled_config):
        sample = {"text": "test", "label": 0}
        messages = _build_messages(sample, labeled_config)
        assert len(messages) == 3
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant"]

    def test_multiline_code_in_template(self):
        config = TaskConfig(
            task={"name": "test"},
            data={
                "source": "test",
                "format": "jsonl",
                "input_columns": ["code", "language"],
                "target_column": "vuln",
            },
            prompt={
                "template": "Analyze this {language}:\n```\n{code}\n```",
            },
        )
        code = "void foo() {\n    int x = *(int*)0;\n    return;\n}"
        sample = {"code": code, "language": "C", "vuln": "1"}
        messages = _build_messages(sample, config)
        assert code in messages[0]["content"]
        assert "C" in messages[0]["content"]


class TestConvertToChatFormat:
    def test_empty_dataset(self, basic_config):
        ds = Dataset.from_dict({"text": [], "label": []})
        result = convert_to_chat_format(ds, basic_config)
        assert "messages" in result.column_names
        assert len(result) == 0

    def test_result_has_only_messages_column(self, basic_config):
        ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
        result = convert_to_chat_format(ds, basic_config)
        assert result.column_names == ["messages"]

    def test_each_row_is_list_of_dicts(self, basic_config):
        ds = Dataset.from_dict({"text": ["a"], "label": [0]})
        result = convert_to_chat_format(ds, basic_config)
        messages = result[0]["messages"]
        assert isinstance(messages, list)
        for msg in messages:
            assert "role" in msg
            assert "content" in msg

    def test_dataset_with_missing_target_column_raises(self, basic_config):
        ds = Dataset.from_dict({"text": ["a"]})
        with pytest.raises(KeyError):
            convert_to_chat_format(ds, basic_config)


class TestLoadDatasetFromConfig:
    def test_full_pipeline_with_local_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "data.jsonl"
        records = [{"text": f"sample {i}", "label": i % 2} for i in range(20)]
        with jsonl_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        config = TaskConfig(
            task={"name": "test"},
            data={
                "source": str(jsonl_file),
                "format": "jsonl",
                "input_columns": ["text"],
                "target_column": "label",
                "test_size": 0.2,
            },
            prompt={"template": "{text}"},
        )
        train_ds, eval_ds = load_dataset_from_config(config)
        assert len(train_ds) + len(eval_ds) == 20
        assert "messages" in train_ds.column_names
        assert "messages" in eval_ds.column_names


class TestVulnLLMChatMessages:
    def test_vulnllm_chat_message_format(self):
        config = TaskConfig(
            task={"name": "vulnllm-detection", "type": "classification"},
            data={
                "source": "UCSB-SURFI/VulnLLM-R-Train-Data",
                "format": "hf_dataset",
                "input_columns": ["code", "language"],
                "target_column": "target",
                "labels": {"0": "not_vulnerable", "1": "vulnerable"},
            },
            prompt={
                "system": "You are a security code reviewer. Analyze source code.",
                "template": "Analyze this {language} function:\n\n```\n{code}\n```",
            },
        )

        sample = {
            "code": "int main() { return 0; }",
            "language": "C",
            "target": 0,
        }
        messages = _build_messages(sample, config)

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert "C" in messages[1]["content"]
        assert "int main()" in messages[1]["content"]
        assert messages[2]["content"] == "not_vulnerable"

    def test_vulnllm_vulnerable_label(self):
        config = TaskConfig(
            task={"name": "vulnllm-detection", "type": "classification"},
            data={
                "source": "UCSB-SURFI/VulnLLM-R-Train-Data",
                "format": "hf_dataset",
                "input_columns": ["code", "language"],
                "target_column": "target",
                "labels": {"0": "not_vulnerable", "1": "vulnerable"},
            },
            prompt={
                "template": "Analyze {language}:\n{code}",
            },
        )

        sample = {"code": "strcpy(buf, input);", "language": "C", "target": 1}
        messages = _build_messages(sample, config)
        assert messages[-1]["content"] == "vulnerable"
