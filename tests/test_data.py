import pytest
from datasets import Dataset

from forge.config import TaskConfig
from forge.data import _balance_by_label, _build_messages, convert_to_chat_format


@pytest.fixture
def classification_config():
    return TaskConfig(
        task={"name": "test", "type": "classification"},
        data={
            "source": "test",
            "format": "jsonl",
            "input_columns": ["text"],
            "target_column": "label",
            "labels": {"0": "negative", "1": "positive"},
        },
        prompt={
            "system": "You are a sentiment classifier.",
            "template": "Classify this text: {text}",
        },
    )


@pytest.fixture
def sample_dataset():
    return Dataset.from_dict(
        {
            "text": ["good movie", "bad movie", "great film", "terrible"],
            "label": [1, 0, 1, 0],
        }
    )


class TestBuildMessages:
    def test_with_system_and_labels(self, classification_config):
        sample = {"text": "good movie", "label": 1}
        messages = _build_messages(sample, classification_config)

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a sentiment classifier."
        assert messages[1]["role"] == "user"
        assert "good movie" in messages[1]["content"]
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "positive"

    def test_without_system(self):
        config = TaskConfig(
            task={"name": "test", "type": "classification"},
            data={"source": "test", "format": "jsonl", "input_columns": ["text"], "target_column": "label"},
            prompt={"template": "{text}"},
        )
        sample = {"text": "hello", "label": "yes"}
        messages = _build_messages(sample, config)

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["content"] == "yes"

    def test_without_labels_mapping(self):
        config = TaskConfig(
            task={"name": "test", "type": "classification"},
            data={"source": "test", "format": "jsonl", "input_columns": ["text"], "target_column": "label"},
            prompt={"template": "{text}"},
        )
        sample = {"text": "hello", "label": 42}
        messages = _build_messages(sample, config)
        assert messages[-1]["content"] == "42"

    def test_multiple_input_columns(self):
        config = TaskConfig(
            task={"name": "test", "type": "classification"},
            data={
                "source": "test",
                "format": "jsonl",
                "input_columns": ["code", "language"],
                "target_column": "target",
            },
            prompt={"template": "Review this {language} code:\n{code}"},
        )
        sample = {"code": "int x = 0;", "language": "C", "target": 0}
        messages = _build_messages(sample, config)
        assert "C" in messages[0]["content"]
        assert "int x = 0;" in messages[0]["content"]


class TestConvertToChatFormat:
    def test_conversion(self, sample_dataset, classification_config):
        result = convert_to_chat_format(sample_dataset, classification_config)
        assert "messages" in result.column_names
        assert len(result) == 4
        assert result[0]["messages"][2]["content"] == "positive"
        assert result[1]["messages"][2]["content"] == "negative"


class TestBalanceByLabel:
    def test_balance_caps(self, classification_config):
        dataset = Dataset.from_dict(
            {
                "text": [f"sample_{i}" for i in range(20)],
                "label": [0] * 15 + [1] * 5,
            }
        )
        classification_config.data.max_samples_per_label = 5
        balanced = _balance_by_label(dataset, classification_config)
        assert len(balanced) == 10

    def test_no_balance_when_none(self, classification_config, sample_dataset):
        classification_config.data.max_samples_per_label = None
        result = _balance_by_label(sample_dataset, classification_config)
        assert len(result) == len(sample_dataset)
