import pytest
import yaml
from pydantic import ValidationError

from forge.config import TaskConfig, _deep_merge, load_task_config


class TestDeepMergeEdgeCases:
    def test_list_override_not_append(self):
        base = {"modules": ["a", "b", "c"]}
        override = {"modules": ["x", "y"]}
        result = _deep_merge(base, override)
        assert result["modules"] == ["x", "y"]

    def test_none_value_in_override(self):
        base = {"key": "value", "other": "stays"}
        override = {"key": None}
        result = _deep_merge(base, override)
        assert result["key"] is None
        assert result["other"] == "stays"

    def test_empty_base(self):
        result = _deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self):
        result = _deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_both_empty(self):
        assert _deep_merge({}, {}) == {}

    def test_base_not_mutated(self):
        base = {"training": {"epochs": 3, "lr": 0.001}}
        override = {"training": {"epochs": 5}}
        _deep_merge(base, override)
        assert base["training"]["epochs"] == 3

    def test_override_replaces_dict_with_scalar(self):
        base = {"training": {"epochs": 3}}
        override = {"training": "replaced"}
        result = _deep_merge(base, override)
        assert result["training"] == "replaced"

    def test_override_replaces_scalar_with_dict(self):
        base = {"key": "scalar"}
        override = {"key": {"nested": 1}}
        result = _deep_merge(base, override)
        assert result["key"] == {"nested": 1}

    def test_three_level_nesting(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _deep_merge(base, override)
        assert result["a"]["b"]["c"] == 99
        assert result["a"]["b"]["d"] == 2


class TestMultiLevelInheritance:
    def test_three_level_inheritance(self, tmp_path):
        grandparent = tmp_path / "grandparent.yaml"
        grandparent.write_text(
            yaml.dump(
                {
                    "task": {"name": "base-task"},
                    "data": {"source": "base/data", "format": "jsonl"},
                    "training": {"epochs": 10, "learning_rate": 1e-4},
                }
            )
        )

        parent = tmp_path / "parent.yaml"
        parent.write_text(
            yaml.dump(
                {
                    "inherits": "grandparent.yaml",
                    "training": {"epochs": 5},
                }
            )
        )

        child = tmp_path / "child.yaml"
        child.write_text(
            yaml.dump(
                {
                    "inherits": "parent.yaml",
                    "task": {"name": "child-task"},
                    "data": {"source": "child/data", "format": "csv"},
                    "training": {"batch_size": 4},
                }
            )
        )

        config = load_task_config(child)
        assert config.task.name == "child-task"
        assert config.data.source == "child/data"
        assert config.training.batch_size == 4
        assert config.training.epochs == 5
        assert config.training.learning_rate == pytest.approx(1e-4)

    def test_child_overrides_grandparent_value(self, tmp_path):
        grandparent = tmp_path / "gp.yaml"
        grandparent.write_text(
            yaml.dump(
                {
                    "task": {"name": "gp-task"},
                    "data": {"source": "gp/data", "format": "jsonl"},
                    "training": {"epochs": 10},
                }
            )
        )

        parent = tmp_path / "parent.yaml"
        parent.write_text(yaml.dump({"inherits": "gp.yaml", "training": {"epochs": 5}}))

        child = tmp_path / "child.yaml"
        child.write_text(
            yaml.dump(
                {
                    "inherits": "parent.yaml",
                    "task": {"name": "child-task"},
                    "data": {"source": "c/data", "format": "csv"},
                    "training": {"epochs": 1},
                }
            )
        )

        config = load_task_config(child)
        assert config.training.epochs == 1


class TestCircularInheritanceDetection:
    def test_self_referential_inherits(self, tmp_path):
        config_file = tmp_path / "self.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "inherits": "self.yaml",
                    "task": {"name": "test"},
                    "data": {"source": "test", "format": "jsonl"},
                }
            )
        )
        with pytest.raises(ValueError, match="Circular inheritance"):
            load_task_config(config_file)

    def test_mutual_circular_inherits(self, tmp_path):
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(
            yaml.dump(
                {
                    "inherits": "b.yaml",
                    "task": {"name": "a"},
                    "data": {"source": "a/data", "format": "jsonl"},
                }
            )
        )
        b.write_text(
            yaml.dump(
                {
                    "inherits": "a.yaml",
                    "task": {"name": "b"},
                    "data": {"source": "b/data", "format": "jsonl"},
                }
            )
        )
        with pytest.raises(ValueError, match="Circular inheritance"):
            load_task_config(a)


class TestInvalidYAML:
    def test_malformed_yaml(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("key: [unclosed bracket")
        with pytest.raises(Exception, match=r"."):
            load_task_config(bad_file)

    def test_empty_yaml_file(self, tmp_path):
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        with pytest.raises((ValidationError, KeyError, TypeError)):
            load_task_config(empty_file)

    def test_yaml_with_only_comments(self, tmp_path):
        comment_file = tmp_path / "comments.yaml"
        comment_file.write_text("# this is just a comment\n# nothing here\n")
        with pytest.raises((ValidationError, KeyError, TypeError)):
            load_task_config(comment_file)

    def test_yaml_wrong_top_level_type(self, tmp_path):
        list_file = tmp_path / "list.yaml"
        list_file.write_text("- item1\n- item2\n")
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            load_task_config(list_file)


class TestFieldValidation:
    def test_invalid_task_type(self):
        with pytest.raises(ValidationError):
            TaskConfig(
                task={"name": "test", "type": "invalid_type"},
                data={"source": "test", "format": "jsonl"},
            )

    def test_invalid_data_format(self):
        with pytest.raises(ValidationError):
            TaskConfig(
                task={"name": "test"},
                data={"source": "test", "format": "excel"},
            )

    def test_invalid_lora_bias(self):
        with pytest.raises(ValidationError):
            TaskConfig(
                task={"name": "test"},
                data={"source": "test", "format": "jsonl"},
                training={"lora": {"bias": "invalid_bias"}},
            )

    def test_valid_task_types(self):
        for task_type in ("classification", "generation"):
            config = TaskConfig(
                task={"name": "test", "type": task_type},
                data={"source": "test", "format": "jsonl"},
            )
            assert config.task.type == task_type

    def test_valid_formats(self):
        for fmt in ("parquet", "jsonl", "csv", "hf_dataset"):
            config = TaskConfig(
                task={"name": "test"},
                data={"source": "test", "format": fmt},
            )
            assert config.data.format == fmt

    def test_valid_lora_bias_values(self):
        for bias in ("none", "all", "lora_only"):
            config = TaskConfig(
                task={"name": "test"},
                data={"source": "test", "format": "jsonl"},
                training={"lora": {"bias": bias}},
            )
            assert config.training.lora.bias == bias

    def test_missing_task_section(self):
        with pytest.raises(ValidationError):
            TaskConfig(data={"source": "test", "format": "jsonl"})

    def test_missing_data_section(self):
        with pytest.raises(ValidationError):
            TaskConfig(task={"name": "test"})

    def test_unknown_fields_ignored(self):
        config = TaskConfig(
            task={"name": "test"},
            data={"source": "test", "format": "jsonl"},
            unknown_field="should_be_ignored",
        )
        assert config.task.name == "test"


class TestEdgeCases:
    def test_empty_labels_dict(self):
        config = TaskConfig(
            task={"name": "test"},
            data={"source": "test", "format": "jsonl", "labels": {}},
        )
        assert config.data.labels == {}

    def test_empty_input_columns_list(self):
        config = TaskConfig(
            task={"name": "test"},
            data={"source": "test", "format": "jsonl", "input_columns": []},
        )
        assert config.data.input_columns == []

    def test_negative_test_size_is_rejected(self):
        with pytest.raises(ValidationError, match="test_size"):
            TaskConfig(
                task={"name": "test"},
                data={"source": "test", "format": "jsonl", "test_size": -0.1},
            )

    def test_test_size_greater_than_one_is_rejected(self):
        with pytest.raises(ValidationError, match="test_size"):
            TaskConfig(
                task={"name": "test"},
                data={"source": "test", "format": "jsonl", "test_size": 1.5},
            )

    def test_test_size_zero_is_rejected(self):
        with pytest.raises(ValidationError, match="test_size"):
            TaskConfig(
                task={"name": "test"},
                data={"source": "test", "format": "jsonl", "test_size": 0.0},
            )

    def test_test_size_one_is_rejected(self):
        with pytest.raises(ValidationError, match="test_size"):
            TaskConfig(
                task={"name": "test"},
                data={"source": "test", "format": "jsonl", "test_size": 1.0},
            )

    def test_valid_test_size_accepted(self):
        config = TaskConfig(
            task={"name": "test"},
            data={"source": "test", "format": "jsonl", "test_size": 0.2},
        )
        assert config.data.test_size == pytest.approx(0.2)

    def test_none_labels(self):
        config = TaskConfig(
            task={"name": "test"},
            data={"source": "test", "format": "jsonl", "labels": None},
        )
        assert config.data.labels is None

    def test_numeric_label_keys_normalized_to_strings(self):
        config = TaskConfig(
            task={"name": "test"},
            data={"source": "test", "format": "jsonl", "labels": {0: "negative", 1: "positive"}},
        )
        assert "0" in config.data.labels
        assert "1" in config.data.labels
        assert config.data.labels["0"] == "negative"
        assert config.data.labels["1"] == "positive"

    def test_target_modules_list_override(self):
        custom_modules = ["q_proj", "v_proj"]
        config = TaskConfig(
            task={"name": "test"},
            data={"source": "test", "format": "jsonl"},
            training={"lora": {"target_modules": custom_modules}},
        )
        assert config.training.lora.target_modules == custom_modules

    def test_task_name_required(self):
        with pytest.raises(ValidationError):
            TaskConfig(
                task={"type": "classification"},
                data={"source": "test", "format": "jsonl"},
            )

    def test_data_source_required(self):
        with pytest.raises(ValidationError):
            TaskConfig(
                task={"name": "test"},
                data={"format": "jsonl"},
            )


class TestCLIOverridesWithInheritance:
    def test_override_after_inheritance(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text(
            yaml.dump(
                {
                    "task": {"name": "base"},
                    "data": {"source": "base/data", "format": "jsonl"},
                    "training": {"epochs": 10, "learning_rate": 1e-4},
                }
            )
        )
        child = tmp_path / "child.yaml"
        child.write_text(
            yaml.dump(
                {
                    "inherits": "base.yaml",
                    "task": {"name": "child"},
                    "data": {"source": "child/data", "format": "csv"},
                }
            )
        )

        config = load_task_config(child)
        config.training.epochs = 2
        config.training.lora.r = 8
        config.training.lora.alpha = 16

        assert config.training.epochs == 2
        assert config.training.lora.r == 8
        assert config.training.lora.alpha == 16
        assert config.training.learning_rate == pytest.approx(1e-4)
