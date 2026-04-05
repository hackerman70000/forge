import random
from collections import defaultdict

from datasets import Dataset, load_dataset
from loguru import logger

from forge.config import TaskConfig


def _load_raw_dataset(config: TaskConfig) -> Dataset:
    source = config.data.source
    fmt = config.data.format

    if fmt == "hf_dataset":
        logger.info(f"Loading dataset from HuggingFace Hub: {source}")
        ds = load_dataset(source, split=config.data.split or "train", keep_in_memory=False)
    elif fmt == "jsonl":
        logger.info(f"Loading JSONL from: {source}")
        ds = load_dataset("json", data_files=source, split="train", keep_in_memory=False)
    elif fmt == "parquet":
        logger.info(f"Loading Parquet from: {source}")
        ds = load_dataset("parquet", data_files=source, split="train", keep_in_memory=False)
    elif fmt == "csv":
        logger.info(f"Loading CSV from: {source}")
        ds = load_dataset("csv", data_files=source, split="train", keep_in_memory=False)
    else:
        msg = f"Unsupported data format: {fmt}"
        raise ValueError(msg)

    logger.info(f"Loaded {len(ds)} samples")
    return ds


def _filter_by_split(dataset: Dataset, config: TaskConfig) -> Dataset:
    split_col = config.data.split_column
    split_val = config.data.split
    if not split_col or not split_val:
        return dataset
    if split_col not in dataset.column_names:
        logger.warning(f"Split column '{split_col}' not found, skipping filter")
        return dataset

    filtered = dataset.filter(lambda x: x[split_col] == split_val)
    logger.info(f"Filtered by {split_col}={split_val}: {len(dataset)} -> {len(filtered)}")
    return filtered


def _balance_by_label(dataset: Dataset, config: TaskConfig) -> Dataset:
    max_per_label = config.data.max_samples_per_label
    if not max_per_label:
        return dataset

    target_col = config.data.target_column
    random.seed(config.data.seed)

    label_indices: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(dataset):
        label = str(sample[target_col])
        label_indices[label].append(idx)

    selected = []
    for label, indices in label_indices.items():
        if len(indices) > max_per_label:
            logger.info(f"  Label '{label}': {len(indices)} -> {max_per_label}")
            indices = random.sample(indices, max_per_label)
        selected.extend(indices)

    selected.sort()
    balanced = dataset.select(selected)
    logger.info(f"Balanced dataset: {len(dataset)} -> {len(balanced)}")
    return balanced


class _DefaultDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _build_messages(sample: dict, config: TaskConfig) -> list[dict[str, str]]:
    template_vars = _DefaultDict({col: sample.get(col, "") for col in config.data.input_columns})
    for key, value in sample.items():
        if key not in template_vars:
            template_vars[key] = value

    user_content = config.prompt.template.format_map(template_vars)

    target_value = sample[config.data.target_column]
    assistant_content = config.data.labels.get(str(target_value), str(target_value)) if config.data.labels else str(target_value)

    messages = []
    if config.prompt.system:
        messages.append({"role": "system", "content": config.prompt.system})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})
    return messages


def convert_to_chat_format(dataset: Dataset, config: TaskConfig) -> Dataset:
    logger.info(f"Converting {len(dataset)} samples to chat format")

    if len(dataset) == 0:
        return Dataset.from_dict({"messages": []})

    def _convert_batch(batch: dict) -> dict:
        batch_size = len(next(iter(batch.values())))
        all_messages = []
        for i in range(batch_size):
            sample = {col: batch[col][i] for col in batch}
            messages = _build_messages(sample, config)
            all_messages.append(messages)
        return {"messages": all_messages}

    return dataset.map(
        _convert_batch,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
    )


def load_dataset_from_config(config: TaskConfig) -> tuple[Dataset, Dataset]:
    raw = _load_raw_dataset(config)
    raw = _filter_by_split(raw, config)
    raw = _balance_by_label(raw, config)

    chat_ds = convert_to_chat_format(raw, config)
    chat_ds = chat_ds.shuffle(seed=config.data.seed, keep_in_memory=False)

    split = chat_ds.train_test_split(
        test_size=config.data.test_size,
        seed=config.data.seed,
        keep_in_memory=False,
    )

    logger.info(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")
    return split["train"], split["test"]
