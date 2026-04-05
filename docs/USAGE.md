# Usage Guide

## Installation

```bash
# Core (config, data loading, CLI)
uv sync

# With training dependencies (Unsloth, PyTorch, etc.)
uv sync --extra training

# With HuggingFace Hub push support
uv sync --extra hub

# Development (tests, linting)
uv sync --extra dev
```

## Quick Start

```bash
# Train a model using a task config
forge train tasks/vulnllm.yaml

# Override parameters from CLI
forge train tasks/vulnllm.yaml --model unsloth/Qwen3-32B-bnb-4bit --epochs 5

# Merge LoRA adapter into base model
forge merge-lora output/vulnllm-detection/lora/
```

## Task Configuration

Every training run is driven by a YAML config file. A config defines what data to use, how to prompt the model, and training hyperparameters.

### Minimal Config

```yaml
task:
  name: my-task
  type: classification # classification | generation

data:
  source: my-org/my-dataset # HuggingFace dataset ID or local path
  target_column: label
```

### Full Config Reference

````yaml
task:
  name: vulnllm-detection
  type: classification

data:
  format: hf_dataset # hf_dataset | parquet | jsonl | csv
  source: UCSB-SURFI/VulnLLM-R-Train-Data
  eval_source: UCSB-SURFI/VulnLLM-R-Test-Data  # separate eval dataset (optional)
  eval_split: function_level  # HF split for eval dataset (optional)
  input_columns: # columns used in prompt template
    - code
    - language
  target_column: target # column with labels
  labels: # map raw values -> human-readable labels
    "0": not_vulnerable
    "1": vulnerable
  split: function_level # HF dataset split name
  split_column: split # column to filter by (after loading)
  test_size: 0.1 # fraction for eval set (0 < x < 1)
  max_samples_per_label: 500 # cap per label for balancing
  seed: 3407

prompt:
  system: |
    You are a security code reviewer.
  template: |
    Analyze this {language} function:

    ```
    {code}
    ```
  enable_thinking: false  # disable <think> tags for models like Qwen3.5 (default: true)

training:
  model: unsloth/Llama-3.3-70B-Instruct-bnb-4bit
  output_dir: output/vulnllm # default: output/<task-name>/<model-short>
  load_in_4bit: true
  chat_template: chatml # chatml | llama-3.1 | mistral | null
  gradient_checkpointing: unsloth
  lora:
    r: 32
    alpha: 64
    dropout: 0.05
    bias: none # none | all | lora_only
    use_rslora: true
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
  epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 2e-4
  warmup_steps: 100
  warmup_ratio: 0.0
  weight_decay: 0.01
  max_seq_length: 2048
  optimizer: adamw_torch_fused
  lr_scheduler: linear # linear | cosine
  neftune_noise_alpha: null # e.g. 5.0 for NEFTune
  label_smoothing: 0.0
  logging_steps: 1
  save_steps: 100
  eval_steps: 50
  save_total_limit: 3
  early_stopping_patience: 3
  seed: 3407

export:
  lora: true
  hub_repo: my-org/my-model # push to HuggingFace Hub after training
  hub_private: true
````

### Config Inheritance

Configs can inherit from a base config using `inherits`. Child values override parent values with deep merge (nested dicts are merged, not replaced).

```yaml
# base.yaml
training:
  lora:
    r: 32
    alpha: 64
  epochs: 3
  learning_rate: 2e-4

export:
  lora: true
```

```yaml
# vulnllm.yaml
inherits: base.yaml

task:
  name: vulnllm-detection

data:
  source: UCSB-SURFI/VulnLLM-R-Train-Data

training:
  epochs: 5 # overrides base
  # lora.r remains 32 from base
```

Multi-level inheritance works: `child.yaml -> parent.yaml -> grandparent.yaml`. Circular references are detected and rejected.

## Data Sources

Forge loads data from multiple sources and automatically converts it to chat format for training.

### HuggingFace Hub

```yaml
data:
  format: hf_dataset
  source: UCSB-SURFI/VulnLLM-R-Train-Data
  split: train
```

### Local Files

```yaml
# JSONL
data:
  format: jsonl
  source: data/training.jsonl

# CSV
data:
  format: csv
  source: data/training.csv

# Parquet
data:
  format: parquet
  source: data/training.parquet
```

### Separate Eval Dataset

By default, forge auto-splits the training data into train/eval using `test_size`. To use a dedicated eval dataset instead:

```yaml
data:
  source: UCSB-SURFI/VulnLLM-R-Train-Data
  eval_source: UCSB-SURFI/VulnLLM-R-Test-Data
  eval_split: function_level  # optional: HF split for eval dataset
```

When `eval_source` is set, `test_size` is ignored.

### Auto-Conversion to Chat Format

Forge converts raw data into chat messages using your prompt template:

**Input** (raw parquet row):

```json
{ "code": "void foo() { ... }", "language": "C", "target": 1 }
```

**Output** (chat format for training):

````json
{
  "messages": [
    { "role": "system", "content": "You are a security code reviewer." },
    {
      "role": "user",
      "content": "Analyze this C function:\n\n```\nvoid foo() { ... }\n```"
    },
    { "role": "assistant", "content": "vulnerable" }
  ]
}
````

The template uses Python format strings. Variables are resolved from:

1. Columns listed in `input_columns`
2. All other columns in the dataset row

Missing variables render as empty strings.

### Label Mapping

The `labels` dict maps raw target values to training labels:

```yaml
data:
  target_column: target
  labels:
    "0": not_vulnerable
    "1": vulnerable
```

Numeric keys are automatically normalized to strings, so both `0: negative` and `"0": negative` work.

If a target value has no mapping, it passes through as a string.

## CLI Reference

### `forge train`

```
forge train <config_path> [OPTIONS]

Arguments:
  config_path          Path to task YAML config

Options:
  -m, --model          Override model name
  -e, --epochs         Override number of epochs
  -b, --batch-size     Override per-device batch size
  --lr                 Override learning rate
  --lora-r             Override LoRA rank (alpha auto-set to 2x)
  -o, --output         Override output directory
  --log-level          Logging level [default: info]
```

CLI overrides take precedence over YAML config, which takes precedence over inherited base config.

### `forge merge-lora`

Merge a LoRA adapter into the base model for deployment with vLLM or other inference engines that don't support dynamic LoRA loading.

```
forge merge-lora <lora_path> [OPTIONS]

Arguments:
  lora_path            Path to LoRA adapter directory

Options:
  -o, --output         Output path [default: <lora_path>/../merged]
  -b, --base-model     Base model [default: auto-detect from adapter_config.json]
  --log-level          Logging level [default: info]
```

The base model is auto-detected from `adapter_config.json`. If a 4-bit quantized model is detected, the FP16 variant is used instead.

### Thinking Mode

Models like Qwen3.5 generate `<think>...</think>` reasoning tags by default. For simple classification tasks this is unnecessary overhead. Disable it in the prompt section:

```yaml
prompt:
  enable_thinking: false
```

## Dev Tasks (Invoke)

```bash
inv setup              # install all deps (training + hub + dev)
inv setup --cpu        # install dev deps only (no GPU)
inv test               # run test suite
inv test --cov         # with coverage report
inv test --html        # with HTML coverage
inv test -k "config"   # filter by pattern
inv lint               # ruff check
inv lint --fix         # auto-fix
inv format             # ruff format
inv format --check     # check only
inv check              # lint + format + test (all-in-one)
inv validate <config>  # validate and print resolved config
```

## Experiment Tracking

Forge uses MLflow to track experiments. Each training run logs:

- All hyperparameters (model, LoRA config, learning rate, etc.)
- Training metrics (loss, learning rate, gradient norm per step)
- GPU memory usage

MLflow data is stored locally at `<output_dir>/mlruns/`. View it with:

```bash
cd output/vulnllm-detection/
mlflow ui
```

A CSV file (`training_metrics.csv`) is also written alongside for quick inspection.

## Output Structure

After training, the output directory contains:

```
output/vulnllm-detection/Llama-3.3-70B-Instruct/
├── lora/                      # LoRA adapter weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── mlruns/                    # MLflow experiment tracking
├── training_metrics.csv       # Per-step metrics
├── forge.log                  # Full training log
└── checkpoint-*/              # Intermediate checkpoints
```

## Examples

### Binary Classification (VulnLLM)

```bash
forge train tasks/vulnllm.yaml
```

### Custom Sentiment Analysis

```yaml
# tasks/sentiment.yaml
inherits: base.yaml

task:
  name: sentiment
  type: classification

data:
  format: hf_dataset
  source: stanfordnlp/imdb
  input_columns: [text]
  target_column: label
  labels:
    "0": negative
    "1": positive
  test_size: 0.1

prompt:
  system: Classify the sentiment of this movie review as positive or negative.
  template: |
    Review: {text}

training:
  model: unsloth/Llama-3.1-8B-Instruct-bnb-4bit
  chat_template: llama-3.1
  epochs: 2
```

### Code Generation Task

````yaml
# tasks/code-repair.yaml
inherits: base.yaml

task:
  name: code-repair
  type: generation

data:
  format: jsonl
  source: data/code_repairs.jsonl
  input_columns: [buggy_code, language, error_message]
  target_column: fixed_code

prompt:
  system: You are an expert programmer. Fix the bug in the given code.
  template: |
    Language: {language}
    Error: {error_message}

    Buggy code:
    ```
    {buggy_code}
    ```

    Provide the fixed code.

training:
  model: unsloth/Qwen3-32B-bnb-4bit
  max_seq_length: 4096
  epochs: 3
````

### Running Experiments with Overrides

```bash
# Quick test run
forge train tasks/vulnllm.yaml --epochs 1 --batch-size 1

# Try a different model
forge train tasks/vulnllm.yaml -m unsloth/Llama-3.1-8B-Instruct-bnb-4bit

# Higher LoRA rank experiment
forge train tasks/vulnllm.yaml --lora-r 64 -o output/vulnllm-r64

# Push to Hub after training
# (set hub_repo in YAML config, then run normally)
forge train tasks/vulnllm-hub.yaml
```
