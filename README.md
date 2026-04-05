# Forge

Generic LLM fine-tuning framework with YAML-driven task configuration.

Fine-tune any LLM on any classification or generation task by defining a single YAML config. Forge handles data loading (HuggingFace Hub, Parquet, CSV, JSONL), auto-conversion to chat format, QLoRA training via Unsloth, and experiment tracking with MLflow.

## Quick Start

```bash
uv sync --extra training --extra dev

forge train tasks/vulnllm.yaml
forge train tasks/vulnllm.yaml --model unsloth/Qwen3-32B-bnb-4bit --epochs 5
forge merge-lora output/lora/
```

## Task Config

````yaml
inherits: base.yaml

task:
  name: vulnllm-detection
  type: classification

data:
  format: hf_dataset
  source: UCSB-SURFI/VulnLLM-R-Train-Data
  eval_source: UCSB-SURFI/VulnLLM-R-Test-Data
  input_columns: [code, language]
  target_column: target
  labels:
    "0": not_vulnerable
    "1": vulnerable

prompt:
  system: You are a security code reviewer.
  template: |
    Analyze this {language} function:
    ```
    {code}
    ```

training:
  model: unsloth/Llama-3.3-70B-Instruct-bnb-4bit
  lora:
    r: 32
    alpha: 64
  epochs: 3

export:
  hub_repo: my-org/my-model
````

Configs support inheritance (`inherits: base.yaml`) with deep merge and CLI overrides.

## Dev

```bash
inv setup          # install all deps
inv test           # run tests
inv check          # lint + format + test
inv validate tasks/vulnllm.yaml
```

Full usage guide: [docs/USAGE.md](docs/USAGE.md)
