import atexit
import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from forge.utils import setup_logging

app = typer.Typer(name="forge", help="Generic LLM fine-tuning framework")

_clean_exit = False


def _on_exit():
    if not _clean_exit:
        logger.error("Process exited unexpectedly")


def _excepthook(exc_type, exc_value, exc_tb):
    logger.exception(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
    sys.__excepthook__(exc_type, exc_value, exc_tb)


@app.command()
def train(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to task YAML config"),
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override model from config"),
    ] = None,
    epochs: Annotated[
        int | None,
        typer.Option("--epochs", "-e", help="Override epochs from config"),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", "-b", help="Override batch size from config"),
    ] = None,
    learning_rate: Annotated[
        float | None,
        typer.Option("--lr", help="Override learning rate from config"),
    ] = None,
    lora_r: Annotated[
        int | None,
        typer.Option("--lora-r", help="Override LoRA rank from config"),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Override output directory"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level"),
    ] = "info",
):
    from forge.config import load_task_config
    from forge.train import Trainer

    task_config = load_task_config(config_path)

    if model:
        task_config.training.model = model
    if epochs:
        task_config.training.epochs = epochs
    if batch_size:
        task_config.training.batch_size = batch_size
    if learning_rate:
        task_config.training.learning_rate = learning_rate
    if lora_r:
        task_config.training.lora.r = lora_r
        task_config.training.lora.alpha = lora_r * 2
    if output_dir:
        task_config.training.output_dir = str(output_dir)

    output = Path(task_config.training.output_dir) if task_config.training.output_dir else Path("output") / task_config.task.name
    output.mkdir(parents=True, exist_ok=True)
    setup_logging(level=log_level.upper(), log_file=output / "forge.log")

    sys.excepthook = _excepthook
    atexit.register(_on_exit)

    _log_config(task_config)

    global _clean_exit
    try:
        trainer = Trainer(task_config)
        result_dir = trainer.train()
        logger.info(f"Model saved to: {result_dir}")
        _clean_exit = True
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Training failed: {e}")
        _clean_exit = True
        raise typer.Exit(1) from None


@app.command()
def merge_lora(
    lora_path: Annotated[
        Path,
        typer.Argument(help="Path to LoRA adapter directory"),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output path for merged model"),
    ] = None,
    base_model: Annotated[
        str | None,
        typer.Option("--base-model", "-b", help="Base model (auto-detected from adapter_config.json)"),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Logging level"),
    ] = "info",
):
    import json

    setup_logging(level=log_level.upper())

    lora_path = lora_path.resolve()
    if not lora_path.exists():
        logger.error(f"LoRA path does not exist: {lora_path}")
        raise typer.Exit(1)

    adapter_config_path = lora_path / "adapter_config.json"
    if not adapter_config_path.exists():
        logger.error(f"Missing adapter_config.json in: {lora_path}")
        raise typer.Exit(1)

    if base_model is None:
        with adapter_config_path.open() as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path")

        if base_model and "bnb-4bit" in base_model:
            fp16_model = base_model.replace("-bnb-4bit", "").replace("-unsloth", "")
            logger.warning(f"4-bit model detected: {base_model}")
            logger.info(f"Using FP16 model: {fp16_model}")
            base_model = fp16_model

    if not base_model:
        logger.error("Could not determine base model. Specify with --base-model")
        raise typer.Exit(1)

    if output is None:
        output = lora_path.parent / "merged"

    output = output.resolve()
    logger.info(f"Base model: {base_model}")
    logger.info(f"LoRA adapter: {lora_path}")
    logger.info(f"Output: {output}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(lora_path))

    logger.info("Merging weights...")
    model = model.merge_and_unload()

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    logger.info(f"Merged model saved to: {output}")


def _log_config(config):
    tc = config.training
    logger.info(f"Task: {config.task.name} ({config.task.type})")
    logger.info(f"Data: {config.data.source} ({config.data.format})")
    logger.info(f"Model: {tc.model}")
    logger.info(f"Epochs: {tc.epochs}")
    logger.info(f"Batch size: {tc.batch_size}")
    logger.info(f"Gradient accumulation: {tc.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {tc.batch_size * tc.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {tc.learning_rate}")
    logger.info(f"LoRA r={tc.lora.r}, alpha={tc.lora.alpha}")
    logger.info(f"Max seq length: {tc.max_seq_length}")
    if config.export.hub_repo:
        logger.info(f"Hub push: {config.export.hub_repo}")
