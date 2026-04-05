import contextlib
import gc
import os
from pathlib import Path
from typing import ClassVar

with contextlib.suppress(ImportError):
    import unsloth  # noqa: F401

from loguru import logger

from forge.config import TaskConfig
from forge.data import load_dataset_from_config

try:
    import torch
    from transformers import EarlyStoppingCallback, TrainerCallback

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TrainerCallback = object


class MetricsCallback(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.csv_path = output_dir / "training_metrics.csv"
        self.csv_file = None
        self.csv_writer = None

    def on_train_begin(self, args, state, control, **kwargs):
        import csv

        self.csv_file = self.csv_path.open("w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "epoch", "loss", "learning_rate", "grad_norm", "gpu_memory_gb", "tokens_per_second", "timestamp"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        import time

        step = state.global_step
        epoch = state.epoch or 0
        loss = logs.get("loss", logs.get("eval_loss"))
        lr = logs.get("learning_rate")
        grad_norm = logs.get("grad_norm")
        tokens_per_sec = logs.get("tokens_per_second")

        gpu_mem = None
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e9

        if self.csv_writer and loss is not None:
            self.csv_writer.writerow([step, epoch, loss, lr, grad_norm, gpu_mem, tokens_per_sec, time.time()])
            self.csv_file.flush()

        lr_str = f"{lr:.2e}" if lr else "N/A"
        grad_str = f"{grad_norm:.4f}" if grad_norm else "N/A"
        gpu_str = f"{gpu_mem:.2f}GB" if gpu_mem else "N/A"
        loss_str = f"{loss:.4f}" if loss else "N/A"

        logger.info(f"Step {step:>4} | Epoch {epoch:.3f} | Loss: {loss_str} | LR: {lr_str} | Grad: {grad_str} | GPU: {gpu_str}")

    def on_train_end(self, args, state, control, **kwargs):
        self.close()

    def close(self):
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            logger.info(f"Metrics saved to {self.csv_path}")

    def __del__(self):
        self.close()


def _log_memory(stage: str):
    import psutil

    proc = psutil.Process()
    ram_gb = proc.memory_info().rss / 1e9
    gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    logger.info(f"[MEM {stage}] RAM: {ram_gb:.2f}GB, GPU: {gpu_gb:.2f}GB")


def _cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Trainer:
    TEMPLATE_MARKERS: ClassVar[dict[str, dict[str, str]]] = {
        "chatml": {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n",
        },
        "llama-3.1": {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
        "mistral": {
            "instruction_part": "[INST]",
            "response_part": "[/INST]",
        },
    }

    def __init__(self, config: TaskConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup(self):
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        tc = self.config.training
        logger.info(f"Loading model: {tc.model}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=tc.model,
            max_seq_length=tc.max_seq_length,
            dtype=None,
            load_in_4bit=tc.load_in_4bit,
        )

        if tc.chat_template:
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template=tc.chat_template,
                map_eos_token=True,
            )

        lora = tc.lora
        logger.info("Applying LoRA adapters")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora.r,
            target_modules=lora.target_modules,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            bias=lora.bias,
            use_gradient_checkpointing=tc.gradient_checkpointing,
            use_rslora=lora.use_rslora,
            random_state=tc.seed,
        )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        pct = (100 * trainable / total) if total > 0 else 0.0
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

        _cleanup_memory()

    def _resolve_output_dir(self) -> Path:
        if self.config.training.output_dir:
            output = Path(self.config.training.output_dir)
        else:
            model_short = self.config.training.model.split("/")[-1].replace("-bnb-4bit", "").replace("-unsloth", "")
            output = Path("output") / self.config.task.name / model_short

        output.mkdir(parents=True, exist_ok=True)
        return output

    def _format_dataset(self, dataset, label: str):
        from datasets import Dataset as HFDataset

        _log_memory(f"before_format_{label}")
        logger.info(f"Formatting {len(dataset)} {label} samples")

        texts = []
        for i, example in enumerate(dataset):
            text = self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
            if (i + 1) % 1000 == 0:
                logger.info(f"  Formatted {i + 1}/{len(dataset)} {label} samples")

        result = HFDataset.from_dict({"text": texts})
        del texts
        _cleanup_memory()
        _log_memory(f"after_format_{label}")
        return result

    def _apply_response_only_training(self, trainer):
        template = self.config.training.chat_template
        if not template:
            return trainer

        markers = self.TEMPLATE_MARKERS.get(template)
        if not markers:
            logger.warning(f"No response markers for template '{template}', training on full sequence")
            return trainer

        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(trainer, **markers, num_proc=1)
        logger.info(f"Applied response-only training for {template}")
        return trainer

    def _setup_mlflow(self, output_dir: Path):
        import mlflow

        mlflow.set_tracking_uri(str(output_dir / "mlruns"))
        mlflow.set_experiment(self.config.task.name)

        run = mlflow.start_run(run_name=self.config.training.model.split("/")[-1])

        mlflow.log_params(
            {
                "model": self.config.training.model,
                "task": self.config.task.name,
                "task_type": self.config.task.type,
                "data_source": self.config.data.source,
                "epochs": self.config.training.epochs,
                "batch_size": self.config.training.batch_size,
                "learning_rate": self.config.training.learning_rate,
                "lora_r": self.config.training.lora.r,
                "lora_alpha": self.config.training.lora.alpha,
                "max_seq_length": self.config.training.max_seq_length,
                "optimizer": self.config.training.optimizer,
                "lr_scheduler": self.config.training.lr_scheduler,
            }
        )

        return run

    def _push_to_hub(self, output_dir: Path):
        repo_id = self.config.export.hub_repo
        if not repo_id:
            return

        logger.info(f"Pushing LoRA adapter to HuggingFace Hub: {repo_id}")
        self.model.push_to_hub(repo_id, private=self.config.export.hub_private)
        self.tokenizer.push_to_hub(repo_id, private=self.config.export.hub_private)
        logger.info(f"Pushed to {repo_id}")

    def train(self) -> Path:
        from trl import SFTConfig, SFTTrainer

        if self.model is None:
            self.setup()

        output_dir = self._resolve_output_dir()
        self._setup_mlflow(output_dir)

        train_dataset, eval_dataset = load_dataset_from_config(self.config)

        os.environ["HF_DATASETS_DISABLE_PARALLEL_PROCESSING"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        train_formatted = self._format_dataset(train_dataset, "train")
        eval_formatted = self._format_dataset(eval_dataset, "eval")

        tc = self.config.training
        _log_memory("before_sftconfig")

        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=tc.epochs,
            per_device_train_batch_size=tc.batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            warmup_steps=tc.warmup_steps,
            warmup_ratio=tc.warmup_ratio,
            weight_decay=tc.weight_decay,
            lr_scheduler_type=tc.lr_scheduler,
            neftune_noise_alpha=tc.neftune_noise_alpha,
            label_smoothing_factor=tc.label_smoothing,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=tc.logging_steps,
            save_steps=tc.save_steps,
            eval_strategy="steps",
            eval_steps=tc.eval_steps,
            save_total_limit=tc.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim=tc.optimizer,
            seed=tc.seed,
            max_length=tc.max_seq_length,
            packing=False,
            dataset_text_field="text",
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataset_num_proc=1,
            report_to="mlflow",
            include_tokens_per_second=True,
            include_num_input_tokens_seen=True,
        )

        _log_memory("before_sfttrainer")
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_formatted,
            eval_dataset=eval_formatted,
            args=training_args,
        )
        _log_memory("after_sfttrainer")

        trainer = self._apply_response_only_training(trainer)

        metrics_callback = MetricsCallback(output_dir)
        trainer.add_callback(metrics_callback)
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=tc.early_stopping_patience))

        try:
            logger.info("Starting training")
            trainer.train()
        finally:
            metrics_callback.close()

        lora_dir = output_dir / "lora"
        logger.info(f"Saving LoRA adapter to {lora_dir}")
        self.model.save_pretrained(lora_dir)
        self.tokenizer.save_pretrained(lora_dir)

        self._push_to_hub(output_dir)

        import mlflow

        mlflow.end_run()

        logger.info("Training completed")
        return output_dir
