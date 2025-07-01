import ast
import logging
import os
import sys
import json
import itertools
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import List, Iterator, Optional, Dict

import typer
from typer_config import use_yaml_config
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset
import transformers
from transformers import (
    TrainingArguments,
    Trainer,
)

# ChronosX-specific imports
from src.chronos.chronosx import ChronosXModel
from src.chronos.tokenizer import ChronosTokenizer

app = typer.Typer(pretty_exceptions_enable=False)

def is_main_process() -> bool:
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ.get("RANK", 0)) == 0

def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    if is_main_process():
        logger.log(log_level, msg)

def save_training_info(ckpt_path: Path, training_config: Dict):
    assert ckpt_path.is_dir()
    with open(ckpt_path / "training_info.json", "w") as fp:
        json.dump({"training_config": training_config}, fp, indent=4)

def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False

class ChronosXDataset(IterableDataset):
    def __init__(
        self,
        datasets: list,
        probabilities: list,
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        mode: str = "training",
        np_dtype=np.float32,
    ):
        super().__init__()
        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")
        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.mode = mode
        self.np_dtype = np_dtype

    def preprocess_entry(self, entry: dict) -> dict:
        # Assume entry has 'target', 'timestamp', and optionally covariates
        entry = {k: entry[k] for k in entry if k in ["target", "timestamp", "static_cats", "future_covariates"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        entry["timestamp"] = pd.to_datetime(entry["timestamp"])
        return entry

    def time_covariates(self, timestamps):
        # Example: hour, dayofweek, month
        return np.stack([
            (timestamps.hour / 23.0) - 0.5,
            (timestamps.dayofweek / 6.0) - 0.5,
            (timestamps.month / 12.0) - 0.5,
        ], axis=-1)

    def __iter__(self):
        # Sample datasets according to probabilities
        datasets = [map(self.preprocess_entry, ds) for ds in self.datasets]
        iterables = datasets  # For now, no shuffling
        for dataset in iterables:
            for entry in dataset:
                target = entry["target"]
                timestamps = entry["timestamp"]
                n = len(target)
                # Rolling windows for training
                max_start = n - self.context_length - self.prediction_length
                if max_start < 1:
                    continue
                for start in range(0, max_start, self.prediction_length):
                    context = target[start:start + self.context_length]
                    future = target[start + self.context_length:start + self.context_length + self.prediction_length]
                    context_timestamps = timestamps[start:start + self.context_length]
                    future_timestamps = timestamps[start + self.context_length:start + self.context_length + self.prediction_length]
                    # Covariates
                    past_covariates = self.time_covariates(context_timestamps)
                    future_covariates = self.time_covariates(future_timestamps)
                    # Tokenize
                    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
                    future_tensor = torch.tensor(future, dtype=torch.float32).unsqueeze(0)
                    # Tokenizer: adapt as needed for ChronosX
                    input_ids, attention_mask, _ = self.tokenizer.context_input_transform(context_tensor)
                    labels, labels_mask = self.tokenizer.label_input_transform(future_tensor, 1.0)
                    labels[labels_mask == 0] = -100
                    yield {
                        "input_ids": input_ids.squeeze(0),
                        "attention_mask": attention_mask.squeeze(0),
                        "labels": labels.squeeze(0),
                        "past_covariates": torch.tensor(past_covariates, dtype=torch.float32),
                        "future_covariates": torch.tensor(future_covariates, dtype=torch.float32),
                    }

@app.command()
@use_yaml_config(param_name="config")
def main(
    training_data_paths: str,
    probability: Optional[str] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    min_past: int = 64,
    max_steps: int = 200_000,
    save_steps: int = 50_000,
    log_steps: int = 500,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 1e-3,
    optim: str = "adamw_torch_fused",
    shuffle_buffer_length: int = 100,
    gradient_accumulation_steps: int = 2,
    model_id: str = "google/t5-efficient-tiny",
    model_type: str = "seq2seq",
    random_init: bool = False,
    tie_embeddings: bool = False,
    output_dir: str = "./output/",
    tf32: bool = True,
    torch_compile: bool = True,
    tokenizer_class: str = "MeanScaleUniformBins",
    tokenizer_kwargs: str = "{'low_limit': -15.0, 'high_limit': 15.0}",
    n_tokens: int = 4096,
    n_special_tokens: int = 2,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    use_eos_token: bool = True,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.0,
    dataloader_num_workers: int = 1,
    max_missing_prop: float = 0.9,
    num_samples: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: Optional[int] = None,
):
    if tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        tf32 = False

    if seed is None:
        seed = random.randint(0, 2**32)
    log_on_main(f"Using SEED: {seed}", logger)
    transformers.set_seed(seed=seed)

    raw_training_config = deepcopy(locals())
    output_dir = Path(output_dir)
    training_data_paths = ast.literal_eval(training_data_paths)
    assert isinstance(training_data_paths, list)

    if isinstance(probability, str):
        probability = ast.literal_eval(probability)
    elif probability is None:
        probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
    assert isinstance(probability, list)
    assert len(training_data_paths) == len(probability)

    if dataloader_num_workers > len(training_data_paths):
        log_on_main(
            f"Setting the number of data loader workers to {len(training_data_paths)}, "
            f"instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(training_data_paths)

    if isinstance(tokenizer_kwargs, str):
        tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
    assert isinstance(tokenizer_kwargs, dict)
    assert model_type in ["seq2seq", "causal"]

    # Output dir logic
    run_dir = output_dir / "chronosx_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_on_main(f"Logging dir: {run_dir}", logger)
    log_on_main(f"Loading and filtering {len(training_data_paths)} datasets for training: {training_data_paths}", logger)
    log_on_main(f"Mixing probabilities: {probability}", logger)

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]

    log_on_main("Initializing model", logger)

    # ChronosX-specific: create tokenizer and model
    tokenizer = ChronosTokenizer(tokenizer_class, **tokenizer_kwargs)
    model = ChronosXModel(
        base_model_id=model_id,
        model_type=model_type,
        vocab_size=n_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        random_init=random_init,
        tie_embeddings=tie_embeddings,
        # Add additional ChronosX-specific args as needed
    )

    # Dataset: replace with ChronosX-aware dataset
    train_dataset = ChronosXDataset(
        datasets=train_datasets,
        probabilities=probability,
        tokenizer=tokenizer,
        context_length=context_length,
        prediction_length=prediction_length,
        # Add additional dataset args as needed
    )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim=optim,
        logging_dir=str(run_dir / "logs"),
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=["tensorboard"],
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        tf32=tf32,
        torch_compile=torch_compile,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    log_on_main("Training", logger)
    trainer.train()

    if is_main_process():
        model.save_pretrained(run_dir / "checkpoint-final")
        save_training_info(run_dir / "checkpoint-final", training_config=raw_training_config)

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()
