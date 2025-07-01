#!/usr/bin/env python3
"""
ChronosX Evaluation Script

This script provides evaluation functionality specifically for the ChronosX model,
handling its unique architecture and covariate integration.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import typer
from tqdm import tqdm

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.chronos.chronosx import ChronosXModel, ChronosBoltWrapper
from src.chronos.tokenizer import ChronosTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_chronosx_model(
    model_path: Union[str, Path],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[ChronosXModel, Dict]:
    """Load a trained ChronosX model and its configuration.
    
    Args:
        model_path: Path to the saved ChronosX model
        device: Device to load the model on
        torch_dtype: Data type for model parameters
        
    Returns:
        Tuple of (model, config)
    """
    model_path = Path(model_path)
    
    # Load model configuration
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load base model
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
    
    model_type = config.get("model_type", "seq2seq")
    model_class = AutoModelForSeq2SeqLM if model_type == "seq2seq" else AutoModelForCausalLM
    
    base_model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    
    # Create ChronosX model
    chronosx = ChronosXModel(
        pretrained_model=base_model,
        covariate_dim=config["covariate_dim"],
        d_model=config["d_model"],
        vocab_size=config["vocab_size"],
        hidden_dim=config.get("hidden_dim", 256),
        use_past_covariates=config.get("use_past_covariates", True),
        use_future_covariates=config.get("use_future_covariates", True),
        freeze_pretrained=config.get("freeze_base", True),
    )
    
    # Load model weights
    model_weights = torch.load(model_path / "pytorch_model.bin", map_location=device)
    chronosx.load_state_dict(model_weights)
    
    # Move to device and set eval mode
    chronosx = chronosx.to(device).eval()
    
    return chronosx, config


def prepare_covariates(
    timestamps: pd.DatetimeIndex,
    freq: str,
    covariate_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Prepare time-based covariates from timestamps.
    
    Args:
        timestamps: DatetimeIndex of the time series
        freq: Frequency string (e.g., 'H' for hourly, 'D' for daily)
        covariate_dim: Dimension of the covariate features
        device: Device to place the covariates on
        
    Returns:
        Tensor of shape [seq_len, covariate_dim]
    """
    # Basic time features
    features = {
        'minute': timestamps.minute / 59.0 - 0.5,
        'hour': timestamps.hour / 23.0 - 0.5,
        'day_of_week': timestamps.dayofweek / 6.0 - 0.5,
        'day_of_month': (timestamps.day - 1) / 30.0 - 0.5,
        'day_of_year': (timestamps.dayofyear - 1) / 365.0 - 0.5,
        'month': (timestamps.month - 1) / 11.0 - 0.5,
        'week_of_year': (timestamps.isocalendar().week - 1) / 52.0 - 0.5,
    }
    
    # Create DataFrame and select first covariate_dim features
    df = pd.DataFrame(features)
    if covariate_dim < len(df.columns):
        df = df.iloc[:, :covariate_dim]
    
    return torch.tensor(df.values, dtype=torch.float32, device=device)


def evaluate_chronosx(
    model: ChronosXModel,
    test_data: Dict[str, np.ndarray],
    prediction_length: int,
    context_length: int,
    covariate_dim: int,
    device: torch.device,
    num_samples: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Evaluate a ChronosX model on test data.
    
    Args:
        model: Loaded ChronosX model
        test_data: Dictionary containing 'target' and optionally 'timestamp' keys
        prediction_length: Number of time steps to predict
        context_length: Number of time steps to use as context
        covariate_dim: Dimension of covariate features
        device: Device to perform computation on
        num_samples: Number of samples to draw for probabilistic forecasting
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        
    Returns:
        Dictionary containing predictions and metrics
    """
    model.eval()
    
    # Prepare data
    target = torch.tensor(test_data['target'], dtype=torch.float32, device=device)
    timestamps = pd.DatetimeIndex(test_data.get('timestamp', []))
    
    # Generate covariates
    if len(timestamps) > 0:
        all_covariates = prepare_covariates(
            timestamps=timestamps,
            freq=pd.infer_freq(timestamps) or 'D',  # Default to daily frequency
            covariate_dim=covariate_dim,
            device=device,
        )
        
        # Split into past and future covariates
        past_covariates = all_covariates[:context_length]
        future_covariates = all_covariates[context_length:context_length + prediction_length]
    else:
        past_covariates = future_covariates = None
    
    # Prepare input sequence
    context = target[:context_length].unsqueeze(0)  # Add batch dimension
    
    # Generate predictions
    with torch.no_grad():
        # Forward pass through ChronosX
        outputs = model(
            context,
            past_covariates=past_covariates.unsqueeze(0) if past_covariates is not None else None,
            future_covariates=future_covariates.unsqueeze(0) if future_covariates is not None else None,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    # Process outputs
    if isinstance(outputs, tuple):
        # Handle different output formats
        predictions = outputs[0].cpu().numpy()
    else:
        predictions = outputs.cpu().numpy()
    
    # Calculate metrics
    mase = np.mean(np.abs(predictions - target[context_length:].numpy()))
    mwsdl = np.mean(np.abs(predictions - target[context_length:].numpy()) / (1 + np.abs(target[context_length:].numpy())))
    mape = np.mean(np.abs((predictions - target[context_length:].numpy()) / target[context_length:].numpy()))
    
    metrics = {
        'MASE': mase,
        'MWSDL': mwsdl,
        'MAPE': mape,
    }
    
    return {
        'predictions': predictions,
        'metrics': metrics,
        'context': context.cpu().numpy(),
    }


app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    config_path: Path,
    metrics_path: Path,
    model_path: str = typer.Option(..., help="Path to the trained ChronosX model"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device to use"),
    torch_dtype: str = typer.Option("bfloat16", help="Data type for model parameters"),
    batch_size: int = typer.Option(32, help="Batch size for evaluation"),
    num_samples: int = typer.Option(100, help="Number of samples for probabilistic forecasting"),
    temperature: Optional[float] = typer.Option(None, help="Sampling temperature"),
    top_k: Optional[int] = typer.Option(None, help="Top-k sampling"),
    top_p: Optional[float] = typer.Option(None, help="Nucleus sampling"),
):
    """Evaluate a trained ChronosX model on test data."""
    # Set up device
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model, config = load_chronosx_model(model_path, device, torch_dtype=torch_dtype)
    
    # Load evaluation configs
    with open(config_path) as fp:
        backtest_configs = json.load(fp)
    
    result_rows = []
    for config in backtest_configs:
        # Load and preprocess dataset (replace with ChronosX-aware logic)
        test_data = {
            'target': np.random.rand(1000),  # Replace with actual data
            'timestamp': pd.date_range('2022-01-01', periods=1000, freq='H'),  # Replace with actual timestamps
        }
        
        # Evaluate model on test_data
        results = evaluate_chronosx(
            model=model,
            test_data=test_data,
            prediction_length=config['prediction_length'],
            context_length=config.get('context_length', 512),
            covariate_dim=config["covariate_dim"],
            device=device,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        result_rows.extend([results['metrics']])
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(result_rows, f, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {metrics_path}")


if __name__ == "__main__":
    app()
