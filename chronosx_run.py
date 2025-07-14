import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")

from src.chronos.x_model import load_adapted_x_model
from chronosx_covariate_ds import create_datasets
from chronosx_trainer import AdaptedXModelTrainer
from chronosx_evaluator import AdaptedXModelEvaluator

def run_full_experiment(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    data_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a complete training and evaluation experiment.

    Args:
        model_config: Configuration for AdaptedXModel
        training_config: Training hyperparameters
        data_config: Data generation/loading configuration

    Returns:
        Dictionary containing all results and metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting full experiment...")

    train_dataset, val_dataset, test_dataset = create_datasets(
        data_source=data_config["data_path"],
        target_column=data_config.get("target_column"),
        covariate_columns=data_config.get("covariate_columns"),
        context_length=training_config.get("context_length", 512),
        prediction_length=training_config.get("prediction_length", 64),
        train_ratio=data_config.get("train_ratio", 0.6),
        val_ratio=data_config.get("val_ratio", 0.2),
        test_ratio=data_config.get("test_ratio", None),
        step_size=data_config.get("step_size", 1),
        random_seed=data_config.get("random_seed", 42),
    )

    logger.info(
        f"Created datasets - Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset) if val_dataset else 0}, "
        f"Test: {len(test_dataset)}"
    )

    # Initialize model
    model = load_adapted_x_model(
        model_name_or_path=model_config["base_model_path"],
        model_type=model_config["model_type"],
        covariate_dim=model_config["covariate_dim"],
        hidden_dim=model_config.get("hidden_dim", 256),
        freeze_pretrained=model_config.get("freeze_pretrained", True),
        specific_model_config=model_config.get("specific_model_config", {}),
    )

    # Initialize trainer
    trainer = AdaptedXModelTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=training_config.get("learning_rate", 1e-3),
        batch_size=training_config.get("batch_size", 32),
        num_epochs=training_config.get("num_epochs", 50),
        device=training_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        ),
        save_dir=training_config.get("save_dir", "./checkpoints"),
        patience=training_config.get("patience", 10),
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Initialize evaluator
    logger.info("Starting evaluation...")
    evaluator = AdaptedXModelEvaluator(model, trainer.device)

    # Evaluate model
    test_metrics, predictions, targets = evaluator.evaluate_dataset(
        test_dataset,
        batch_size=training_config.get("batch_size", 32),
    )

    # Generate forecast plots
    evaluator.generate_forecast_plots(
        test_dataset,
        num_series=5,
        save_path=str(trainer.save_dir / "forecast_plots.png"),
    )

    # Export predictions and targets to a single pandas CSV
    predictions_df = pd.DataFrame({"predictions": predictions.flatten(), "targets": targets.flatten()})
    predictions_df.to_csv(str(trainer.save_dir / "predictions.csv"), index=False)

    # Compile results
    results = {
        "model_config": model_config,
        "training_config": training_config,
        "data_config": data_config,
        "final_train_loss": trainer.train_losses[-1] if trainer.train_losses else None,
        "final_val_loss": trainer.val_losses[-1] if trainer.val_losses else None,
        "best_val_loss": trainer.best_val_loss,
        "test_metrics": test_metrics,
        "training_history": {
            "train_losses": trainer.train_losses,
            "val_losses": trainer.val_losses,
        },
    }

    # Save results
    results_path = trainer.save_dir / "experiment_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in results.items()
            if k != "training_history"
        }
        json_results["training_history"] = results["training_history"]
        json.dump(json_results, f, indent=2)

    logger.info(f"Experiment completed. Results saved to {results_path}")

    return results


def hyperparameter_search(
    base_model_config: Dict[str, Any],
    base_data_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    n_trials: int = 10,
) -> Dict[str, Any]:
    """
    Perform hyperparameter search for AdaptedXModel.

    Args:
        base_model_config: Base model configuration
        base_data_config: Base data configuration
        param_grid: Dictionary of hyperparameters to search
        n_trials: Number of random trials to run

    Returns:
        Best configuration and results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting hyperparameter search with {n_trials} trials...")

    best_score = float("inf")
    best_config = None
    all_results = []

    for trial in range(n_trials):
        logger.info(f"Trial {trial + 1}/{n_trials}")

        # Sample hyperparameters
        trial_config = {}
        for param, values in param_grid.items():
            trial_config[param] = np.random.choice(values)

        # Create full configuration
        training_config = {
            **trial_config,
            "num_epochs": 20,  # Reduced for hyperparameter search
            "save_dir": f"./hp_search/trial_{trial + 1}",
            "patience": 5,
        }

        try:
            # Run experiment
            results = run_full_experiment(
                model_config=base_model_config,
                training_config=training_config,
                data_config=base_data_config,
            )

            # Use validation loss as optimization metric
            score = results["best_val_loss"]
            results["trial_config"] = trial_config
            results["trial_score"] = score

            all_results.append(results)

            if score < best_score:
                best_score = score
                best_config = trial_config
                logger.info(f"New best score: {best_score:.6f}")

        except Exception as e:
            logger.error(f"Trial {trial + 1} failed: {e}")
            continue

    # Save hyperparameter search results
    hp_results = {
        "best_config": best_config,
        "best_score": best_score,
        "all_results": all_results,
        "param_grid": param_grid,
    }

    with open("./hp_search_results.json", "w") as f:
        json.dump(hp_results, f, indent=2)

    logger.info("Hyperparameter search completed")
    return hp_results


if __name__ == "__main__":
    model_config = {
        "base_model_path": "amazon/chronos-t5-tiny",  # or other supported models
        "model_type": "chronos",
        "covariate_dim": 7,
        "hidden_dim": 256,
        "freeze_pretrained": True,
        "specific_model_config": {
            "num_samples": 50,
        }
    }

    training_config = {
        "learning_rate": 1e-3,
        "batch_size": 16,
        "num_epochs": 100,
        "context_length": 512,
        "prediction_length": 48,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "./experiment_results",
        "patience": 10,
    }

    # data_config = {num
    #     "data_path": "~/models/Time-LLM/dataset/demand/demand_data_all_cleaned_featsel.csv",
    #     "train_ratio": 0.6,  # 60% training
    #     "val_ratio": 0.2,  # 20% validation, 20% test (1 - 0.6 - 0.2 = 0.2)
    # }
    data_config = {
        "data_path": "~/models/Time-LLM/dataset/demand/demand_data_all_cleaned_featsel.csv",
        "train_ratio": 0.2, # for testing purposes only
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    }

    # Run single experiment
    print("Running single experiment...")
    results = run_full_experiment(model_config, training_config, data_config)

    print("\nExperiment Results:")
    print(f"Final Training Loss: {results['final_train_loss']:.6f}")
    print(f"Final Validation Loss: {results['final_val_loss']:.6f}")
    print(f"Best Validation Loss: {results['best_val_loss']:.6f}")

    print("\nTest Metrics:")
    for metric, value in results["test_metrics"].items():
        print(f"  {metric}: {value:.6f}")

    # # Run hyperparameter search
    # print("\nRunning hyperparameter search...")
    # param_grid = {
    #     "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    #     "batch_size": [8, 16, 32],
    #     "hidden_dim": [128, 256, 512],
    # }

    # hp_results = hyperparameter_search(
    #     base_model_config=model_config,
    #     base_data_config=data_config,
    #     param_grid=param_grid,
    #     n_trials=5,  # Reduced for example
    # )

    # print(f"\nBest hyperparameters: {hp_results['best_config']}")
    # print(f"Best validation loss: {hp_results['best_score']:.6f}")


# Additional utility functions for advanced evaluation
# def cross_validation_experiment(
#     model_config: Dict[str, Any],
#     training_config: Dict[str, Any],
#     data_config: Dict[str, Any],
#     n_folds: int = 5
# ) -> Dict[str, Any]:
#     """
#     Perform k-fold cross-validation experiment.
#     """
#     logger = logging.getLogger(__name__)
#     logger.info(f"Starting {n_folds}-fold cross-validation...")

#     # Generate data
#     time_series, past_covs, future_covs = create_synthetic_data(
#         num_series=data_config.get('num_series', 100),
#         series_length=data_config.get('series_length', 1000),
#         covariate_dim=data_config.get('covariate_dim', 5),
#         noise_level=data_config.get('noise_level', 0.1)
#     )

#     fold_size = len(time_series) // n_folds
#     all_metrics = []

#     for fold in range(n_folds):
#         logger.info(f"Fold {fold + 1}/{n_folds}")

#         # Create train/test split for this fold
#         test_start = fold * fold_size
#         test_end = (fold + 1) * fold_size

#         test_ts = time_series[test_start:test_end]
#         train_ts = time_series[:test_start] + time_series[test_end:]

#         test_past_cov = past_covs[test_start:test_end] if past_covs else None
#         train_past_cov = (past_covs[:test_start] + past_covs[test_end:]) if past_covs else None

#         test_future_cov = future_covs[test_start:test_end] if future_covs else None
#         train_future_cov = (future_covs[:test_start] + future_covs[test_end:]) if future_covs else None

#         # Create datasets
#         train_dataset = TimeSeriesCovariateDataset(
#             train_ts, train_past_cov, train_future_cov,
#             training_config['context_length'], training_config['prediction_length'],
#             mode='training'
#         )

#         test_dataset = TimeSeriesCovariateDataset(
#             test_ts, test_past_cov, test_future_cov,
#             training_config['context_length'], training_config['prediction_length'],
#             mode='test'
#         )

#         # Train and evaluate
#         model = MockAdaptedXModel(  # Replace with actual model loading
#             covariate_dim=model_config['covariate_dim'],
#             hidden_dim=model_config.get('hidden_dim', 256)
#         )

#         trainer = AdaptedXModelTrainer(
#             model=model,
#             train_dataset=train_dataset,
#             val_dataset=None,
#             **{k: v for k, v in training_config.items()
#                if k in ['learning_rate', 'batch_size', 'device']},
#             num_epochs=20,  # Reduced for CV
#             save_dir=f'./cv_fold_{fold + 1}',
#             patience=5
#         )

#         trainer.train()

#         evaluator = AdaptedXModelEvaluator(model, trainer.device)
#         metrics = evaluator.evaluate_dataset(test_dataset)
#         all_metrics.append(metrics)

#     # Aggregate results
#     aggregated_metrics = {}
#     for metric in all_metrics[0].keys():
#         values = [fold_metrics[metric] for fold_metrics in all_metrics]
#         aggregated_metrics[f'{metric}_mean'] = np.mean(values)
#         aggregated_metrics[f'{metric}_std'] = np.std(values)

#     cv_results = {
#         'aggregated_metrics': aggregated_metrics,
#         'fold_metrics': all_metrics,
#         'n_folds': n_folds
#     }

#     logger.info("Cross-validation completed")
#     return cv_results

# def ablation_study(
#     base_model_config: Dict[str, Any],
#     base_training_config: Dict[str, Any],
#     base_data_config: Dict[str, Any]
# ) -> Dict[str, Any]:
#     """
#     Perform ablation study to understand component contributions.
#     """
#     logger = logging.getLogger(__name__)
#     logger.info("Starting ablation study...")

#     ablation_configs = {
#         'full_model': base_model_config,
#         'no_past_covariates': {**base_model_config},
#         'no_future_covariates': {**base_model_config},
#         'no_covariates': {**base_model_config},
#         'small_hidden': {**base_model_config, 'hidden_dim': 64},
#         'large_hidden': {**base_model_config, 'hidden_dim': 512}
#     }

#     ablation_results = {}

#     for config_name, model_config in ablation_configs.items():
#         logger.info(f"Running ablation: {config_name}")

#         try:
#             results = run_full_experiment(
#                 model_config=model_config,
#                 training_config={
#                     **base_training_config,
#                     'num_epochs': 30,  # Reduced for ablation
#                     'save_dir': f'./ablation_{config_name}'
#                 },
#                 data_config=base_data_config
#             )

#             ablation_results[config_name] = results['test_metrics']

#         except Exception as e:
#             logger.error(f"Ablation {config_name} failed: {e}")
#             continue

#     # Calculate relative improvements
#     if 'no_covariates' in ablation_results and 'full_model' in ablation_results:
#         baseline = ablation_results['no_covariates']
#         full_model = ablation_results['full_model']

#         improvements = {}
#         for metric in baseline.keys():
#             if metric in ['Coverage_80', 'Coverage_90']:  # Higher is better
#                 improvement = (full_model[metric] - baseline[metric]) * 100
#             else:  # Lower is better
#                 improvement = (baseline[metric] - full_model[metric]) / baseline[metric] * 100
#             improvements[f'{metric}_improvement_%'] = improvement

#         ablation_results['improvements_over_baseline'] = improvements

#     logger.info("Ablation study completed")
#     return ablation_results
