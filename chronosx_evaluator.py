import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from chronosx_covariate_ds import TimeSeriesCovariateDataset
from chronos.x_model import AdaptedXModel

class AdaptedXModelEvaluator:
    """
    Comprehensive evaluation suite for AdaptedXModel.
    """

    def __init__(
        self,
        model: AdaptedXModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def evaluate_dataset(
        self,
        test_dataset: TimeSeriesCovariateDataset,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Evaluate model on a test dataset.

        Args:
            test_dataset: Test dataset
            batch_size: Batch size for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        all_predictions = []
        all_targets = []
        all_errors = []

        self.logger.info("Starting evaluation...")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = {
                    k: v.to(self.device) if v is not None else None
                    for k, v in batch.items()
                }

                try:
                    # Generate multiple samples for probabilistic evaluation
                    batch_predictions = []
                    input_ids, attention_mask, scale = self.model.model_wrapper.input_transform(batch["target"])
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        scale=scale,
                        past_covariates=batch["past_covariates"],
                        future_covariates=batch["future_covariates"],
                        num_samples=self.model.model_wrapper.model.config.num_samples,
                        max_length=self.model.model_wrapper.model.config.prediction_length,
                    )

                    # Extract predictions
                    if hasattr(outputs, "outputs"):
                        preds = outputs.outputs
                    elif hasattr(outputs, "logits"):
                        preds = outputs.logits
                    else:
                        preds = outputs

                    # Handle shape mismatches
                    if len(preds.shape) == 3 and len(batch["target"].shape) == 2:
                        preds = preds.mean(dim=1)

                    batch_predictions.append(preds.cpu())

                    # Stack predictions: [num_samples, batch_size, prediction_length]
                    batch_predictions = torch.stack(batch_predictions)
                    targets = batch["target"].cpu()

                    all_predictions.append(batch_predictions)
                    all_targets.append(targets)

                    # Calculate errors for this batch
                    mean_pred = batch_predictions.mean(dim=0)
                    batch_errors = torch.abs(mean_pred - targets).mean(dim=1)
                    all_errors.extend(batch_errors.tolist())

                except Exception as e:
                    self.logger.error(f"Error in evaluation step: {e.with_traceback()}")
                    continue

        # Concatenate all predictions and targets
        all_predictions = torch.cat(
            all_predictions, dim=1
        )  # [num_samples, total_samples, pred_len]
        all_targets = torch.cat(all_targets, dim=0)  # [total_samples, pred_len]

        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)

        return metrics, all_predictions, all_targets

    def _collate_fn(self, batch):
        """Custom collate function."""
        collated = {}

        for key in ["input_data", "target", "mask"]:
            collated[key] = torch.stack([item[key] for item in batch])

        for key in ["past_covariates", "future_covariates"]:
            items = [item[key] for item in batch if item[key] is not None]
            if items:
                collated[key] = torch.stack(items)
            else:
                collated[key] = None

        return collated

    def _calculate_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            predictions: [num_samples, batch_size, pred_length]
            targets: [batch_size, pred_length]
        """
        # Point forecast metrics (using mean prediction)
        mean_pred = predictions.mean(dim=0)  # [batch_size, pred_length]

        # Flatten for metric calculation
        mean_pred_flat = mean_pred.flatten().numpy()
        targets_flat = targets.flatten().numpy()

        # Point forecast metrics
        mae = mean_absolute_error(targets_flat, mean_pred_flat)
        mape = mean_absolute_percentage_error(targets_flat, mean_pred_flat)
        mse = mean_squared_error(targets_flat, mean_pred_flat)
        rmse = np.sqrt(mse)

        # MASE (Mean Absolute Scaled Error) - simplified version
        # Note: This is a simplified MASE calculation
        naive_forecast = torch.roll(targets, 1, dims=1)[:, 1:]  # Simple naive forecast
        naive_errors = torch.abs(targets[:, 1:] - naive_forecast).mean()
        mase = mae / (naive_errors.item() + 1e-8)

        # Probabilistic metrics
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantile_preds = torch.quantile(predictions, torch.tensor(quantiles), dim=0)

        # Quantile Loss
        quantile_losses = []
        for i, q in enumerate(quantiles):
            q_pred = quantile_preds[i]
            error = targets - q_pred
            loss = torch.maximum(q * error, (q - 1) * error).mean()
            quantile_losses.append(loss.item())

        # Weighted Quantile Loss (WQL)
        weights = 2.0 / torch.sum(torch.abs(targets))
        wql = sum(quantile_losses) / len(quantile_losses)

        # Coverage metrics
        coverage_80 = self._calculate_coverage(predictions, targets, 0.1, 0.9)
        coverage_90 = self._calculate_coverage(predictions, targets, 0.05, 0.95)

        # Prediction interval width (normalized by target scale)
        target_scale = torch.abs(targets).mean()
        pi_width_80 = (
            torch.quantile(predictions, 0.9, dim=0)
            - torch.quantile(predictions, 0.1, dim=0)
        ).mean() / target_scale
        pi_width_90 = (
            torch.quantile(predictions, 0.95, dim=0)
            - torch.quantile(predictions, 0.05, dim=0)
        ).mean() / target_scale

        metrics = {
            "MAE": mae,
            "MAPE": mape,
            "MSE": mse,
            "RMSE": rmse,
            "MASE": mase,
            "WQL": wql,
            "Coverage_80": coverage_80,
            "Coverage_90": coverage_90,
            "PI_Width_80": pi_width_80.item(),
            "PI_Width_90": pi_width_90.item(),
            "Mean_Quantile_Loss": np.mean(quantile_losses),
        }

        return metrics

    def _calculate_coverage(self, predictions, targets, lower_q, upper_q):
        """Calculate prediction interval coverage."""
        lower_bound = torch.quantile(predictions, lower_q, dim=0)
        upper_bound = torch.quantile(predictions, upper_q, dim=0)

        coverage = ((targets >= lower_bound) & (targets <= upper_bound)).float().mean()
        return coverage.item()

    def compare_with_baseline(
        self, test_dataset: TimeSeriesCovariateDataset, baseline_model: Any = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare AdaptedXModel with baseline (e.g., model without covariates).
        """
        # Evaluate AdaptedXModel
        adapted_metrics = self.evaluate_dataset(test_dataset)

        results = {"AdaptedXModel": adapted_metrics}

        if baseline_model is not None:
            baseline_evaluator = AdaptedXModelEvaluator(baseline_model, self.device)
            baseline_metrics = baseline_evaluator.evaluate_dataset(test_dataset)
            results["Baseline"] = baseline_metrics

            # Calculate improvement
            improvements = {}
            for metric in adapted_metrics:
                if metric in baseline_metrics:
                    if metric in ["Coverage_80", "Coverage_90"]:  # Higher is better
                        improvement = (
                            adapted_metrics[metric] - baseline_metrics[metric]
                        ) * 100
                    else:  # Lower is better
                        improvement = (
                            (baseline_metrics[metric] - adapted_metrics[metric])
                            / baseline_metrics[metric]
                            * 100
                        )
                    improvements[f"{metric}_improvement_%"] = improvement

            results["Improvements"] = improvements

        return results

    # TODO rewrite this to generate plots based on pred and true array inputs
    def generate_forecast_plots(
        self,
        test_dataset: TimeSeriesCovariateDataset,
        num_series: int = 5,
        num_samples: int = 100,
        save_path: Optional[str] = None,
    ):
        """Generate forecast visualization plots."""
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        fig, axes = plt.subplots(num_series, 1, figsize=(12, 3 * num_series))
        if num_series == 1:
            axes = [axes]

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_series:
                    break

                batch = {
                    k: v.to(self.device) if v is not None else None
                    for k, v in batch.items()
                }

                # Generate samples
                samples = []
                for _ in range(num_samples):
                    input_ids, attention_mask, scale = self.model.model_wrapper.encode(batch["target"])
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        scale=scale,
                        past_covariates=batch["past_covariates"],
                        future_covariates=batch["future_covariates"],
                        num_samples=num_samples,
                        max_length=self.model.model_wrapper.model.config.prediction_length,
                    )

                    if hasattr(outputs, "prediction_outputs"):
                        pred = outputs.prediction_outputs
                    elif hasattr(outputs, "logits"):
                        pred = outputs.logits
                    else:
                        pred = outputs

                    if len(pred.shape) == 3:
                        pred = pred.mean(dim=1)

                    samples.append(pred.cpu().squeeze())

                samples = torch.stack(samples)
                target = batch["target"].cpu().squeeze()
                context = batch["input_data"].cpu().squeeze()

                # Plot
                ax = axes[i]

                # Plot context
                ctx_x = range(len(context))
                ax.plot(ctx_x, context, "b-", label="Context", alpha=0.7)

                # Plot target
                target_x = range(len(context), len(context) + len(target))
                ax.plot(target_x, target, "r-", label="True", linewidth=2)

                # Plot prediction quantiles
                pred_mean = samples.mean(dim=0)
                pred_10 = torch.quantile(samples, 0.1, dim=0)
                pred_90 = torch.quantile(samples, 0.9, dim=0)

                ax.plot(target_x, pred_mean, "g-", label="Prediction", linewidth=2)
                ax.fill_between(
                    target_x, pred_10, pred_90, alpha=0.3, color="green", label="80% PI"
                )

                ax.set_title(f"Time Series {i+1}")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
