import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from chronosx_covariate_ds import TimeSeriesCovariateDataset

class AdaptedXModelTrainer:
    """
    Trainer class for AdaptedXModel with covariate support.
    """

    def __init__(
        self,
        model: Any,  # AdaptedXModel
        train_dataset: TimeSeriesCovariateDataset,
        val_dataset: Optional[TimeSeriesCovariateDataset] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./checkpoints",
        patience: int = 10,
        min_delta: float = 1e-4,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.min_delta = min_delta

        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
            )

        # Setup optimizer (only train adapter weights)
        adapter_params = list(self.model.covariate_adapter.parameters())
        self.optimizer = optim.AdamW(adapter_params, lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _collate_fn(self, batch):
        """Custom collate function to handle variable-length sequences and None values."""
        collated = {}

        # Handle input_data, target, mask
        for key in ["input_data", "target", "mask"]:
            collated[key] = torch.stack([item[key] for item in batch])

        # Handle optional covariates
        for key in ["past_covariates", "future_covariates"]:
            items = [item[key] for item in batch if item[key] is not None]
            if items:
                collated[key] = torch.stack(items)
            else:
                collated[key] = None

        return collated

    def _compute_loss(self, outputs, targets):
        """Compute training loss. This is a simplified version - adapt based on your model's output format."""
        if hasattr(outputs, "logits"):
            predictions = outputs.logits
        elif hasattr(outputs, "outputs"):
            predictions = outputs.outputs
        else:
            predictions = outputs

        # Reshape if needed to match target dimensions
        if len(predictions.shape) == 3 and len(targets.shape) == 2:
            # Average across the sequence dimension or take the last prediction
            predictions = predictions.mean(dim=1)

        assert predictions.device == targets.device

        # TODO implement loss function determined by model type
        loss = nn.CrossEntropyLoss()(predictions, targets) # Chronos uses cross entropy
        return loss

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            batch = {
                k: v.to(self.device) if v is not None else None
                for k, v in batch.items()
            }

            self.optimizer.zero_grad()

            try:
                # Forward pass
                outputs = self.model(
                    input_data=batch["input_data"],
                    mask=batch["mask"],
                    past_covariates=batch["past_covariates"],
                    future_covariates=batch["future_covariates"],
                )

                # Compute loss
                loss = self._compute_loss(outputs, batch["target"])

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.covariate_adapter.parameters(), max_norm=1.0
                )

                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix({"loss": loss.item()})

            except Exception as e:
                self.logger.error(f"Error in training step: {e.with_traceback()}")
                continue

        return epoch_loss / max(num_batches, 1)

    def validate(self):
        """Validate the model."""
        if not self.val_dataset:
            return None

        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {
                    k: v.to(self.device) if v is not None else None
                    for k, v in batch.items()
                }

                try:
                    outputs = self.model(
                        input_data=batch["input_data"],
                        mask=batch["mask"],
                        past_covariates=batch["past_covariates"],
                        future_covariates=batch["future_covariates"],
                    )

                    loss = self._compute_loss(outputs, batch["target"])
                    epoch_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    self.logger.error(f"Error in validation step: {e}")
                    continue

        return epoch_loss / max(num_batches, 1)

    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.num_epochs} epochs")

        for epoch in range(self.num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            if val_loss is not None:
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)

            # Logging
            log_msg = (
                f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}"
            )
            if val_loss is not None:
                log_msg += f" - Val Loss: {val_loss:.6f}"
            self.logger.info(log_msg)

            # Early stopping and model saving
            if val_loss is not None:
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(epoch, is_best=True)
                else:
                    self.epochs_without_improvement += 1

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        self.logger.info("Training completed")
        self.plot_training_history()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_name = (
            "best_model.pt" if is_best else f"checkpoint_epoch_{epoch+1}.pt"
        )
        checkpoint_path = self.save_dir / checkpoint_name

        # Save only the adapter weights and configuration
        checkpoint = {
            "epoch": epoch,
            "adapter_state_dict": self.model.covariate_adapter.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "model_config": {
                "covariate_dim": self.model.covariate_dim,
                "hidden_dim": self.model.hidden_dim,
                "model_type": self.model.model_type,
            },
        }

        torch.save(checkpoint, checkpoint_path)
        if is_best:
            self.logger.info(f"Best model saved to {checkpoint_path}")

    def plot_training_history(self):
        """Plot training and validation loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss", alpha=0.8)
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss", alpha=0.8)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("AdaptedXModel Training History")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(
            self.save_dir / "training_history.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
