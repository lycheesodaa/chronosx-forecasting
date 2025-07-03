import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import os
from pathlib import Path


class TimeSeriesCovariateDataset(Dataset):
    """
    Consolidated dataset for time series with covariates.
    Handles all data loading, splitting, and window creation internally.
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        mode: str = "train",
        target_column: Optional[str] = None,
        covariate_columns: Optional[List[str]] = None,
        context_length: int = 512,
        prediction_length: int = 64,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = None,
        step_size: int = 1,
        min_past: int = 64,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize the dataset with all processing handled internally.

        Args:
            data_source: Path to CSV file or pandas DataFrame
            mode: "train", "val", or "test"
            target_column: Name of target column (if None, uses first column)
            covariate_columns: List of covariate column names (if None, uses all except target)
            context_length: Length of input context
            prediction_length: Length of prediction horizon
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation (test = 1 - train_ratio - val_ratio)
            step_size: Step size for sliding windows
            min_past: Minimum past observations required
            random_seed: Random seed for reproducibility
        """
        self.mode = mode
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.step_size = step_size
        self.min_past = min_past
        self.window_size = context_length + prediction_length

        if random_seed is not None:
            np.random.seed(random_seed)

        # Load and process data
        self._load_data(data_source, target_column, covariate_columns)
        self._create_overlapping_splits(train_ratio, val_ratio, test_ratio)
        self._extract_mode_data()
        self._create_sliding_windows()
        self._validate_dataset()

        print(f"Created {mode} dataset with {len(self)} samples")

    def _load_data(
        self,
        data_source: Union[str, Path, pd.DataFrame],
        target_column: Optional[str],
        covariate_columns: Optional[List[str]],
    ):
        """Load time series data from CSV or DataFrame"""
        if isinstance(data_source, (str, Path)):
            data_path = os.path.expanduser(str(data_source))
            try:
                self.df = pd.read_csv(data_path, index_col=0)
                print(f"Loaded data from {data_path}")
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not load data from {data_path}. Error: {str(e)}"
                )
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
            print("Loaded data from DataFrame")
        else:
            raise ValueError("data_source must be a file path or pandas DataFrame")

        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")

        # Determine target column
        if target_column is None:
            self.target_column = self.df.columns[0]
        else:
            if target_column not in self.df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            self.target_column = target_column

        # Determine covariate columns
        if covariate_columns is None:
            self.covariate_columns = [
                col for col in self.df.columns if col != self.target_column
            ]
        else:
            missing_cols = [
                col for col in covariate_columns if col not in self.df.columns
            ]
            if missing_cols:
                raise ValueError(f"Covariate columns not found: {missing_cols}")
            self.covariate_columns = covariate_columns

        # Extract arrays
        self.time_series = self.df[self.target_column].values.astype(np.float32)
        self.past_covariates = (
            self.df[self.covariate_columns].values.astype(np.float32)
            if self.covariate_columns
            else None
        )
        self.future_covariates = None  # Not used in this implementation

        print(f"Target column: {self.target_column}")
        print(
            f"Covariate columns ({len(self.covariate_columns)}): {self.covariate_columns}"
        )
        print(f"Time series shape: {self.time_series.shape}")
        print(
            f"Past covariates shape: {self.past_covariates.shape if self.past_covariates is not None else None}"
        )

    def _create_overlapping_splits(self, train_ratio: float, val_ratio: float, test_ratio: float = None):
        """
        Create overlapping train/val/test splits that allow validation and test
        to extend context_length into previous data.
        """
        n_total = len(self.time_series)
        test_ratio = 1.0 - train_ratio - val_ratio if test_ratio is None else test_ratio

        if test_ratio <= 0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")

        # Calculate split points for the PREDICTION portions only
        n_train_pred = int(n_total * train_ratio)
        n_val_pred = int(n_total * val_ratio)
        n_test_pred = n_total - n_train_pred - n_val_pred

        # Define prediction regions (non-overlapping)
        train_pred_end = n_train_pred
        val_pred_start = train_pred_end
        val_pred_end = val_pred_start + n_val_pred
        test_pred_start = val_pred_end
        test_pred_end = n_total

        # Define full regions (including context overlap)
        # Training: from start, no extension needed
        train_start = 0
        train_end = train_pred_end

        # Validation: extend context_length back into training data
        val_start = max(0, val_pred_start - self.context_length)
        val_end = val_pred_end

        # Test: extend context_length back into validation data
        test_start = max(0, test_pred_start - self.context_length)
        test_end = test_pred_end

        self.split_info = {
            "train": {
                "full_range": (train_start, train_end),
                "pred_range": (0, train_pred_end),  # Relative to full_range
                "total_length": train_end - train_start,
                "pred_length": train_pred_end,
            },
            "val": {
                "full_range": (val_start, val_end),
                "pred_range": (
                    val_pred_start - val_start,
                    val_end - val_start,
                ),  # Relative to full_range
                "total_length": val_end - val_start,
                "pred_length": n_val_pred,
            },
            "test": {
                "full_range": (test_start, test_end),
                "pred_range": (
                    test_pred_start - test_start,
                    test_end - test_start,
                ),  # Relative to full_range
                "total_length": test_end - test_start,
                "pred_length": n_test_pred,
            },
        }

        print(f"\nData split summary (total length: {n_total}):")
        print(
            f"  Train: Full=[{train_start}:{train_end}] ({train_end-train_start}), Pred=[0:{train_pred_end}] ({train_pred_end})"
        )
        print(
            f"  Val:   Full=[{val_start}:{val_end}] ({val_end-val_start}), Pred=[{val_pred_start-val_start}:{val_end-val_start}] ({n_val_pred})"
        )
        print(
            f"  Test:  Full=[{test_start}:{test_end}] ({test_end-test_start}), Pred=[{test_pred_start-test_start}:{test_end-test_start}] ({n_test_pred})"
        )

        # Check for sufficient data
        for mode_name, info in self.split_info.items():
            if info["total_length"] < self.window_size:
                print(
                    f"⚠️  WARNING: {mode_name} split ({info['total_length']}) < window_size ({self.window_size})"
                )

    def _extract_mode_data(self):
        """Extract data for the specified mode"""
        if self.mode not in self.split_info:
            raise ValueError(
                f"Mode '{self.mode}' not in {list(self.split_info.keys())}"
            )

        info = self.split_info[self.mode]
        start, end = info["full_range"]

        # Extract time series and covariates for this mode
        self.mode_time_series = self.time_series[start:end]
        self.mode_past_covariates = (
            self.past_covariates[start:end]
            if self.past_covariates is not None
            else None
        )

        # Store prediction range info for validation/test modes
        self.pred_start_rel, self.pred_end_rel = info["pred_range"]

        print(f"Mode '{self.mode}' data extracted:")
        print(f"  Full range: [{start}:{end}] -> {self.mode_time_series.shape}")
        print(
            f"  Prediction range (relative): [{self.pred_start_rel}:{self.pred_end_rel}]"
        )
        if self.mode_past_covariates is not None:
            print(f"  Past covariates: {self.mode_past_covariates.shape}")

    def _create_sliding_windows(self):
        """Create sliding windows from the mode-specific data"""
        total_length = len(self.mode_time_series)

        if total_length < self.window_size:
            print(f"⚠️  Cannot create windows: {total_length} < {self.window_size}")
            self.windows = []
            self.past_cov_windows = []
            return

        if self.mode == "train":
            # Training: can use any valid window
            max_start = total_length - self.window_size
            start_positions = list(range(0, max_start + 1, self.step_size))
        else:
            # Validation/Test: only use windows that predict in the prediction range
            # Window must have its prediction part fall within pred_range
            min_pred_start = self.pred_start_rel
            max_pred_start = self.pred_end_rel - self.prediction_length

            if max_pred_start < min_pred_start:
                print(f"⚠️  Cannot create windows: prediction range too small")
                self.windows = []
                self.past_cov_windows = []
                return

            # Convert prediction positions to window start positions
            min_window_start = max(0, min_pred_start - self.context_length)
            max_window_start = min(
                total_length - self.window_size, max_pred_start - self.context_length
            )

            if max_window_start < min_window_start:
                print(f"⚠️  Cannot create windows: window range invalid")
                self.windows = []
                self.past_cov_windows = []
                return

            start_positions = list(
                range(min_window_start, max_window_start + 1, self.step_size)
            )

        # Create windows
        self.windows = []
        self.past_cov_windows = []

        for start in start_positions:
            end = start + self.window_size

            # Time series window
            ts_window = self.mode_time_series[start:end]
            self.windows.append(ts_window)

            # Past covariates window
            if self.mode_past_covariates is not None:
                past_cov_window = self.mode_past_covariates[start:end]
                self.past_cov_windows.append(past_cov_window)

        if not self.past_cov_windows and self.mode_past_covariates is not None:
            self.past_cov_windows = None

        print(f"Created {len(self.windows)} sliding windows")
        if self.past_cov_windows:
            print(f"Created {len(self.past_cov_windows)} past covariate windows")

    def _validate_dataset(self):
        """Validate the created dataset"""
        if not self.windows:
            print(f"⚠️  No valid windows created for mode '{self.mode}'")
            return

        # Check window shapes
        expected_shape = (self.window_size,)
        for i, window in enumerate(self.windows[:3]):  # Check first 3
            if window.shape != expected_shape:
                raise ValueError(
                    f"Window {i} has shape {window.shape}, expected {expected_shape}"
                )

        # Check covariate shapes
        if self.past_cov_windows:
            expected_cov_shape = (self.window_size, len(self.covariate_columns))
            for i, cov_window in enumerate(self.past_cov_windows[:3]):  # Check first 3
                if cov_window.shape != expected_cov_shape:
                    raise ValueError(
                        f"Covariate window {i} has shape {cov_window.shape}, expected {expected_cov_shape}"
                    )

        print(f"✅ Dataset validation passed")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx >= len(self.windows):
            raise IndexError(
                f"Index {idx} out of range for {len(self.windows)} windows"
            )

        ts = self.windows[idx]

        if self.mode == "train":
            # Random sampling for training
            max_start = len(ts) - self.prediction_length - 1
            min_start = max(0, max_start - self.context_length + 1)
            start_idx = np.random.randint(min_start, max_start + 1)
        else:
            # Fixed sampling for validation/test
            start_idx = max(0, len(ts) - self.context_length - self.prediction_length)

        # Extract context and target
        end_context = min(
            start_idx + self.context_length, len(ts) - self.prediction_length
        )
        end_target = min(end_context + self.prediction_length, len(ts))

        context = ts[start_idx:end_context]
        target = ts[end_context:end_target]

        # Handle context padding
        if len(context) < self.context_length:
            padding_size = self.context_length - len(context)
            padding = np.full(padding_size, np.nan)
            context = np.concatenate([padding, context])
            mask = np.concatenate(
                [
                    np.zeros(padding_size, dtype=bool),
                    np.ones(len(context) - padding_size, dtype=bool),
                ]
            )
        else:
            mask = np.ones(len(context), dtype=bool)

        # Process past covariates
        past_cov = None
        if self.past_cov_windows is not None and idx < len(self.past_cov_windows):
            past_cov_data = self.past_cov_windows[idx]
            past_cov = past_cov_data[start_idx:end_context]

            if len(past_cov) < self.context_length:
                padding_size = self.context_length - len(past_cov)
                padding = np.zeros((padding_size, past_cov.shape[-1]))
                past_cov = np.concatenate([padding, past_cov], axis=0)

        # Whether or not the model uses past covariates (IIB) or future covariates (OIB) 
        # is automatically handled by the model based on whether past_covariates=None or future_covariates=None 
        return {
            "input_data": torch.FloatTensor(context),
            "target": torch.FloatTensor(target),
            "mask": torch.BoolTensor(mask),
            "past_covariates": (
                torch.FloatTensor(past_cov) if past_cov is not None else None
            ),
            "future_covariates": None,
        }

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            "mode": self.mode,
            "total_data_length": len(self.time_series),
            "mode_data_length": len(self.mode_time_series),
            "num_windows": len(self.windows),
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "window_size": self.window_size,
            "target_column": self.target_column,
            "covariate_columns": self.covariate_columns,
            "num_covariates": len(self.covariate_columns),
            "split_info": self.split_info,
        }


# Convenience functions for creating datasets
def create_datasets(
    data_source: Union[str, Path, pd.DataFrame],
    target_column: Optional[str] = None,
    covariate_columns: Optional[List[str]] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = None,
    step_size: int = 1,
    min_past: int = 64,
    random_seed: Optional[int] = None,
) -> Tuple[
    TimeSeriesCovariateDataset, TimeSeriesCovariateDataset, TimeSeriesCovariateDataset
]:
    """
    Create train, validation, and test datasets with the same configuration.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    common_kwargs = {
        "data_source": data_source,
        "target_column": target_column,
        "covariate_columns": covariate_columns,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "step_size": step_size,
        "min_past": min_past,
        "random_seed": random_seed,
    }

    train_dataset = TimeSeriesCovariateDataset(mode="train", **common_kwargs)
    val_dataset = TimeSeriesCovariateDataset(mode="val", **common_kwargs)
    test_dataset = TimeSeriesCovariateDataset(mode="test", **common_kwargs)

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    data_source: Union[str, Path, pd.DataFrame],
    batch_size: int = 32,
    target_column: Optional[str] = None,
    covariate_columns: Optional[List[str]] = None,
    context_length: int = 512,
    prediction_length: int = 64,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = None,
    step_size: int = 1,
    min_past: int = 64,
    random_seed: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = create_datasets(
        data_source=data_source,
        target_column=target_column,
        covariate_columns=covariate_columns,
        context_length=context_length,
        prediction_length=prediction_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        step_size=step_size,
        min_past=min_past,
        random_seed=random_seed,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
