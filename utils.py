#%% utils.py
"""
This file contains utility functions for train_transfer.py.
"""

import datetime
import inspect
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, balanced_accuracy_score
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Set up module-level logger
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# Import model mappings
from models.models import MODEL_CLASS_MAP, PREFIX_MAP


def get_username():
    """Get the username from environment variable."""
    return os.environ.get("USER", "mwe626")


def replace_username(path_str, username=None):
    """Replace placeholder username in path string."""
    path_str = str(path_str)
    path_str = path_str.replace(
        f"macke/{get_username()}", f"macke/{username if username else get_username()}"
    )
    return Path(path_str)


# Cache directory setup
_repo_root = Path(__file__).parent
_default_cache_dir_template = _repo_root / ".cache" / "moabb"
_raw_cache_dir_source_str = os.environ.get("MOABB_CACHE_DIR", str(_default_cache_dir_template))

CACHE_ROOT_DIR: Optional[Path] = None


def update_cache_root_dir(target_username_for_replacement: str):
    """Update the global CACHE_ROOT_DIR with username replacement."""
    global CACHE_ROOT_DIR
    CACHE_ROOT_DIR = replace_username(_raw_cache_dir_source_str, username=target_username_for_replacement)
    log.info(f"CACHE_ROOT_DIR updated to: {CACHE_ROOT_DIR}")


# Initialize with default username
update_cache_root_dir(target_username_for_replacement="mwe626")

# Regression Metrics Tracker
class RegressionMetricsTracker:
    """Tracks metrics for regression or binary classification tasks."""
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.y_true_hist = deque(maxlen=window_size)
        self.y_pred_hist = deque(maxlen=window_size)
        self.all_y_true = []
        self.all_y_pred = []

    def update(self, y_true: float, y_pred: float):
        """Update with a new true label and its prediction."""
        self.y_true_hist.append(y_true)
        self.y_pred_hist.append(y_pred)
        self.all_y_true.append(y_true)
        self.all_y_pred.append(y_pred)

    def _get_true_binary(self, history):
        """Helper to get binary labels based on the mode."""
        if not history:
            return np.array([])
        return (np.array(history) > 0.5).astype(int)

    def get_rolling_balanced_accuracy(self):
        true_binary = self._get_true_binary(self.y_true_hist)
        if len(true_binary) == 0 or len(np.unique(true_binary)) < 2:
            return np.nan
        pred_binary = (np.array(self.y_pred_hist) > 0.5).astype(int)
        return balanced_accuracy_score(true_binary, pred_binary)

    def get_rolling_roc_auc(self):
        true_binary = self._get_true_binary(self.y_true_hist)
        if len(true_binary) < 2 or len(np.unique(true_binary)) < 2:
            return np.nan
        return roc_auc_score(true_binary, self.y_pred_hist)

    def get_overall_balanced_accuracy(self):
        true_binary = self._get_true_binary(self.all_y_true)
        if len(true_binary) == 0 or len(np.unique(true_binary)) < 2:
            return np.nan
        pred_binary = (np.array(self.all_y_pred) > 0.5).astype(int)
        return balanced_accuracy_score(true_binary, pred_binary)

    def get_overall_roc_auc(self):
        true_binary = self._get_true_binary(self.all_y_true)
        if len(true_binary) < 2 or len(np.unique(true_binary)) < 2:
            return np.nan
        return roc_auc_score(true_binary, self.all_y_pred)

# Evaluate Zero-Shot
def evaluate_zero_shot(model, test_epochs, test_labels, device, batch_size=64,
                       is_extreme_mask=None, original_soft_labels=None):
    """
    Evaluate a pretrained model, returning metrics for all trials and a subset of 'extreme' trials.
    Assumes `test_labels` and `original_soft_labels` are always continuous (float) values.
    """
    model.eval()

    # If there's no data, return a NaN-filled dictionary immediately.
    if test_epochs is None or test_epochs.size == 0:
        return {
            "balanced_accuracy_all": np.nan, "roc_auc_all": np.nan, "r2_all": np.nan, "mse_all": np.nan,
            "balanced_accuracy_extreme": np.nan, "roc_auc_extreme": np.nan, "r2_extreme": np.nan, "mse_extreme": np.nan
        }

    all_preds_prob = []
    # Create a DataLoader for efficient batch processing
    from torch.utils.data import TensorDataset, DataLoader
    eval_dataset = TensorDataset(torch.from_numpy(test_epochs).float())
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    # Get model predictions for all test epochs
    with torch.no_grad():
        for (epochs_batch,) in eval_loader:  # Note the comma to unpack the single-element tuple
            epochs_batch = epochs_batch.to(device)
            logits = model(epochs_batch)
            preds = torch.sigmoid(logits)
            all_preds_prob.extend(preds.cpu().flatten().tolist())

    all_preds_prob = np.array(all_preds_prob)
    all_true_labels = np.array(test_labels)  # These are now ALWAYS soft (float) labels.

    metrics = {}

    # --- 1. Calculate metrics for ALL trials ---
    # Derive hard labels (0/1) from the true soft labels for classification metrics.
    all_true_hard = (all_true_labels > 0.5).astype(int)

    if len(np.unique(all_true_hard)) > 1:
        metrics['balanced_accuracy_all'] = balanced_accuracy_score(all_true_hard, (all_preds_prob > 0.5))
        metrics['roc_auc_all'] = roc_auc_score(all_true_hard, all_preds_prob)
    else:
        metrics['balanced_accuracy_all'] = np.nan
        metrics['roc_auc_all'] = np.nan

    # Always calculate regression metrics directly on the soft labels.
    metrics['r2_all'] = r2_score(all_true_labels, all_preds_prob)
    metrics['mse_all'] = mean_squared_error(all_true_labels, all_preds_prob)

    # --- 2. Calculate metrics for EXTREME trials ---
    # Initialize extreme metrics to NaN. They will be overwritten if calculable.
    metrics.update({
        "balanced_accuracy_extreme": np.nan, "roc_auc_extreme": np.nan,
        "r2_extreme": np.nan, "mse_extreme": np.nan
    })

    # Proceed only if the mask and original soft labels are provided and there are extreme trials.
    if is_extreme_mask is not None and original_soft_labels is not None and np.any(is_extreme_mask):
        extreme_indices = np.where(is_extreme_mask)[0]

        if len(extreme_indices) > 1:
            # Filter predictions and original soft labels using the mask
            extreme_preds_prob = all_preds_prob[extreme_indices]
            extreme_true_soft = original_soft_labels[extreme_indices]

            # Derive hard labels for the extreme subset from the *original soft labels*
            extreme_true_hard = (extreme_true_soft > 0.5).astype(int)

            if len(np.unique(extreme_true_hard)) > 1:
                metrics['balanced_accuracy_extreme'] = balanced_accuracy_score(extreme_true_hard, (extreme_preds_prob > 0.5))
                metrics['roc_auc_extreme'] = roc_auc_score(extreme_true_hard, extreme_preds_prob)

            # Always calculate regression metrics for the extreme subset
            metrics['r2_extreme'] = r2_score(extreme_true_soft, extreme_preds_prob)
            metrics['mse_extreme'] = mean_squared_error(extreme_true_soft, extreme_preds_prob)

    return metrics

# Evaluate Single Trial
def evaluate_single_trial(
    model: nn.Module,
    single_epoch_tensor: torch.Tensor,
    single_label_tensor: torch.Tensor,
    device: torch.device,
    output_logits: Optional[torch.Tensor] = None,
) -> Dict[str, any]:
    """Evaluate model on a single trial, returning prediction and true value."""
    if single_epoch_tensor.ndim == 2:
        single_epoch_tensor = single_epoch_tensor.unsqueeze(0)

    if output_logits is None:
        model.eval()
        with torch.no_grad():
            logits = model(single_epoch_tensor.to(device))
    else:
        logits = output_logits.to(device)

    pred_prob = torch.sigmoid(logits).item()
    true_label = single_label_tensor.item()
    
    # MSE is only relevant in regression mode
    loss = np.nan

    return {
        "true_label": true_label,
        "pred_prob": pred_prob,
        "loss": loss,
    }

def get_model_class(model_name: str) -> Type[nn.Module]:
    """Get model class from model name using central map."""
    model_class = MODEL_CLASS_MAP.get(model_name)
    if model_class is None:
        raise ValueError(
            f"Unknown model name: {model_name}. Available models: {list(MODEL_CLASS_MAP.keys())}"
        )
    return model_class


def filter_args_for_model(
    args_dict: Dict[str, Any], model_name: str, model_class: Type[nn.Module]
) -> Dict[str, Any]:
    """Filter arguments to only include those relevant for specific model constructor."""
    model_params = {}

    # Get expected parameters from model's __init__ signature
    try:
        init_signature = inspect.signature(model_class.__init__)
        expected_params = set(init_signature.parameters.keys())
        expected_params.discard("self")
    except ValueError:
        warnings.warn(
            f"Could not inspect signature for {model_name}. Cannot filter args precisely.",
            stacklevel=2,
        )
        expected_params = set()

    # Prefix-based autodiscovery
    prefix = PREFIX_MAP.get(model_name)
    if prefix is not None:
        prefix = prefix.lower()
        for arg_key, arg_val in args_dict.items():
            key_lower = arg_key.lower()
            if key_lower.startswith(prefix):
                stripped_param = arg_key[len(prefix):]
                if not stripped_param:
                    continue
                if not expected_params or stripped_param in expected_params:
                    model_params[stripped_param] = arg_val

    # Direct match
    for arg_key, arg_val in args_dict.items():
        if expected_params and arg_key in expected_params:
            if arg_key not in model_params:
                model_params[arg_key] = arg_val

    return model_params


def get_output_dir(
    base_output_root: Union[str, Path], experiment_name: str, timestamp: bool = True
) -> Path:
    """Create and return unique output directory for experiment run."""
    base_path = Path(base_output_root) / experiment_name
    if timestamp:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_path / timestamp_str
    else:
        run_dir = base_path

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_checkpoint_dir(run_output_dir: Path) -> Path:
    """Get checkpoint directory within run's output directory."""
    chkpt_dir = run_output_dir / "checkpoints"
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    return chkpt_dir


def save_checkpoint(state: dict, path: Union[str, Path]):
    """Save model and optimizer state dictionary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(state, path)
        log.info(f"Checkpoint saved to {path}")
    except Exception as e:
        log.error(f"Failed to save checkpoint to {path}: {e}")


def save_results_df(dataframe, path: Union[str, Path]):
    """Save pandas DataFrame to CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataframe.to_csv(path, index=False)
        log.info(f"Results DataFrame saved to {path}")
    except Exception as e:
        log.error(f"Failed to save results DataFrame to {path}: {e}")
