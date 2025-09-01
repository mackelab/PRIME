#%%
"""
Main script for conducting transfer learning experiments on EEG-TMS data using 
cross-subject k-fold cross-validation.

This script handles:
- Pre-training models on a set of source subjects.
- Evaluating models in a zero-shot manner on a target subject.
- Performing subject-specific calibration.
- Simulating online fine-tuning on a trial-by-trial basis.
- Aggregating and reporting performance metrics.
"""

# Core Python libraries
import argparse
import copy
import gc
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil
from collections import deque

# Third-party libraries for data handling and computation
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from omegaconf import ListConfig
from rich.table import Table


# Visualization and utility libraries
from rich.console import Console
from torchinfo import summary

# Local project-specific modules
from datasets import *
from TMS_EEG_moabb import TMSEEGDataset, TMSEEGClassification, TMSEEGClassificationTEP, TMSEEGDatasetTEP, TMSEEGDatasetTEPfree, TMSEEGClassificationTEPfree
from models.builder import build_model
from tta_wrapper import TTAWrapper, _apply_alignment_transform_np
from utils import (RegressionMetricsTracker, evaluate_single_trial,
                   evaluate_zero_shot, filter_args_for_model,get_checkpoint_dir, get_model_class,
                   get_output_dir, save_checkpoint, save_results_df)

# Optional dependencies for specific functionalities
try:
    import pyriemann
    PYRIEMANN_AVAILABLE = True
except ImportError:
    PYRIEMANN_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# --- Global Setup ---
# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("ERROR")

# Setup main logger
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# %% --- Data Handling ---

class DictDataset(Dataset):
    """A simple PyTorch Dataset that accepts tensors for epochs and labels."""
    def __init__(self, epochs_tensor: torch.Tensor, labels_tensor: torch.Tensor):
        if len(epochs_tensor) != len(labels_tensor):
            raise ValueError("Epochs and labels must have the same length.")
        self.epochs = epochs_tensor
        # Ensure labels are float for compatibility with regression and BCE loss
        self.labels = labels_tensor.float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {"epoch": self.epochs[index], "label": self.labels[index]}


def create_dataloader(epochs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle_data: bool = True) -> Optional[DataLoader]:
    """Creates a PyTorch DataLoader from NumPy arrays of epochs and labels.
    
    Args:
        epochs: EEG data with shape (n_trials, n_channels, n_times).
        labels: Target labels with shape (n_trials,).
        batch_size: The number of samples per batch.
        shuffle_data: Whether to shuffle the data at every epoch.

    Returns:
        A DataLoader instance or None if input data is invalid.
    """
    if epochs is None or labels is None or epochs.size == 0:
        return None
        
    epochs_tensor = torch.from_numpy(epochs).float()
    labels_tensor = torch.from_numpy(labels).float()
    
    dataset = DictDataset(epochs_tensor, labels_tensor)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=0,  # Set to 0 for main process loading, adjust if needed
        pin_memory=True
    )


# %% --- Core Training Functions ---

def pretrain_model(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   n_epochs: int, device: torch.device, args: OmegaConf,
                   run_name_suffix: str = "") -> nn.Module:
    """Pre-trains a model on a given dataset.
    """
    model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(n_epochs):
        total_epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Pre-train Epoch {epoch+1}/{n_epochs}", leave=False)
        
        for batch in pbar:
            X_batch = batch['epoch'].to(device)
            y_batch = batch['label'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_epoch_loss / len(train_loader)
        log.info(f"Pre-train Epoch {epoch+1}/{n_epochs} ({run_name_suffix}) | Avg Loss: {avg_loss:.4f}")

    return model



def train_finetuning_step(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                          device: torch.device, args: OmegaConf, trial_idx: int, wandb_run) -> Tuple[nn.Module, float]:
    """Performs a single fine-tuning step on a small window of recent trials.
    """
    if not loader:
        return model, 0.0
    
    criterion = nn.BCEWithLogitsLoss() 
  
    model.train()
    total_loss = 0.0
    
    for epoch in range(args.finetune_epochs):
        for batch in loader:
            x_batch = batch["epoch"].to(device, non_blocking=True)
            y_batch = batch["label"].to(device, non_blocking=True).unsqueeze(1)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch, is_finetuning_batch=True)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    avg_loss = total_loss / (len(loader) * args.finetune_epochs)
    
    # Optional W&B logging
    if WANDB_AVAILABLE and wandb.run:
        wandb.log({
            "finetune_step_loss": avg_loss,
            "finetune_lr": optimizer.param_groups[0]["lr"]
        }, step=trial_idx)
    
    return model, avg_loss

# %%
# Experiment setup

def setup_experiment(cli_args=None):
    """Parse arguments, setup configuration, and initialize experiment."""
    
    DEFAULT_YAML = """
dataset_names: ["TMSEEGClassification"]
subjects: null
data_root: "/mnt/lustre/home/macke/${oc.env:USER}/mne_data"
pretrained_checkpoint_dir: null
fmin: null
fmax: null
tmin: -0.55
tmax: -0.050
resample: null
models_to_run:
  - ShallowConvNet
  - DeepConvNet
  - EEGNetv4
  - ATCNet
  - PRIME
  - Ablation_NoS4
  - Ablation_ConvInsteadOfS4
  - Ablation_S4_WithConvClassifier
n_splits: 2
pretrain_epochs: 100
lr_pretrain: 0.0003
optimizer_type_pretrain: "AdamW"
weight_decay_pretrain: 0.0
batch_size_pretrain: 64
window_size: 50
finetune_epochs: 1
finetune_warmup_trials: 0
lr_finetune: 0.0001
optimizer_type_finetune: "AdamW" # Options: "Adam", "AdamW"
weight_decay_finetune: 0.0
batch_size_finetune: 50 
seed: 42
device: "cuda"
print_dataset_structure_and_exit: false
no_pretrain: false
base_output_dir: "results"
experiment_name: "transfer_kfold"
save_last_pretrained_checkpoint: false
save_last_finetuned_checkpoint: false
save_checkpoints: false
save_results: true
save_predictions_and_labels: true
use_wandb: false
wandb_project: "kfold_transfer"
wandb_group_prefix: "kfold_transfer"
wandb_run_description: "Default run"
num_pretrain_subjects: "max"
num_finetuning_subjects: null
num_trials_per_subject: null


# TTA Configuration
use_tta: true
ea_backrotation: true
alignment_type: "euclidean" # Options: "none", "euclidean", "riemannian" 
alignment_cov_epsilon: 1.0e-6
alignment_transform_epsilon: 1.0e-7
alignment_ref_ema_beta: 0.99
tta_cov_buffer_size: 50
use_adabn: false
finetune_mode: "full" # Options: "full", "decision_only", "none", "decision_criterion_only"

experiment_mode: "cross_subject_kfold"  

# channel_subset:
#   - C3
#   - FC1
#   - FC5
#   - CP1
#   - CP5
#   - C4
#   - FC2
#   - FC6
#   - CP2
#   - CP6

use_subject_specific_calibration: true 
num_calibration_trials: 100       # Number of initial trials from the test subject to use for calibration
lr_calibration: 0.0001           
calibration_epochs: 50           # Number of epochs to train on the small calibration set

shuffle_test_labels: false
"""
    
    # Load default config
    config = OmegaConf.create(DEFAULT_YAML)

    # Suppress verbose MOABB warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="moabb")
    warnings.filterwarnings("ignore", message="warnEpochs*")

    # Set MOABB loggers to ERROR level
    logging.getLogger("moabb").setLevel(logging.ERROR)
    logging.getLogger("moabb.paradigms").setLevel(logging.ERROR)
    logging.getLogger("moabb.datasets").setLevel(logging.ERROR)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="K-Fold Transfer with YAML Config")
    parser.add_argument("-c", "--config", action="append", help="Path to YAML config file(s)", default=[])
    parser.add_argument("--print-dataset-structure-and-exit", action="store_true")
    
    parsed_args, remaining_argv = parser.parse_known_args(args=cli_args)
    
    # Load and merge config files
    if parsed_args.config:
        for config_file in parsed_args.config:
            try:
                user_config = OmegaConf.load(config_file)
                config = OmegaConf.merge(config, user_config)
                print(f"Loaded config from: {config_file}")
            except Exception as e:
                print(f"Failed to load config {config_file}: {e}")
    
    # Apply CLI overrides
    if remaining_argv:
        try:
            cli_conf = OmegaConf.from_cli(remaining_argv)
            if cli_conf:
                config = OmegaConf.merge(config, cli_conf)
        except Exception as e:
            print(f"Failed to parse CLI overrides: {e}")
    
    if parsed_args.print_dataset_structure_and_exit:
        config.print_dataset_structure_and_exit = True
    
    OmegaConf.resolve(config)
    
    # Set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create output directory
    run_output_dir = get_output_dir(
        base_output_root=config.base_output_dir,
        experiment_name=config.experiment_name,
        timestamp=True,
    )
    console = Console()
    console.print(f"[blue]Output directory: {run_output_dir}[/blue]")
    
    # Save config
    try:
        config_save_path = run_output_dir / "config.yaml"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config_save_path)
    except Exception as e:
        print(f"Failed to save config: {e}")
    
    # Setup device
    device = torch.device(config.device)
    if config.device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available. Switching to CPU.[/yellow]")
        device = torch.device("cpu")
        config.device = "cpu"
    
    # Environment flags
    config.pyriemann_available = PYRIEMANN_AVAILABLE
    if config.use_wandb and not WANDB_AVAILABLE:
        console.print("[yellow]WandB not available. Disabling.[/yellow]")
        config.use_wandb = False
    config.wandb_available = WANDB_AVAILABLE
    
    # Process subjects and dataset names
    if config.subjects is not None:
        if isinstance(config.subjects, int):
            config.subjects = [config.subjects]
        elif isinstance(config.subjects, (list, ListConfig)):
            config.subjects = [int(s) for s in config.subjects]
    
    if not isinstance(config.dataset_names, (list, ListConfig)):
        if isinstance(config.dataset_names, str):
            config.dataset_names = [d.strip() for d in config.dataset_names.split(",") if d.strip()]
    config.dataset_names = [str(d) for d in config.dataset_names]
    
    return config, device, console, run_output_dir

def run_fold_pretraining(
    dataset_name: str,
    fold_idx: int,
    train_subject_ids: list,
    args: OmegaConf,
    device: torch.device,
    console: Console,
    run_output_dir: Path,
):
    """
    Load data and pretrain models for one fold using either soft or hard labels.
    """
    pretrained_models_fold = {}
    n_channels, n_timepoints = -1, -1
    n_outputs_model = 1  # Set to 1 for single-output regression/classification
    log_memory_usage(f"start_pretraining_fold_{fold_idx+1}", log)

    # --- Data Loading ---
    actual_pretrain_subject_ids = list(train_subject_ids)
    num_subjects_to_pretrain_on = getattr(args, "num_pretrain_subjects", "max")
    if isinstance(
        num_subjects_to_pretrain_on, int
    ) and num_subjects_to_pretrain_on < len(train_subject_ids):
        if num_subjects_to_pretrain_on > 0:
            rng = np.random.RandomState(args.seed + fold_idx)
            actual_pretrain_subject_ids = rng.choice(
                train_subject_ids, size=num_subjects_to_pretrain_on, replace=False
            ).tolist()
            console.print(
                f"  Sub-sampling: Using {len(actual_pretrain_subject_ids)} subjects for pretraining"
            )
    console.print(
        f"  Loading pretraining data for {len(actual_pretrain_subject_ids)} subjects..."
    )

    try:
        pretrain_epochs_data, pretrain_labels_data, global_backrot_matrix_np = None, None, None

        # The conditional logic has been removed. The following code now runs for ALL datasets.
        console.print(
            f"  [bold blue]Using generic data loader for '{dataset_name}'.[/bold blue]"
        )
        paradigm_kwargs = {
            "fmin": args.fmin,
            "fmax": args.fmax,
            "tmin": args.tmin,
            "tmax": args.tmax,
            "resample": args.resample,
        }
        if hasattr(args, "channel_subset") and args.channel_subset:
            paradigm_kwargs["channels"] = args.channel_subset
        
        # This call will now be used for TMSEEGClassification, etc., as well.
        pretrain_epochs_data, pretrain_labels_data, n_channels, n_timepoints, _, global_backrot_matrix_np = load_cached_pretrain_data(
            dataset_names=[dataset_name],
            subject_ids=actual_pretrain_subject_ids,
            paradigm_kwargs=paradigm_kwargs,
            data_root=args.data_root,
            args=args,
            apply_trial_ablation=True,
        )

        if pretrain_epochs_data is None or pretrain_epochs_data.size == 0:
            console.print(
                f"[yellow]No pretraining data loaded for fold {fold_idx+1}. Skipping.[/yellow]"
            )
            return {}, -1, -1, False

        X_train = pretrain_epochs_data
        y_train = pretrain_labels_data # Use a generic name now

        console.print(f"    Total pretrain trials: {len(X_train)}.")
        del pretrain_epochs_data, pretrain_labels_data
        gc.collect()

        # --- Create DataLoader ---
        # Pass the flag to ensure labels are formatted correctly by the dataloader
        pretrain_loader = create_dataloader(
            X_train, y_train, args.batch_size_pretrain, 
            shuffle_data=True       )
        
        if pretrain_loader is None or len(pretrain_loader) == 0:
            console.print(
                f"[yellow]Could not create a valid dataloader from the data. Skipping.[/yellow]"
            )
            return {}, -1, -1, False

        # --- Model Training Loop ---
        console.print(f"    Training models: {args.models_to_run}")
        base_args_dict = OmegaConf.to_container(args, resolve=True)
        printed_summaries = set()

        for model_idx, model_name in enumerate(args.models_to_run):
            console.print(
                f"      Model {model_idx+1}/{len(args.models_to_run)}: [bold yellow]{model_name}[/bold yellow]"
            )
            try:
                model_specific_args = filter_args_for_model(
                    base_args_dict, model_name, get_model_class(model_name)
                )
                model_pretrain = build_model(
                    model_name=model_name,
                    n_channels=n_channels,
                    n_times=n_timepoints,
                    n_outputs=n_outputs_model,
                    device=device,
                    model_specific_args=model_specific_args,
                    target_type="classification",
                )

                optimizer_params = {
                    "lr": args.lr_pretrain,
                    "weight_decay": args.weight_decay_pretrain,
                }
                if args.optimizer_type_pretrain.lower() == "adamw":
                    optimizer_pretrain = torch.optim.AdamW(
                        model_pretrain.parameters(), **optimizer_params
                    )
                else:
                    optimizer_pretrain = torch.optim.Adam(
                        model_pretrain.parameters(), **optimizer_params
                    )

                if (
                    (dataset_name, model_name) not in printed_summaries
                    and n_channels > 0
                    and n_timepoints > 0
                ):
                    try:
                        summary_str = summary(
                            model_pretrain,
                            input_size=(1, n_channels, n_timepoints),
                            verbose=0,
                        )
                        console.print(str(summary_str))
                        printed_summaries.add((dataset_name, model_name))
                    except Exception:
                        pass  # Fail silently if summary fails

                # The `pretrain_model` function will handle the loss logic internally
                model_pretrain = pretrain_model(
                    model=model_pretrain,
                    train_loader=pretrain_loader,
                    optimizer=optimizer_pretrain,
                    n_epochs=args.pretrain_epochs,
                    device=device,
                    args=args,
                    run_name_suffix=f"{dataset_name}_Fold_{fold_idx+1}_{model_name}",
                )


                pretrained_models_fold[model_name] = copy.deepcopy(
                    model_pretrain.state_dict()
                )

                # Save the final pretrained model checkpoint if the new flag is enabled
                if args.get("save_last_pretrained_checkpoint", False):
                    checkpoint_dir = get_checkpoint_dir(run_output_dir)
                    save_path = (
                        checkpoint_dir
                        / f"last_pretrained_{model_name}_ds_{dataset_name}_fold_{fold_idx+1}.pt"
                    )
                    save_checkpoint(
                        {"model_state_dict": model_pretrain.state_dict()}, save_path
                    )
                    console.print(
                        f"      [green]Saved last pretrained model checkpoint to {save_path.name}[/green]"
                    )

                # This is for saving intermediate checkpoints if needed by another flag
                if args.save_checkpoints:
                    checkpoint_dir = get_checkpoint_dir(run_output_dir)
                    save_path = (
                        checkpoint_dir
                        / f"model_{model_name}_ds_{dataset_name}_fold_{fold_idx+1}_pretrained.pt"
                    )
                    save_checkpoint(
                        {"model_state_dict": pretrained_models_fold[model_name]},
                        save_path,
                    )

            except Exception as e:
                log.error(
                    f"Error training {model_name} in fold {fold_idx+1}: {e}",
                    exc_info=True,
                )
                console.print(f"      [red]Failed to train {model_name}: {e}[/red]")
            finally:
                if "model_pretrain" in locals():
                    del model_pretrain
                if "optimizer_pretrain" in locals():
                    del optimizer_pretrain
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if 'global_backrot_matrix_np' in locals() and global_backrot_matrix_np is not None:
            backrot_matrix_path = run_output_dir / f"global_backrot_matrix_ds_{dataset_name}_fold_{fold_idx+1}.npy"
            try:
                np.save(backrot_matrix_path, global_backrot_matrix_np)
                console.print(f"    [green]Saved global back-rotation matrix to {backrot_matrix_path.name}[/green]")
            except Exception as e:
                console.print(f"    [red]Failed to save back-rotation matrix: {e}[/red]")

        if not pretrained_models_fold:
            console.print(
                f"[red]No models were successfully pretrained in fold {fold_idx+1}.[/red]"
            )
            return {}, -1, -1, False

        console.print(
            f"    [green]Successfully pretrained {len(pretrained_models_fold)} models for fold {fold_idx+1}.[/green]"
        )
        return pretrained_models_fold, n_channels, n_timepoints, True

    except Exception as e:
        log.error(f"Critical error in pretraining for fold {fold_idx+1}: {e}", exc_info=True)
        console.print(f"[red]Critical error in pretraining: {e}[/red]")
        return {}, -1, -1, False
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# %%
# Memory and pretraining utilities

def log_memory_usage(stage: str, log_obj=None):
    """Log current memory usage for debugging."""
    if log_obj:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        log_obj.debug(f"Memory usage at {stage}: {memory_mb:.1f} MB")

#%%
def run_online_finetuning_simulation(model, test_subj_epochs,
                                     labels_for_finetuning, labels_for_evaluation,
                                     is_extreme_mask, original_soft_labels,
                                     args, device, console, wandb_run,
                                     run_output_dir, model_name, dataset_name, subject_id, fold_idx, global_backrot_matrix_np: Optional[np.ndarray] = None):
    """
    Run an online finetuning simulation.
    """
    if test_subj_epochs is None or test_subj_epochs.size == 0:
        log.error(f"Empty test data provided for Subj {subject_id}. Skipping simulation.")
        return {}, 0.0, []

    n_trials_subj = test_subj_epochs.shape[0]
    log_prefix = f"{dataset_name}_Fold_{fold_idx}_Subj_{subject_id}_{model_name}"
    log.info(f"Starting online simulation for {log_prefix} ({n_trials_subj} trials)")

    # Initialize metrics tracker
    metrics_tracker = RegressionMetricsTracker(window_size=args.window_size)
    optimizer_finetune = None
    trial_times = []
    trial_metrics_log = []

    try:
        # --- Setup Optimizer and History Buffers ---
        if args.finetune_mode != 'none' and args.finetune_epochs > 0:
            optimizer_params = {"lr": args.lr_finetune, "weight_decay": args.weight_decay_finetune}
            optimizer_finetune = torch.optim.AdamW(model.parameters(), **optimizer_params)
            max_window_size = min(args.window_size, n_trials_subj)
            epoch_buffer = deque(maxlen=max_window_size)
            label_buffer = deque(maxlen=max_window_size)

        # --- Main Trial-by-Trial Loop ---
        online_iterator = tqdm(range(n_trials_subj), desc=f"Online Sim ({log_prefix})", leave=False)
        for trial_idx in online_iterator:
            trial_start_time = time.time()
            try:
                single_epoch_np = test_subj_epochs[trial_idx]
                single_label_for_finetuning_np = labels_for_finetuning[trial_idx]
                single_label_for_evaluation_np = labels_for_evaluation[trial_idx]
                
                single_epoch_t = torch.from_numpy(single_epoch_np).float().unsqueeze(0).to(device)
                single_label_for_eval_t = torch.tensor([single_label_for_evaluation_np], dtype=torch.float, device=device)

                model.eval()
                # --------------- PREDICT ---------------
                with torch.no_grad():
                    logits = model.predict(single_epoch_t)

                # The model's prediction is scored against the (potentially shuffled) evaluation label.
                eval_result = evaluate_single_trial(
                    model.wrapped_model, single_epoch_t, single_label_for_eval_t,
                    device, output_logits=logits
                )

                # --------------- ADAPT (unlabelled) ---------------
                if args.use_tta and args.alignment_type not in ['none', None]:
                    model.adapt_alignment(single_epoch_np)

                metrics_tracker.update(
                    y_true=eval_result["true_label"],
                    y_pred=eval_result["pred_prob"]
                )
                
                if optimizer_finetune:
                    epoch_buffer.append(single_epoch_np)
                    label_buffer.append(single_label_for_finetuning_np)

                step_loss = np.nan
                min_buffer_for_training = args.batch_size_finetune
                if (optimizer_finetune and len(epoch_buffer) >= min_buffer_for_training):
                    epochs_for_finetune = np.array(epoch_buffer)
                    labels_for_finetune_from_buffer = np.array(label_buffer)

                    if args.use_tta and args.alignment_type not in ['none', None]:
                        transform = model.alignment_transform_torch.cpu().numpy()
                        epochs_for_finetune = _apply_alignment_transform_np(epochs_for_finetune, transform)
                        
                        if global_backrot_matrix_np is not None:
                            epochs_for_finetune = _apply_alignment_transform_np(
                                epochs_for_finetune, global_backrot_matrix_np
                            )
                    
                    window_loader = create_dataloader(
                        epochs_for_finetune, labels_for_finetune_from_buffer,
                        batch_size=min(args.batch_size_finetune, len(epochs_for_finetune)), 
                        shuffle_data=True,
                    )
                    
                    if window_loader:
                        model.train()
                        _, step_loss = train_finetuning_step(
                            model=model, loader=window_loader, optimizer=optimizer_finetune,
                            device=device, args=args, trial_idx=trial_idx, wandb_run=wandb_run
                        )
                        model.eval()

                trial_times.append(time.time() - trial_start_time)
                
                online_iterator.set_postfix(
                    b_acc=f"{metrics_tracker.get_rolling_balanced_accuracy():.3f}", 
                    auc=f"{metrics_tracker.get_rolling_roc_auc():.3f}", 
                )

                trial_metrics_log.append({'trial_idx': trial_idx, 'rolling_balanced_accuracy': metrics_tracker.get_rolling_balanced_accuracy(), 'rolling_roc_auc': metrics_tracker.get_rolling_roc_auc(), 'overall_balanced_accuracy_at_trial': metrics_tracker.get_overall_balanced_accuracy(), 'overall_roc_auc_at_trial': metrics_tracker.get_overall_roc_auc(), 'finetune_loss': step_loss})
                if wandb_run:
                    wandb_run.log({f"online/{log_prefix}/rolling_b_acc": metrics_tracker.get_rolling_balanced_accuracy()}, step=trial_idx)
                    wandb_run.log({f"online/{log_prefix}/rolling_roc_auc": metrics_tracker.get_rolling_roc_auc()}, step=trial_idx)
            except Exception as e:
                log.error(f"Error processing trial {trial_idx} for {log_prefix}: {e}", exc_info=True)
                continue
        
        # --- Final Metrics Calculation ---
        avg_time_per_trial = np.mean(trial_times) if trial_times else 0.0
        y_true_all = np.array(metrics_tracker.all_y_true)
        y_pred_all = np.array(metrics_tracker.all_y_pred)
        
        final_metrics = {}
        
        # Always derive hard labels from soft for classification metrics
        y_true_hard_all = (y_true_all > 0.5).astype(int)
        if len(np.unique(y_true_hard_all)) > 1:
            final_metrics["balanced_accuracy_all"] = balanced_accuracy_score(y_true_hard_all, y_pred_all > 0.5)
            final_metrics["roc_auc_all"] = roc_auc_score(y_true_hard_all, y_pred_all)


        final_metrics.update({"balanced_accuracy_extreme": np.nan, "roc_auc_extreme": np.nan})
        
        if is_extreme_mask is not None and np.any(is_extreme_mask):
            extreme_indices = np.where(is_extreme_mask)[0]
            if len(extreme_indices) > 1:
                extreme_preds_soft = y_pred_all[extreme_indices]
                extreme_true_soft = original_soft_labels[extreme_indices]
                extreme_true_hard = (extreme_true_soft > 0.5).astype(int)
                
                if len(np.unique(extreme_true_hard)) > 1:
                    final_metrics["balanced_accuracy_extreme"] = balanced_accuracy_score(extreme_true_hard, (extreme_preds_soft > 0.5))
                    final_metrics["roc_auc_extreme"] = roc_auc_score(extreme_true_hard, extreme_preds_soft)
                

        log.info(f"Online sim finished. All (Bal Acc / ROC): {final_metrics.get('balanced_accuracy_all', np.nan):.4f} / {final_metrics.get('roc_auc_all', np.nan):.4f}.")

        if args.get('save_predictions_and_labels', False):
            output_filename = run_output_dir / f"predictions_{model_name}_ds_{dataset_name}_subj_{subject_id}_fold_{fold_idx}.npz"
            np.savez_compressed(output_filename, predictions=y_pred_all, actual_values=y_true_all)
            log.info(f"Saved predictions and actuals for Subj {subject_id} to {output_filename}")

        if wandb_run: wandb_run.log({f"final/{log_prefix}": final_metrics})

        return final_metrics, avg_time_per_trial, trial_metrics_log
        
    except Exception as e:
        log.error(f"Critical error in online simulation for {log_prefix}: {e}", exc_info=True)
        return {}, 0.0, []
    finally:
        if 'optimizer_finetune' in locals(): del optimizer_finetune
        if 'metrics_tracker' in locals(): del metrics_tracker
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# %%
def run_subject_evaluation(test_subject_id, fold_idx, pretrained_models_fold, n_channels, n_timepoints,
                           args, device, console, run_output_dir, no_pretrain, dataset_name):
    """
    Load data, run pre-calib eval, optional calibration, post-calib eval, and online finetuning.
    """
    subject_results = {model_name: {} for model_name in args.models_to_run}
    all_models_trial_metrics = {}
    console.print(f"  Processing Test Subject {test_subject_id} (Dataset: {dataset_name}, Fold {fold_idx+1})...")

    global_backrot_matrix_np = None
    if getattr(args, "ea_backrotation", False):
        backrot_matrix_path = run_output_dir / f"global_backrot_matrix_ds_{dataset_name}_fold_{fold_idx+1}.npy"
        if backrot_matrix_path.exists():
            global_backrot_matrix_np = np.load(backrot_matrix_path)
            console.print(f"        [green]Loaded global back-rotation matrix for evaluation.[/green]")
        else:
            console.print(f"        [yellow]Warning: Back-rotation is ON but matrix file was not found at {backrot_matrix_path}.[/yellow]")

    try:
        # --- Data Loading ---
        all_test_subj_epochs, all_test_subj_labels_soft = None, None

        # This logic correctly loads soft labels from all data sources
        if dataset_name in ["TMSEEGClassification", "TMSEEGClassificationTEP", "TMSEEGClassificationTEPfree"]:
            console.print(f"    [bold green]Using Custom TMS/TEP Paradigm for test subject data.[/bold green]")
            try:
                if dataset_name == "TMSEEGClassification":
                    dataset = TMSEEGDataset()
                    paradigm = TMSEEGClassification(tmin=args.tmin, tmax=args.tmax)
                elif dataset_name == "TMSEEGClassificationTEPfree":
                    dataset = TMSEEGDatasetTEPfree()
                    paradigm = TMSEEGClassificationTEPfree(tmin=args.tmin, tmax=args.tmax)
                else: # TMSEEGClassificationTEP
                    dataset = TMSEEGDatasetTEP()
                    paradigm = TMSEEGClassificationTEP(tmin=args.tmin, tmax=args.tmax)

                all_test_subj_epochs, all_test_subj_labels_soft, _ = paradigm.get_data(
                    dataset=dataset,
                    subjects=[test_subject_id]
                )
                
                if all_test_subj_epochs is not None and all_test_subj_epochs.size > 0:
                    actual_n_trials, actual_n_channels, actual_n_timepoints = all_test_subj_epochs.shape
                    if n_channels == -1 or n_timepoints == -1:
                        n_channels, n_timepoints = actual_n_channels, actual_n_timepoints
                    elif n_channels != actual_n_channels or n_timepoints != actual_n_timepoints:
                        console.print(f"    [bold red]Warning: Dimension mismatch! Expected ({n_channels}, {n_timepoints}), got ({actual_n_channels}, {actual_n_timepoints}). Using loaded dimensions.[/bold red]")
                        n_channels, n_timepoints = actual_n_channels, actual_n_timepoints
                        
            except Exception as e:
                log.error(f"Error loading custom data for test subject {test_subject_id}: {e}", exc_info=True)
        else:
            console.print(f"    [bold blue]Using generic data loader for test subject {test_subject_id}.[/bold blue]")
            all_test_subj_epochs, all_test_subj_labels_soft, n_channels_loaded, n_timepoints_loaded, _, _ = load_cached_pretrain_data(
                dataset_names=[dataset_name], subject_ids=[test_subject_id], 
                paradigm_kwargs={"fmin": args.fmin, "fmax": args.fmax, "resample": args.resample},
                data_root=args.data_root, args=args, apply_trial_ablation=False
            )
            
            if n_channels == -1 or n_timepoints == -1:
                n_channels, n_timepoints = n_channels_loaded, n_timepoints_loaded
        
        if all_test_subj_epochs is None or all_test_subj_epochs.size == 0:
            console.print(f"    [yellow]No valid data for subject {test_subject_id}. Skipping.[/yellow]")
            return subject_results, all_models_trial_metrics

        # --- Label Preparation ---
        is_extreme_mask = (all_test_subj_labels_soft <= 0.25) | (all_test_subj_labels_soft >= 0.75)
        all_test_subj_labels_ground_truth = all_test_subj_labels_soft
        
        if getattr(args, "shuffle_test_labels", False):
            console.print("[bold red]WARNING: SHUFFLING TEST SUBJECT LABELS FOR CONTROL ANALYSIS.[/bold red]")
            all_test_subj_labels_for_eval = shuffle(
                all_test_subj_labels_ground_truth.copy(),
                random_state=args.seed + test_subject_id
            )
        else:
            all_test_subj_labels_for_eval = all_test_subj_labels_ground_truth

        sr_hz_eval = next((p_data["specs"][dataset_name].get("sr") for p_name, p_data in PARADIGM_DATA.items() if dataset_name in p_data.get("specs", {})), None)
        
        for model_name in args.models_to_run:
            console.print(f"      Evaluating Model: [bold yellow]{model_name}[/bold yellow]")
            try:
                subject_results[model_name] = {"pre_calib_zero_shot": {}, "post_calib_zero_shot": {}, "finetuned": {}}
                
                model_eval = build_model(
                    model_name=model_name, n_channels=n_channels, n_times=n_timepoints, n_outputs=1,
                    device=device, model_specific_args=filter_args_for_model(OmegaConf.to_container(args, resolve=True), model_name, get_model_class(model_name))
                )
                model_eval_wrapped = TTAWrapper(model_eval, args, sr_hz=sr_hz_eval, global_backrot_matrix_np=global_backrot_matrix_np).to(device)

                if model_name in pretrained_models_fold:
                    model_eval_wrapped.wrapped_model.load_state_dict(pretrained_models_fold[model_name])
                    console.print("        Loaded generic pre-trained state.")

                # --- STAGE 1: PRE-CALIBRATION EVALUATION ---
                console.print(f"          Evaluating Pre-Calibration Zero-Shot Performance on all {len(all_test_subj_epochs)} trials...")
                pre_calib_metrics = evaluate_zero_shot(
                    model=model_eval_wrapped, test_epochs=all_test_subj_epochs,
                    test_labels=all_test_subj_labels_for_eval,
                    device=device, batch_size=args.batch_size_finetune,
                    is_extreme_mask=is_extreme_mask,
                    original_soft_labels=all_test_subj_labels_for_eval
                )
                subject_results[model_name]["pre_calib_zero_shot"] = pre_calib_metrics
                console.print(f"          [bold]Pre-Calib Bal. Acc / ROC AUC: {pre_calib_metrics.get('balanced_accuracy_all', np.nan):.4f} / {pre_calib_metrics.get('roc_auc_all', np.nan):.4f}[/bold]")

                # --- Split data for calibration and online phases ---
                if args.use_subject_specific_calibration and args.num_calibration_trials > 0 and len(all_test_subj_epochs) > args.num_calibration_trials:
                    calib_idx = args.num_calibration_trials
                    calibration_epochs, online_epochs = all_test_subj_epochs[:calib_idx], all_test_subj_epochs[calib_idx:]
                    calibration_labels_for_training = all_test_subj_labels_ground_truth[:calib_idx]
                    online_labels_for_finetuning = all_test_subj_labels_ground_truth[calib_idx:]
                    online_labels_for_eval = all_test_subj_labels_for_eval[calib_idx:]
                    online_is_extreme_mask = is_extreme_mask[calib_idx:]
                else:
                    online_epochs, online_labels_for_finetuning, online_labels_for_eval, online_is_extreme_mask = all_test_subj_epochs, all_test_subj_labels_ground_truth, all_test_subj_labels_for_eval, is_extreme_mask
                    calibration_epochs, calibration_labels_for_training = None, None

                # --- STAGE 2: CALIBRATION (FINE-TUNING) ---
                if calibration_epochs is not None and len(calibration_epochs) > 0:
                    console.print(f"        Starting subject-specific calibration for {args.calibration_epochs} epochs...")
                    
                    epochs_for_calib_loader = calibration_epochs
                    if args.use_tta and args.alignment_type not in ['none', None]:
                        model_eval_wrapped.init_alignment_from_calibration(calibration_epochs)
                        transform_matrix_np = model_eval_wrapped.alignment_transform_torch.cpu().numpy()
                        aligned_calibration_epochs = _apply_alignment_transform_np(calibration_epochs, transform_matrix_np)
                        if global_backrot_matrix_np is not None:
                            aligned_calibration_epochs = _apply_alignment_transform_np(aligned_calibration_epochs, global_backrot_matrix_np)
                        epochs_for_calib_loader = aligned_calibration_epochs

                    calib_loader = create_dataloader(
                        epochs_for_calib_loader,
                        calibration_labels_for_training,
                        batch_size=min(args.batch_size_finetune, len(calibration_epochs)),
                        shuffle_data=True
                    )

                    if calib_loader:
                        is_decision_only_mode = getattr(args, "finetune_mode", "full") == "decision_only"
                        try:
                            # If mode is 'decision_only', temporarily enable full updates for this block
                            if is_decision_only_mode:
                                console.print("        [bold yellow]Temporarily enabling full model update for calibration phase.[/bold yellow]")
                                model_eval_wrapped.enable_full_model_update(enabled=True)
                            
                            # This optimizer will now get ALL parameters if the override was enabled
                            optimizer_calib = torch.optim.AdamW(model_eval_wrapped.parameters(), lr=args.lr_calibration)
                            criterion = nn.BCEWithLogitsLoss() 
                            model_eval_wrapped.train()
                            for epoch in range(args.calibration_epochs):
                                pbar = tqdm(calib_loader, desc=f"Calib. Epoch {epoch+1}/{args.calibration_epochs}", leave=False)
                                for batch in pbar:
                                    X_batch, y_batch = batch['epoch'].to(device), batch['label'].to(device).unsqueeze(1)
                                    optimizer_calib.zero_grad()
                                    logits = model_eval_wrapped(X_batch)
                                    loss = criterion(logits, y_batch) 
                                    optimizer_calib.step()
                                    pbar.set_postfix(loss=loss.item())
                            model_eval_wrapped.eval()
                            
                        finally:
                            # ALWAYS revert the setting after calibration, even if an error occurred
                            if is_decision_only_mode:
                                console.print("        [bold yellow]Reverting to 'decision_only' for online phase.[/bold yellow]")
                                model_eval_wrapped.enable_full_model_update(enabled=False)
                        
                        console.print("        [green]Calibration complete.[/green]")

                # --- STAGE 3: ONLINE EVALUATION ---
                if online_epochs is not None and len(online_epochs) > 0:
                    console.print(f"          Evaluating Post-Calibration Zero-Shot Performance on {len(online_epochs)} trials...")
                    post_calib_metrics = evaluate_zero_shot(
                        model=model_eval_wrapped, test_epochs=online_epochs,
                        test_labels=online_labels_for_eval,
                        device=device, batch_size=args.batch_size_finetune,
                        is_extreme_mask=online_is_extreme_mask,
                        original_soft_labels=online_labels_for_eval
                    )
                    subject_results[model_name]["post_calib_zero_shot"] = post_calib_metrics
                    console.print(f"          [bold]Post-Calib Bal. Acc / ROC AUC: {post_calib_metrics.get('balanced_accuracy_all', np.nan):.4f} / {post_calib_metrics.get('roc_auc_all', np.nan):.4f}[/bold]")
                    
                    console.print(f"        Starting online finetuning simulation on the remaining {len(online_epochs)} trials...")
                    final_finetuned_metrics, _, per_trial_metrics = run_online_finetuning_simulation(
                        model=model_eval_wrapped, 
                        test_subj_epochs=online_epochs, 
                        labels_for_finetuning=online_labels_for_finetuning,
                        labels_for_evaluation=online_labels_for_eval,
                        is_extreme_mask=online_is_extreme_mask,
                        original_soft_labels=online_labels_for_eval,
                        args=args, device=device, console=console, wandb_run=None,
                        run_output_dir=run_output_dir, model_name=model_name, dataset_name=dataset_name,
                        subject_id=test_subject_id, fold_idx=fold_idx + 1,
                        global_backrot_matrix_np=global_backrot_matrix_np
                    )
                    subject_results[model_name]["finetuned"] = final_finetuned_metrics
                    all_models_trial_metrics[model_name] = per_trial_metrics

                    # Save the final fine-tuned model state for interpretability analysis
                    if args.get('save_last_finetuned_checkpoint', False):
                        checkpoint_dir = get_checkpoint_dir(run_output_dir)
                        save_path = checkpoint_dir / f"finetuned_{model_name}_ds_{dataset_name}_subj_{test_subject_id}_fold_{fold_idx+1}.pt"
                        
                        # We save the state_dict of the entire TTAWrapper to include alignment info
                        checkpoint_data = {
                            'model_state_dict': model_eval_wrapped.state_dict(),
                        }
                        
                        save_checkpoint(checkpoint_data, save_path)
                        console.print(f"        [green]Saved fine-tuned model for interpretability to {save_path.name}[/green]")

                else:
                    console.print("        No data available for the online phase. Final results will be empty.")
            except Exception as e:
                log.error(f"Error processing model {model_name} for subject {test_subject_id}: {e}", exc_info=True)
    except Exception as e:
        log.error(f"Critical error processing subject {test_subject_id}: {e}", exc_info=True)
    
    return subject_results, all_models_trial_metrics

# %% --- Experiment Execution and Result Aggregation ---

def run_cross_subject_experiment(
    args: OmegaConf, device: torch.device,
    console: Console, run_output_dir: Path
) -> Tuple[dict, list, list]:
    """Manages the cross-subject k-fold cross-validation experiment.
    """
    console.print("\n[bold magenta]===== Starting Cross-Subject K-Fold Experiment =====[/bold magenta]")
    all_datasets_results = {}
    processed_dataset_names = []
    all_trial_metrics = []

    for dataset_name in args.dataset_names:
        console.print(f"\n[bold cyan]### Dataset: {dataset_name} ###[/bold cyan]")
        
        # --- Subject Selection and Splitting ---
        all_subjects = get_subject_list_for_datasets([dataset_name], args.data_root)
        subjects_to_run = [s for s in all_subjects if s in args.subjects] if args.subjects else all_subjects

        if len(subjects_to_run) < args.n_splits:
            log.warning(f"Insufficient subjects ({len(subjects_to_run)}) for {args.n_splits} splits. Skipping dataset.")
            continue
        
        processed_dataset_names.append(dataset_name)
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        results_current_dataset = {m: {f: {} for f in range(args.n_splits)} for m in args.models_to_run}
        
        # --- K-Fold Cross-Validation Loop ---
        for fold_idx, (train_indices, test_indices) in enumerate(kf.split(subjects_to_run)):
            train_subject_ids = [subjects_to_run[i] for i in train_indices]
            test_subject_ids = [subjects_to_run[i] for i in test_indices]
            console.print(f"\n  [bold blue]=> Fold {fold_idx+1}/{args.n_splits} | Train: {train_subject_ids} | Test: {test_subject_ids}[/bold blue]")
            
            try:
                # --- Model Preparation for the Fold ---
                # This block decides whether to train models, load them, or just get data dimensions.
                pretrain_success = False
                fold_pretrained_models, n_channels, n_timepoints = {}, -1, -1

                if args.pretrained_checkpoint_dir:
                    # Option 1: Load pre-trained models from a specified directory.
                    log.info(f"Loading pre-trained models from: {args.pretrained_checkpoint_dir}")
                    # First, get data dimensions from a sample test subject.
                    _, _, n_channels, n_timepoints, _, _ = load_cached_pretrain_data(
                        dataset_names=[dataset_name], subject_ids=[test_subject_ids[0]],
                        paradigm_kwargs={"fmin": args.fmin, "fmax": args.fmax, "resample": args.resample},
                        data_root=args.data_root, args=args
                    )
                    for model_name in args.models_to_run:
                        chkpt_path = Path(args.pretrained_checkpoint_dir) / f"last_pretrained_{model_name}_ds_{dataset_name}_fold_{fold_idx+1}.pt"
                        if chkpt_path.is_file():
                            state_dict = torch.load(chkpt_path, map_location='cpu')['model_state_dict']
                            fold_pretrained_models[model_name] = state_dict
                    pretrain_success = bool(fold_pretrained_models)

                elif not args.no_pretrain:
                    # Option 2: Run pre-training from scratch using the fold's training subjects.
                    fold_pretrained_models, n_channels, n_timepoints, pretrain_success = run_fold_pretraining(
                        dataset_name, fold_idx, train_subject_ids, args, device, console, run_output_dir
                    )
                else:
                    # Option 3: No pre-training. Initialize models with random weights.
                    # We still need to get data dimensions for model construction.
                    log.info("`no_pretrain` is True. Models will be randomly initialized.")
                    _, _, n_channels, n_timepoints, _, _ = load_cached_pretrain_data(
                        dataset_names=[dataset_name], subject_ids=[test_subject_ids[0]],
                        paradigm_kwargs={"fmin": args.fmin, "fmax": args.fmax, "resample": args.resample},
                        data_root=args.data_root, args=args
                    )
                    pretrain_success = n_channels > 0 and n_timepoints > 0

                if not pretrain_success:
                    log.error(f"Could not prepare models for fold {fold_idx+1}. Skipping.")
                    continue
                
                # --- Evaluate on each test subject in the fold ---
                for test_subject_id in test_subject_ids:
                    subject_results, subject_trial_metrics = run_subject_evaluation(
                        test_subject_id, fold_idx, fold_pretrained_models,
                        n_channels, n_timepoints, args, device, console, run_output_dir, args.no_pretrain, dataset_name
                    )

                    # Store results and trial metrics
                    for model_name, res_dict in subject_results.items():
                        results_current_dataset[model_name][fold_idx][test_subject_id] = res_dict
                    for model_name, trial_data in subject_trial_metrics.items():
                        for entry in trial_data:
                            entry.update({"dataset": dataset_name, "model": model_name, "fold": fold_idx + 1, "subject_id": test_subject_id})
                            all_trial_metrics.append(entry)

            except Exception as e:
                log.error(f"Error processing Fold {fold_idx+1}: {e}", exc_info=True)
        
        all_datasets_results[dataset_name] = results_current_dataset
        
    return all_datasets_results, processed_dataset_names, all_trial_metrics


def aggregate_and_report_results(results: dict, dataset_names: list, args: OmegaConf,
                                 console: Console, run_output_dir: Path):
    """Aggregates, averages, and reports all experimental results.
    """
    console.print("\n[bold magenta]===== Aggregated Results Summary =====[/bold magenta]")
    
    # Define the evaluation stages and metrics to be reported
    STAGES = {
        "pre_calib_zero_shot": "Pre-Calibration",
        "post_calib_zero_shot": "Post-Calibration",
        "finetuned": "Online Fine-tuned"
    }
    METRICS = {
        "balanced_accuracy_all": "Bal. Acc (All)", "roc_auc_all": "ROC AUC (All)",
        "balanced_accuracy_extreme": "Bal. Acc (Extreme)", "roc_auc_extreme": "ROC AUC (Extreme)",
    }

    # --- Data Aggregation ---
    # Average results across folds for each subject
    final_subject_metrics = {}
    for ds_name, ds_results in results.items():
        final_subject_metrics[ds_name] = {m: {} for m in ds_results}
        for model_name, model_results in ds_results.items():
            # Collect all results for each subject across all folds
            subj_fold_data = {}
            for fold_data in model_results.values():
                for subj_id, metric_dict in fold_data.items():
                    if subj_id not in subj_fold_data:
                        subj_fold_data[subj_id] = {stage: [] for stage in STAGES}
                    for stage_key, stage_metrics in metric_dict.items():
                        subj_fold_data[subj_id][stage_key].append(stage_metrics)
            
            # Average the metrics
            for subj_id, collected_metrics in subj_fold_data.items():
                final_subject_metrics[ds_name][model_name][subj_id] = {}
                for stage_key, stage_list in collected_metrics.items():
                    df = pd.DataFrame(stage_list)
                    final_subject_metrics[ds_name][model_name][subj_id][stage_key] = df.mean().to_dict()

    # --- Reporting ---
    csv_rows = []
    for ds_name in dataset_names:
        console.print(f"\n[bold green]### Results for Dataset: {ds_name} ###[/bold green]")
        ds_metrics = final_subject_metrics.get(ds_name, {})
        all_subjects = sorted({sid for res in ds_metrics.values() for sid in res})

        for stage_key, stage_title in STAGES.items():
            console.print(f"\n[bold]--- {stage_title} Performance ---[/bold]")
            for metric_key, metric_title in METRICS.items():
                # Check if there is any data for this metric to avoid empty tables
                has_data = any(metric_key in (ds_metrics.get(m, {}).get(s, {}).get(stage_key, {})) for m in ds_metrics for s in all_subjects)
                if not has_data: continue

                table = Table(title=f"{metric_title} - {ds_name}")
                table.add_column("Model", style="cyan")
                for sid in all_subjects: table.add_column(f"Subj {sid}", justify="right")
                table.add_column("Mean", style="green", justify="right")

                for model_name, model_data in ds_metrics.items():
                    scores = [model_data.get(sid, {}).get(stage_key, {}).get(metric_key, np.nan) for sid in all_subjects]
                    row = [model_name] + [f"{s:.4f}" if not np.isnan(s) else "N/A" for s in scores]
                    row.append(f"[bold]{np.nanmean(scores):.4f}[/bold]")
                    table.add_row(*row)
                console.print(table)
        
        # Collect data for CSV export
        for model_name, model_data in ds_metrics.items():
            for subject_id, subject_data in model_data.items():
                row = {"dataset": ds_name, "model": model_name, "subject_id": subject_id}
                for stage_key, stage_metrics in subject_data.items():
                    for metric_key, val in stage_metrics.items():
                        row[f"{stage_key}_{metric_key}"] = val
                csv_rows.append(row)

    # --- Save to CSV ---
    if args.save_results and csv_rows:
        results_df = pd.DataFrame(csv_rows)
        results_df.to_csv(run_output_dir / "results_summary.csv", index=False)
        log.info(f"Aggregated results summary saved to {run_output_dir / 'results_summary.csv'}")


if __name__ == "__main__":
    # 1. Initialize experiment: parse args, create directories, set seed
    log = logging.getLogger(__name__)
    try:
        args, device, console, run_output_dir = setup_experiment()
        # Setup file logging to save run details
        file_handler = logging.FileHandler(run_output_dir / "run.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)
        log.info("Final Configuration:\n%s", OmegaConf.to_yaml(args))
    except Exception as e:
        log.error(f"Experiment setup failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Run the main experiment logic
    try:
        if args.experiment_mode == "cross_subject_kfold":
            all_results, processed_datasets, trial_metrics = run_cross_subject_experiment(
                args, device, console, run_output_dir
            )
        else:
            raise ValueError(f"Unknown experiment_mode: '{args.experiment_mode}'")
        
        # 3. Aggregate and report results
        if not all_results:
            log.warning("Experiment finished but no results were generated.")
        else:
            aggregate_and_report_results(all_results, processed_datasets, args, console, run_output_dir)
        
        # Save trial-by-trial metrics
        if args.save_results and trial_metrics:
            trial_df = pd.DataFrame(trial_metrics)
            trial_df.to_csv(run_output_dir / "results_trial_metrics.csv", index=False)
            log.info(f"Trial-by-trial metrics saved to {run_output_dir / 'results_trial_metrics.csv'}")
            
        console.print("\n[bold green]✅ Experiment Complete.[/bold green]")
        log.info("Experiment finished successfully.")

    except KeyboardInterrupt:
        log.warning("Experiment interrupted by user (Ctrl+C).")
    except Exception as e:
        log.critical(f"A critical error occurred during the main execution: {e}", exc_info=True)
    finally:
        # 4. Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory_usage("final_cleanup", log)
        console.print("\n[dim]Cleanup complete.[/dim]")

