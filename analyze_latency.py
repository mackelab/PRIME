# %%
"""
================================================================================
Latency Measurement Script for PRIME

The script iterates through a user-defined set of datasets, subjects, models,
and computational devices (CPU/GPU). For each combination, it measures:
1.  **Prediction Latency**: The time taken for a single forward pass, including
    Test-Time Augmentation (TTA) if enabled.
2.  **TTA Statistics Update Latency**: The time required to update the internal
    statistics for TTA alignment.
3.  **Fine-tuning Step Latency**: The time for a single supervised fine-tuning
    step, including forward pass, loss calculation, backward pass, and
    optimizer step.
"""

# 1. IMPORTS
import itertools
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Third-party libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    pass

# Project-specific imports with error handling
try:
    # This assumes the project root is the parent directory of this script's location
    # and has been added to the Python path.
    from TMS_EEG_moabb import (
        TMSEEGClassification,
        TMSEEGClassificationTEPfree,
        TMSEEGDataset,
        TMSEEGDatasetTEP
    )
    from datasets import PARADIGM_DATA, load_cached_pretrain_data
    from utils import filter_args_for_model, get_model_class
    from models.builder import build_model
    from tta_wrapper import TTAWrapper
    PROJECT_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: A required project module could not be imported: {e}", file=sys.stderr)
    print("Please ensure that the project root is in your PYTHONPATH and all required files are present.", file=sys.stderr)
    PROJECT_MODULES_AVAILABLE = False


# 2. CONFIGURATION
def get_config() -> DictConfig:
    """
    Returns the main configuration for the latency experiment.
    """
    conf = {
        # --- Experiment Setup ---
        "experiment_name": "latency_analysis_merged",
        "base_output_dir": "results_latency_merged",
        "seed": 42,

        # --- Models and Datasets to Run ---
        "models_to_run": ["PRIME"],
        "datasets_by_paradigm": {
            "TMS": ["TMSEEGClassificationTEPfree"],
        },

        # --- Data Loading Parameters ---
        "data_root": "/mnt/lustre/home/macke/${oc.env:USER}/mne_data",
        "tmin": -0.055,  # Relevant for TMS paradigms
        "tmax": -0.005,   # Relevant for TMS paradigms
        "fmin": None,     # Relevant for other paradigms (e.g., MI)
        "fmax": None,     # Relevant for other paradigms (e.g., MI)
        "resample": None, # Resampling frequency in Hz

        # --- Latency Measurement Settings ---
        "n_repeats_inference": 100,      # Number of loops for inference latency
        "n_finetune_steps_to_measure": 50, # Number of fine-tuning steps to measure
        "batch_size_finetune": 50,         # Batch size for fine-tuning latency test

        # --- Model and TTA Configuration ---
        "use_tta": False,
        "alignment_type": "none",
        "alignment_cov_epsilon": 1.0e-6,
        "alignment_transform_epsilon": 1.0e-7,
        "alignment_ref_ema_beta": 0.9,
        "use_adabn": True,
        "finetune_mode": "full",

    }
    return OmegaConf.create(conf)


# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================
_cuda_available: Optional[bool] = None

def is_cuda_available() -> bool:
    """Checks for CUDA availability and caches the result."""
    global _cuda_available
    if _cuda_available is None:
        try:
            import torch
            _cuda_available = torch.cuda.is_available()
        except (ImportError, Exception) as e:
            print(f"Warning: Could not check for CUDA. Assuming False. Error: {e}", file=sys.stderr)
            _cuda_available = False
    return _cuda_available

def get_subject_list(dataset_name: str) -> List[int]:
    """
    Retrieves the list of available subject IDs for a given dataset.
    """
    print(f"Fetching subject list for dataset: {dataset_name}")
    if not PROJECT_MODULES_AVAILABLE:
        print("  Warning: Project modules not found. Defaulting to subject [1].", file=sys.stderr)
        return [1]

    try:
        if dataset_name == "TMSEEGClassificationTEPfree":
            dataset_instance = TMSEEGDatasetTEP()
            subjects = dataset_instance.subject_list
        else:
            print(f"  Warning: Subject list retrieval not implemented for '{dataset_name}'. Defaulting to subject [1].")
            return [1]

        print(f"  Found {len(subjects)} subjects: {subjects}")
        return subjects

    except Exception as e:
        print(f"  Error getting subject list for {dataset_name}: {e}. Defaulting to [1].", file=sys.stderr)
        return [1]

class DictDataset(Dataset):
    """A simple dataset wrapper for dictionary-based item access."""
    def __init__(self, epochs: torch.Tensor, labels: torch.Tensor):
        if len(epochs) != len(labels):
            raise ValueError("Epochs and labels must have the same length.")
        self.epochs = epochs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {"epoch": self.epochs[index], "label": self.labels[index]}

# 4. DATA AND MODEL HANDLING
def create_dataloader(
    epochs: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    cfg: DictConfig
) -> Optional[DataLoader]:
    """Creates a DataLoader from numpy arrays of epochs and labels."""
    if epochs is None or labels is None or epochs.size == 0:
        return None

    epochs_tensor = torch.from_numpy(epochs).float()
    # Convert soft labels (probabilities) to hard binary labels
    hard_labels = (labels > 0.5).astype(np.int64)
    labels_tensor = torch.from_numpy(hard_labels)

    dataset = DictDataset(epochs_tensor, labels_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle for realistic fine-tuning steps
        num_workers=0,
        pin_memory=True
    )

def load_latency_data(
    dataset_name: str,
    subject_id: int,
    cfg: DictConfig,
    console: Console
) -> Tuple:
    """Loads data for a specific dataset and subject for latency testing."""
    effective_target_type = 'regression' if "TMSEEG" in dataset_name else 'classification'
    console.print(f"  Loading data for {dataset_name} (Subject {subject_id}, Task: {effective_target_type})...")

    if not PROJECT_MODULES_AVAILABLE:
        console.print(f"  [red]Error: Cannot load {dataset_name} because project modules are not available.[/red]")
        return (None,) * 8

    epochs_data, labels_data, n_ch, n_t, sr_hz, n_out = None, None, 0, 0, 250, 1

    try:
        # --- Path 1: Custom TMS/EEG Datasets ---
        if "TMSEEG" in dataset_name:
            console.print("    [green]Using custom TMS paradigm loader.[/green]")
            if dataset_name == "TMSEEGClassificationTEPfree":
                dataset = TMSEEGDatasetTEP()
                paradigm = TMSEEGClassificationTEPfree(tmin=cfg.tmin, tmax=cfg.tmax)
            else:
                raise NotImplementedError(f"Loading for TMS dataset '{dataset_name}' is not implemented.")

            epochs_data, labels_data, _ = paradigm.get_data(dataset=dataset, subjects=[subject_id])
            if epochs_data.size > 0:
                _, n_ch, n_t = epochs_data.shape

        # --- Path 2: Generic MOABB Datasets ---
        else:
            console.print("    [blue]Using generic MOABB loader.[/blue]")
            epochs_data, labels_data, n_ch, n_t, _, _ = load_cached_pretrain_data(
                dataset_names=[dataset_name],
                subject_ids=[subject_id],
                paradigm_kwargs={"fmin": cfg.fmin, "fmax": cfg.fmax, "resample": cfg.resample},
                data_root=cfg.data_root, args=cfg,
                verbose=False, target_type=effective_target_type, apply_trial_ablation=False
            )

        if epochs_data is None or epochs_data.size == 0:
            raise ValueError("Data loading returned no epochs.")

        # --- Determine sampling rate and number of outputs ---
        for p_data in PARADIGM_DATA.values():
            if dataset_name in p_data.get("specs", {}):
                spec = p_data["specs"][dataset_name]
                sr_hz = spec.get("sr", sr_hz)
                n_out = 1 if effective_target_type == 'regression' else spec.get("n_cls", len(np.unique(labels_data)))
                break
        effective_sr_hz = cfg.resample if cfg.resample else sr_hz

        console.print(f"    Data Loaded: {epochs_data.shape[0]} trials, {n_ch}Ch, {n_t}T, SR={effective_sr_hz}Hz")

        # --- Create DataLoader and a single input tensor for tests ---
        ft_dataloader = create_dataloader(epochs_data, labels_data, cfg.batch_size_finetune, cfg)
        single_input_tensor = torch.from_numpy(epochs_data[0:1]).float() if epochs_data.shape[0] > 0 else None

        return (
            ft_dataloader, single_input_tensor, n_ch, n_t, n_out,
            effective_sr_hz, effective_target_type, None # No error message
        )

    except Exception as e:
        console.print(f"  [red]Error loading data for {dataset_name} (Subj {subject_id}): {e}[/red]")
        logging.error(f"Data loading failed for {dataset_name}, Subj {subject_id}", exc_info=True)
        return (None,) * 7 + (str(e),)

def build_latency_model(
    model_name: str,
    n_channels: int,
    n_timepoints: int,
    n_outputs: int,
    sr_hz: float,
    device: torch.device,
    cfg: DictConfig,
    target_type: str
) -> Tuple[Optional[TTAWrapper], int, int]:
    """Builds the model and wraps it with TTAWrapper."""
    logging.info(f"Building {model_name}: C={n_channels}, T={n_timepoints}, N_out={n_outputs}, SR={sr_hz}Hz, Task={target_type}")
    if n_outputs == 0:
        logging.warning(f"Correcting n_outputs from 0 to 1 for model {model_name}.")
        n_outputs = 1

    if not PROJECT_MODULES_AVAILABLE:
        logging.error("Cannot build model: project modules are not available.")
        return None, 0, 0

    ModelClass = get_model_class(model_name)
    model_specific_args = filter_args_for_model(cfg, model_name, ModelClass)

    base_model = build_model(
        model_name=model_name, n_channels=n_channels, n_times=n_timepoints,
        n_outputs=n_outputs, device=device,
        model_specific_args=model_specific_args,
        target_type=target_type
    )

    wrapped_model = TTAWrapper(base_model, cfg, sr_hz=sr_hz)
    wrapped_model.to(device)

    total_params = sum(p.numel() for p in wrapped_model.parameters())
    trainable_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    logging.info(f"  {model_name} (wrapped): Total={total_params:,}, Trainable={trainable_params:,}")

    return wrapped_model, total_params, trainable_params


# 5. LATENCY MEASUREMENT FUNCTIONS
def measure_latency(
    func: callable,
    n_repeats: int,
    warmup_reps: int = 10
) -> Tuple[float, ...]:
    """A generic utility to measure the execution time of a function."""
    # Warm-up phase
    for _ in range(warmup_reps):
        func()

    # Measurement phase
    latencies = []
    for _ in range(n_repeats):
        start_time = time.perf_counter()
        func()
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # Convert to ms

    if not latencies:
        return (np.nan,) * 6

    return (
        np.mean(latencies), np.std(latencies), np.median(latencies),
        np.min(latencies), np.max(latencies), np.percentile(latencies, 99)
    )

def measure_single_prediction_latency(
    model: TTAWrapper,
    input_tensor: torch.Tensor,
    device: torch.device,
    cfg: DictConfig
) -> Tuple[float, ...]:
    """Measures the latency of a single TTA-enabled forward pass."""
    model.eval()
    input_on_device = input_tensor.to(device)

    def pred_step():
        with torch.no_grad():
            _ = model(input_on_device, apply_tta=cfg.use_tta, is_finetuning_batch=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    return measure_latency(pred_step, n_repeats=cfg.n_repeats_inference)

def measure_tta_update_latency(
    model: TTAWrapper,
    input_tensor: torch.Tensor,
    device: torch.device,
    cfg: DictConfig
) -> Tuple[float, ...]:
    """Measures the latency of updating TTA alignment statistics."""
    if not hasattr(model, 'update_tta_statistics'):
        logging.error("Model does not have 'update_tta_statistics' method.")
        return (np.nan,) * 6

    model.eval()
    input_on_device = input_tensor.to(device)

    def update_step():
        model.update_tta_statistics(input_on_device)
        if device.type == "cuda":
            torch.cuda.synchronize()

    return measure_latency(update_step, n_repeats=cfg.n_repeats_inference)

def measure_finetuning_step_latency(
    model: TTAWrapper,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: DictConfig
) -> Tuple[float, ...]:
    """Measures the latency of a full fine-tuning step."""
    model.train()
    data_iterator = iter(dataloader)
    latencies = []
    
    criterion_bce = nn.BCEWithLogitsLoss() 

    # Warm-up phase
    for _ in range(min(10, len(dataloader))):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)
        inputs = batch["epoch"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs, is_finetuning_batch=True)
        loss = criterion_bce(outputs, labels.float())
        loss.backward()
        optimizer.step()
        if device.type == "cuda": torch.cuda.synchronize()

    # Measurement phase
    for _ in range(cfg.n_finetune_steps_to_measure):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)
            
        inputs = batch["epoch"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs, is_finetuning_batch=True)
        loss = criterion_bce(outputs, labels.float())
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        latencies.append((end_time - start_time) * 1000)

    if not latencies: return (np.nan,) * 6

    return (
        np.mean(latencies), np.std(latencies), np.median(latencies),
        np.min(latencies), np.max(latencies), np.percentile(latencies, 99)
    )

# 6. MAIN EXECUTION
def main():
    """Main function to orchestrate and run the latency experiments."""
    # --- Initial Setup ---
    console = Console()
    cfg = get_config()
    OmegaConf.resolve(cfg) # Resolve any interpolations like ${oc.env:USER}

    # Setup output directory
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(cfg.base_output_dir) / f"{cfg.experiment_name}_{ts}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = run_output_dir / "latency_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logging.info(f"Run output directory: {run_output_dir.resolve()}")
    logging.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Check for project modules
    if not PROJECT_MODULES_AVAILABLE:
        logging.error("Essential project modules are missing. Aborting run.")
        return

    # Set seed for reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if is_cuda_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # --- Build Experiment Combinations ---
    all_datasets = sorted(list(set(ds for d_list in cfg.datasets_by_paradigm.values() for ds in d_list)))
    devices = ["cpu", "cuda"] if is_cuda_available() else ["cpu"]
    all_jobs = []
    for dataset in all_datasets:
        subjects = get_subject_list(dataset)
        combinations = list(itertools.product([dataset], subjects, cfg.models_to_run, devices))
        all_jobs.extend(combinations)

    logging.info(f"Target Models: {cfg.models_to_run}")
    logging.info(f"Target Datasets: {all_datasets}")
    logging.info(f"Target Devices: {devices}")
    logging.info(f"Total jobs to run: {len(all_jobs)}")

    # --- Run Experiments ---
    all_results = []
    job_pbar = tqdm(all_jobs, desc="Overall Progress")
    for i, (dataset_name, subject_id, model_name, device_str) in enumerate(job_pbar):
        job_num = i + 1
        job_desc = f"Job {job_num}/{len(all_jobs)}: {dataset_name}-S{subject_id}-{model_name}-{device_str}"
        job_pbar.set_description(job_desc)
        console.print(f"\n--- Running {job_desc} ---", style="bold yellow")

        device = torch.device(device_str)
        model_wrapped, error_msg = None, np.nan
        job_result = {
            "dataset": dataset_name, "subject_id": subject_id,
            "model": model_name, "device": device_str, "error": error_msg
        }

        try:
            # 1. Load Data
            (ft_loader, single_tensor, n_ch, n_t, n_out, sr, target_type, err) = load_latency_data(
                dataset_name, subject_id, cfg, console
            )
            if err: raise RuntimeError(f"Data loading failed: {err}")

            job_result.update({
                "n_channels": n_ch, "n_timepoints": n_t, "sr_hz": sr,
                "target_type": target_type
            })

            # 2. Build Model
            model_wrapped, total_p, trainable_p = build_latency_model(
                model_name, n_ch, n_t, n_out, sr, device, cfg, target_type
            )
            if model_wrapped is None: raise RuntimeError("Model building failed.")
            job_result.update({"total_params": total_p, "trainable_params": trainable_p})

            # 3. Measure Latencies
            if single_tensor is not None:
                console.print("  Measuring single prediction latency...")
                (p_mean, p_std, p_med, p_min, p_max, p_p99) = measure_single_prediction_latency(model_wrapped, single_tensor, device, cfg)
                job_result.update({
                    "pred_ms_mean": p_mean, "pred_ms_std": p_std, "pred_ms_median": p_med,
                    "pred_ms_min": p_min, "pred_ms_max": p_max, "pred_ms_p99": p_p99
                })
                console.print(f"    - Median Prediction Latency: {p_med:.3f} ms")

                if cfg.use_tta and cfg.alignment_type != 'none':
                    console.print("  Measuring TTA update latency...")
                    (u_mean, u_std, u_med, u_min, u_max, u_p99) = measure_tta_update_latency(model_wrapped, single_tensor, device, cfg)
                    job_result.update({
                        "tta_update_ms_mean": u_mean, "tta_update_ms_std": u_std, "tta_update_ms_median": u_med,
                        "tta_update_ms_min": u_min, "tta_update_ms_max": u_max, "tta_update_ms_p99": u_p99
                    })
                    console.print(f"    - Median TTA Update Latency: {u_med:.3f} ms")

            if ft_loader:
                console.print("  Measuring fine-tuning step latency...")
                optimizer = torch.optim.AdamW(model_wrapped.parameters(), lr=1e-4)
                (f_mean, f_std, f_med, f_min, f_max, f_p99) = measure_finetuning_step_latency(
                    model_wrapped, ft_loader, optimizer, device, cfg
                )
                job_result.update({
                    "finetune_ms_mean": f_mean, "finetune_ms_std": f_std, "finetune_ms_median": f_med,
                    "finetune_ms_min": f_min, "finetune_ms_max": f_max, "finetune_ms_p99": f_p99
                })
                console.print(f"    - Median Fine-tune Step Latency: {f_med:.3f} ms")

        except Exception as e:
            error_msg = str(e)
            job_result["error"] = error_msg
            logging.error(f"Job failed: {job_desc}. Error: {e}", exc_info=True)
            console.print(f"[red]  ERROR for {job_desc}: {e}[/red]")
        finally:
            all_results.append(job_result)
            if model_wrapped: del model_wrapped
            if is_cuda_available(): torch.cuda.empty_cache()
            time.sleep(1) # Small delay between jobs

    # --- Finalize and Save Results ---
    console.print("\n[bold magenta]All jobs complete. Finalizing results...[/bold magenta]")
    if not all_results:
        console.print("[yellow]No results were generated.[/yellow]")
        return

    results_df = pd.DataFrame(all_results)
    
    # Display summary table in console
    summary_cols = [
        "dataset", "subject_id", "model", "device",
        "pred_ms_median", "finetune_ms_median", "tta_update_ms_median", "error"
    ]
    display_df = results_df[[c for c in summary_cols if c in results_df.columns]].copy()
    
    table = Table(title="Latency Measurement Summary", show_header=True, header_style="bold cyan")
    for col in display_df.columns:
        table.add_column(col, justify="left")
    for _, row in display_df.iterrows():
        table.add_row(*[f"{v:.3f}" if isinstance(v, float) and pd.notna(v) else str(v) for v in row])
    console.print(table)

    # Save detailed results to CSV
    output_csv_path = run_output_dir / "latency_results_detailed.csv"
    try:
        results_df.to_csv(output_csv_path, index=False, float_format='%.4f')
        console.print(f"\n[bold green]Successfully saved detailed results to:[/bold green] {output_csv_path}")
    except Exception as e:
        console.print(f"\n[bold red]Error saving results to CSV: {e}[/bold red]")

    console.print("\n[bold blue]Latency experiment finished.[/bold blue]")


if __name__ == "__main__":
    main()
# %%
