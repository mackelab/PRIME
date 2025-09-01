#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified script for running and aggregating interpretability analyses on TMS-EEG data.

This script combines job submission logic with the core interpretability analysis and plotting
functions. It is designed to be executed from the command line and supports two primary modes:
'submit' and 'aggregate'.

1. Submit Mode (`--mode submit`):
   - Scans a directory for pre-trained model checkpoints.
   - Groups subjects into chunks for parallel processing.
   - Submits jobs to a SLURM cluster using `submitit`.
   - Each job runs an interpretability analysis (e.g., occlusion) for a subset of subjects.
   - The choice of analysis (`occlusion`, `spatial_occlusion`, `frequency_occlusion`) is
     controlled by the `--method` argument.

2. Aggregate Mode (`--mode aggregate`):
   - Gathers the results from all completed jobs.
   - Computes population-level statistics (mean or median).
   - Performs statistical significance testing on the results.
   - Generates and saves plots of the aggregated importance maps.

Example usage:
# To submit jobs for spatio-spectral occlusion analysis:
python interpretability.py --mode submit --method occlusion

# After jobs complete, to aggregate the results:
python interpretability.py --mode aggregate --method occlusion --results_dir /path/to/results
"""

# 0. IMPORTS
import argparse
import logging
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import submitit
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import butter, filtfilt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import fdrcorrection

# Project-specific imports (ensure these are in your PYTHONPATH)
from models.deep_tepnet import PRIME
from tta_wrapper import TTAWrapper
from TMS_EEG_moabb import (TMSEEGClassificationTEPfree, TMSEEGDatasetTEPfree,
                           TMSEEGClassification, TMSEEGDataset,
                           TMSEEGClassificationTEP, TMSEEGDatasetTEP)


# 1. CONFIGURATION

# --- Basic Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Paths ---
USER = os.environ.get('USER', 'default_user')
REPO_DIR = Path(f"/mnt/lustre/work/macke/{USER}/repos/eegjepa")

# --- SLURM & JOB CONFIGURATION ---
SLURM_PARTITION = "a100-galvani"
MEM_GB_PER_JOB = 64
CPUS_PER_JOB = 8
GPUS_PER_JOB = 1
JOB_TIME = "0-02:00:00"
SUBJECTS_PER_JOB = 2

# --- ANALYSIS & DATA CONFIGURATION ---
BASE_LOG_DIR = REPO_DIR / "slurm_logs_interpretability"
CHECKPOINTS_DIR = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/results_final/10ms_pp_w50/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F/20250707_170031/checkpoints/")

ANALYSIS_CONFIG = {
    "method": "occlusion",  #available: occlusion, spatial_occlusion, frequency_occlusion
    "device": "cuda",
    "OCCLUSION_BATCH_SIZE": 64,
    "OCCLUSION_N_NEIGHBORS": 8,
    "N_CHANS": 60, "N_TIMES": 50, "N_OUTPUTS": 1, "SFREQ": 1000,
    "FILTER_TIME_LENGTH": 10, "T_MIN": -0.500, "T_MAX": -0.010,
    "FREQ_BANDS": {
        'Theta (4-8 Hz)': (4, 8), 'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-25 Hz)': (13, 25), 'Gamma (25-47 Hz)': (25, 47),
    },
    "model_args": {'n_chans': 60, 'n_outputs': 1, 'n_times': 50, 'filter_time_length': 10},
    "wrapper_args": argparse.Namespace(
        use_tta=False, alignment_type="none", finetune_mode="full", ea_backrotation=False,
        use_adabn=False, alignment_cov_epsilon=1e-6, alignment_transform_epsilon=1e-7,
        tta_cov_buffer_size=50, alignment_ref_ema_beta=1.0
    )
}

# 2. PLOTTING & VISUALIZATION

def _apply_matplotlib_settings(width_mm=90, height_mm=60):
    """Applies consistent matplotlib settings for publication-quality plots."""
    mm_to_inch = 1 / 25.4
    settings = {
        "text.usetex": False, "mathtext.default": "regular",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 7, "figure.titlesize": 7, "legend.fontsize": 7,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "savefig.dpi": 300, "figure.dpi": 150, "savefig.format": "pdf",
        "savefig.bbox": "tight", "pdf.fonttype": 42,
        "figure.figsize": (width_mm * mm_to_inch, height_mm * mm_to_inch),
    }
    plt.rcParams.update(settings)

def plot_spatio_spectral_maps(spatio_spectral_maps, baseline_aucs, info, freq_bands,
                              method_name="Occlusion", stat_label="Single Subject", scale_mode="global"):
    """Draws one topomap per frequency band, showing the change (delta) in ROC AUC."""
    if spatio_spectral_maps is None:
        logging.error("Cannot plot maps: input data is None.")
        return
    n_bands = len(freq_bands)
    _apply_matplotlib_settings(width_mm=60 * n_bands, height_mm=65)

    fig, axes = plt.subplots(1, n_bands, squeeze=False)
    title_prefix = f"Population {stat_label.title()}" if stat_label != "Single Subject" else stat_label
    fig.suptitle(f"{title_prefix} Spatio-Spectral Importance ({method_name})", y=0.98)

    global_vmax = None
    if scale_mode == "global":
        all_vals = np.concatenate([v for v in spatio_spectral_maps.values() if v is not None and v.size > 0])
        if all_vals.size > 0:
            global_vmax = np.max(np.abs(all_vals))

    plots_made = False
    for ax, band in zip(axes.flatten(), freq_bands):
        delta_auc_values = spatio_spectral_maps.get(band)
        baseline_auc = baseline_aucs.get(band)
        if delta_auc_values is None or baseline_auc is None:
            ax.set_title(f"{band}\n(no data)"); ax.axis("off"); continue

        vmax = np.max(np.abs(delta_auc_values)) if scale_mode == "per-band" else global_vmax
        if vmax is None or vmax == 0: vmax = 1e-9

        im, _ = mne.viz.plot_topomap(delta_auc_values, info, axes=ax, cmap="RdBu_r", vlim=(-vmax, vmax),
                                     show=False, sensors=False, outlines="head", extrapolate="head")
        plots_made = True
        ax.set_title(f"{band}\n(Baseline AUC: {baseline_auc:.3f})")

    if plots_made and scale_mode == "global" and global_vmax is not None:
        norm = mpl.colors.Normalize(vmin=-global_vmax, vmax=global_vmax)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        cb = fig.colorbar(sm, cax=cax); cb.set_label("Δ ROC AUC")

    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    out_file = f"pop_{stat_label}_{scale_mode}_{method_name.lower().replace(' ', '_')}_delta_auc.pdf"
    plt.savefig(out_file); logging.info(f"✓ Saved figure: {out_file}"); plt.close(fig)

def plot_spatial_map(importance_map, baseline_auc, info, method_name, subject_label):
    """Plots a single topomap for broadband spatial importance."""
    _apply_matplotlib_settings()
    delta_auc_values = next(iter(importance_map.values()))
    fig, ax = plt.subplots()
    title = f"{subject_label} Broadband Importance\n({method_name} | Baseline AUC: {baseline_auc:.3f})"
    vmax = np.max(np.abs(delta_auc_values)) or 1e-9

    im, _ = mne.viz.plot_topomap(delta_auc_values, info, axes=ax, cmap="RdBu_r", vlim=(-vmax, vmax),
                                 show=False, sensors=False, outlines="head", extrapolate="head")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax=cax); cb.set_label("Δ ROC AUC")
    ax.set_title(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_file = f"{subject_label.lower().replace(' ', '_')}_spatial_map_delta_auc.pdf"
    plt.savefig(out_file); logging.info(f"✓ Saved figure: {out_file}"); plt.close(fig)

def plot_frequency_importance(importance_data, method_name, subject_label):
    """Plots a layered violin and box plot for frequency importance distribution."""
    _apply_matplotlib_settings()
    fig, ax = plt.subplots()
    bands = list(importance_data.keys())
    data_to_plot = [[v for v in importance_data.get(band, []) if not np.isnan(v)] for band in bands]
    if not any(data_to_plot):
        logging.warning("No valid data to plot for frequency importance."); plt.close(fig); return

    violin_parts = ax.violinplot(data_to_plot, showmeans=False, showmedians=False, showextrema=False, widths=0.8)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#e1e1e0'); pc.set_edgecolor('#555555'); pc.set_linewidth(0.5); pc.set_alpha(1.0)
    
    bp = ax.boxplot(data_to_plot, vert=True, patch_artist=True, widths=0.25, showfliers=False,
                    whiskerprops={'color': '#555555', 'linewidth': 0.7},
                    capprops={'color': '#555555', 'linewidth': 0.7},
                    medianprops={'color': '#555555', 'linewidth': 1.0, 'zorder': 4})
    for patch in bp['boxes']:
        patch.set_facecolor('w'); patch.set_edgecolor('#555555'); patch.set_linewidth(0.7)

    ax.set_ylabel("Δ ROC AUC (%)"); ax.set_title(f"{subject_label} Frequency Importance ({method_name})")
    ax.set_xticks(np.arange(1, len(bands) + 1)); ax.set_xticklabels(bands, rotation=45, ha="right")
    ax.grid(False); fig.tight_layout(pad=0.5)
    out_file = f"{subject_label.lower().replace(' ', '_')}_frequency_importance.pdf"
    plt.savefig(out_file); logging.info(f"✓ Saved figure: {out_file}"); plt.close(fig)

# 3. CORE INTERPRETABILITY & DATA HANDLING
def load_model_from_checkpoint(checkpoint_path, model_args, wrapper_args):
    """Loads a model state_dict from a checkpoint file into a TTAWrapper."""
    logging.info(f"-> Loading model from: {Path(checkpoint_path).name}")
    try:
        base_model = PRIME(**model_args)
        wrapped_model = TTAWrapper(base_model, wrapper_args)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        wrapped_model.load_state_dict(checkpoint['model_state_dict'])
        wrapped_model.eval()
        logging.info("   ...model loaded successfully.")
        return wrapped_model
    except Exception as e:
        logging.error(f"   [ERROR] Failed to load checkpoint: {e}")
        return None

def load_data_for_subject(dataset_name, subject_id, tmin, tmax, config):
    """Loads epochs data and MNE info for a single subject."""
    logging.info(f"→ Loading data for Subject {subject_id} (Dataset: {dataset_name})")
    dataset_map = {"TMSEEGClassificationTEPfree": (TMSEEGDatasetTEPfree, TMSEEGClassificationTEPfree),
                   "TMSEEGClassificationTEP": (TMSEEGDatasetTEP, TMSEEGClassificationTEP),
                   "TMSEEGClassification": (TMSEEGDataset, TMSEEGClassification)}
    dataset_class, paradigm_class = dataset_map[dataset_name]
    dataset = dataset_class()
    paradigm = paradigm_class(tmin=tmin, tmax=tmax)
    epochs_data, labels, _ = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    if epochs_data is None or epochs_data.size == 0:
        logging.warning(f"No data returned for subject {subject_id}.")
        return None, None, None

    try:
        raw = next(iter(next(iter(dataset._get_single_subject_data(subject_id).values())).values()))
        ch_names = raw.info["ch_names"][:config["N_CHANS"]]
    except Exception:
        ch_names = [f"CH{i:02d}" for i in range(config["N_CHANS"])]
        logging.warning("Could not get channel names; using generic placeholders.")

    info = mne.create_info(ch_names=ch_names, sfreq=config["SFREQ"], ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False, on_missing="warn")
    logging.info(f"   Loaded data shape: {epochs_data.shape}")
    return epochs_data, labels, info

def _create_neighbor_occlusion_masks(info, n_neighbors, device):
    """Creates boolean masks to occlude a channel and its k-nearest neighbors."""
    n_channels = len(info.ch_names)
    mask_bank = torch.ones((n_channels, n_channels), dtype=torch.bool, device=device)
    mask_bank.fill_diagonal_(False)
    if n_neighbors == 0: return mask_bank

    montage = info.get_montage()
    ch_pos_dict = montage.get_positions()['ch_pos'] if montage else {}
    present_ch = [ch for ch in info.ch_names if ch in ch_pos_dict]
    if len(present_ch) <= n_neighbors:
        logging.warning("Not enough channels with positions to find neighbors. Using single-channel occlusion.")
        return mask_bank
        
    pos_array = np.array([ch_pos_dict[ch] for ch in present_ch])
    distances = squareform(pdist(pos_array))
    name_to_idx = {name: i for i, name in enumerate(info.ch_names)}
    subset_to_main_idx = [name_to_idx[ch] for ch in present_ch]

    for i, ch_name in enumerate(present_ch):
        master_idx = name_to_idx[ch_name]
        dist_copy = distances[i].copy()
        dist_copy[i] = np.inf
        neighbor_indices_in_subset = np.argsort(dist_copy)[:n_neighbors]
        for neighbor_subset_idx in neighbor_indices_in_subset:
            neighbor_main_idx = subset_to_main_idx[neighbor_subset_idx]
            mask_bank[master_idx, neighbor_main_idx] = False
    return mask_bank

def compute_occlusion_maps_batched(model, data_tensor, labels, info, sfreq, freq_bands, n_times_model, device, n_neighbors):
    """Computes spatio-spectral importance using channel-wise band-stop filters."""
    model.eval()
    data_tensor = data_tensor.to(device)
    n_epochs, n_channels, _ = data_tensor.shape
    neighbor_masks = _create_neighbor_occlusion_masks(info, n_neighbors, device)

    nyquist = sfreq / 2.0
    filter_bank = {b: butter(4, [fmin/nyquist, fmax/nyquist], btype='bandstop') for b, (fmin, fmax) in freq_bands.items()}
    
    with torch.no_grad():
        baseline_logits = model(data_tensor[..., -n_times_model:]).squeeze()
        baseline_auc = roc_auc_score(labels, torch.sigmoid(baseline_logits).cpu().numpy())
    logging.info(f"Baseline AUC: {baseline_auc:.4f}")

    importance_maps = {}
    data_np_full = data_tensor.cpu().numpy()
    for band_name, (b, a) in filter_bank.items():
        logging.info(f"Analyzing importance of '{band_name}'...")
        occluded_aucs = np.zeros(n_channels)
        for chan_idx in range(n_channels):
            channels_to_occlude_mask = ~neighbor_masks[chan_idx, :]
            original_signals = data_np_full[:, channels_to_occlude_mask.cpu(), :]
            filtered_signals = filtfilt(b, a, original_signals, axis=-1).copy()
            
            occluded_data = data_tensor.clone()
            occluded_data[:, channels_to_occlude_mask, :] = torch.from_numpy(filtered_signals).float().to(device)
            
            with torch.no_grad():
                out_logits = model(occluded_data[..., -n_times_model:]).squeeze()
                occluded_aucs[chan_idx] = roc_auc_score(labels, torch.sigmoid(out_logits).cpu().numpy())
        importance_maps[band_name] = baseline_auc - occluded_aucs
        
    return importance_maps, importance_maps, {band: baseline_auc for band in freq_bands}

def compute_frequency_band_occlusion(model, data_tensor, labels, sfreq, freq_bands, n_times_model, device):
    """Computes importance of entire frequency bands by occluding them one by one."""
    model.eval()
    data_tensor = data_tensor.to(device)
    
    with torch.no_grad():
        baseline_logits = model(data_tensor[..., -n_times_model:]).squeeze()
        baseline_auc = roc_auc_score(labels, torch.sigmoid(baseline_logits).cpu().numpy())

    nyquist = sfreq / 2.0
    filter_bank = {b: butter(4, [fmin/nyquist, fmax/nyquist], btype='bandstop') for b, (fmin, fmax) in freq_bands.items()}
    data_np = data_tensor.cpu().numpy()
    importance_map = {}

    for band_name, (b, a) in filter_bank.items():
        filtered_np = filtfilt(b, a, data_np, axis=-1).copy()
        filtered_data = torch.from_numpy(filtered_np[..., -n_times_model:]).to(device, dtype=torch.float32)
        with torch.no_grad():
            occluded_logits = model(filtered_data).squeeze()
            occluded_auc = roc_auc_score(labels, torch.sigmoid(occluded_logits).cpu().numpy())
        importance_map[band_name] = baseline_auc - occluded_auc
        
    return importance_map, importance_map, {"baseline_auc": baseline_auc}

def compute_spatial_occlusion_map(model, data_tensor, labels, info, n_times_model, device, batch_size, n_neighbors):
    """Computes broadband spatial importance by occluding channel neighborhoods."""
    model.eval()
    data_tensor = data_tensor.to(device)
    n_epochs, n_channels, _ = data_tensor.shape
    broadband_data = data_tensor[..., -n_times_model:]
    mask_bank = _create_neighbor_occlusion_masks(info, n_neighbors, device)

    all_baseline_logits = []
    with torch.no_grad():
        for i0 in range(0, n_epochs, batch_size):
            logits = model(broadband_data[i0:i0+batch_size]).squeeze(dim=-1)
            all_baseline_logits.append(logits.cpu())
    baseline_auc = roc_auc_score(labels, torch.sigmoid(torch.cat(all_baseline_logits)).numpy())
    logging.info(f"Broadband Spatial Occlusion: Baseline AUC = {baseline_auc:.4f}")

    all_occluded_logits = torch.zeros(n_epochs, n_channels)
    with torch.no_grad():
        for i0 in range(0, n_epochs, batch_size):
            epoch_chunk = broadband_data[i0:i0+batch_size]
            occl = epoch_chunk.unsqueeze(1) * mask_bank.view(1, n_channels, n_channels, 1)
            occl = occl.view(-1, n_channels, n_times_model)
            all_occluded_logits[i0:i0+batch_size] = model(occl).view(epoch_chunk.shape[0], n_channels).cpu()

    occluded_aucs = np.array([roc_auc_score(labels, torch.sigmoid(all_occluded_logits[:, i]).numpy()) for i in range(n_channels)])
    importance_values = baseline_auc - occluded_aucs
    
    return {"Broadband": importance_values}, {"Broadband": importance_values}, {"Broadband": baseline_auc}

def process_subject(checkpoint_path, config):
    """Main pipeline to process a single subject: load data, model, and run interpretability method."""
    try:
        m = re.search(r"ds_(?P<dataset>.+?)_subj_(?P<subject>\d+)", checkpoint_path.name)
        if not m:
            return None, None, None, None
        dataset_name, subject_id = m.group("dataset"), int(m.group("subject"))
        logging.info(f"\n--- Processing Subject {subject_id} | Dataset: {dataset_name} ---")

        epochs_np, labels, info = load_data_for_subject(dataset_name, subject_id, config["T_MIN"], config["T_MAX"], config)
        if epochs_np is None:
            return None, None, None, None
        labels = (labels > 0.5).astype(int)

        model = load_model_from_checkpoint(checkpoint_path, config["model_args"], config["wrapper_args"])
        if model is None:
            return None, None, None, None
        model.to(config["device"]).eval()
        data_tensor = torch.from_numpy(epochs_np).float()

        method = config.get("method", "occlusion")
        logging.info(f"Running interpretability method: '{method}'")

        # Capture the results from the computation functions
        if method == "spatial_occlusion":
            subject_maps, deltas, baseline_aucs = compute_spatial_occlusion_map(
                model, data_tensor, labels, info, config["N_TIMES"], config["device"],
                config["OCCLUSION_BATCH_SIZE"], config["OCCLUSION_N_NEIGHBORS"])
        elif method == "frequency_occlusion":
            subject_maps, deltas, baseline_aucs = compute_frequency_band_occlusion(
                model, data_tensor, labels, config["SFREQ"], config["FREQ_BANDS"],
                config["N_TIMES"], config["device"])
        elif method == "occlusion":
            subject_maps, deltas, baseline_aucs = compute_occlusion_maps_batched(
                model, data_tensor, labels, info, config["SFREQ"], config["FREQ_BANDS"],
                config["N_TIMES"], config["device"], config["OCCLUSION_N_NEIGHBORS"])
        else:
            raise NotImplementedError(f"Method '{method}' is not implemented.")

        # Explicitly return the four required items
        return subject_maps, deltas, info, baseline_aucs

    except Exception as e:
        logging.error(f"FAILED on {checkpoint_path.name}: {e}", exc_info=True)
        return None, None, None, None

# 4. JOB SUBMISSION & AGGREGATION
def chunk_list(data: list, size: int):
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(data), size): yield data[i:i + size]

def run_analysis_for_chunk(checkpoint_chunk, config, results_dir):
    """Worker function executed by SLURM for a chunk of subjects."""
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    logging.info(f"--- SLURM Job {job_id} starting, processing {len(checkpoint_chunk)} subjects ---")
    for cp_path in checkpoint_chunk:
        subject_id = re.search(r"subj_(\d+)", cp_path.stem).group(1)
        try:
            maps, deltas, info, baseline_aucs = process_subject(cp_path, config)
            if maps:
                output_file = results_dir / f"result_subj_{subject_id}.pkl"
                out_obj = {"subject_id": int(subject_id), "channel_names": info["ch_names"],
                           "importance_map": maps, "deltas": deltas, "baseline_aucs": baseline_aucs}
                with open(output_file, "wb") as f:
                    pickle.dump(out_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"--- Job {job_id}: Successfully saved result for subject {subject_id} ---")
            else:
                logging.warning(f"--- Job {job_id}: Analysis returned no result for subject {subject_id} ---")
        except Exception as e:
            logging.error(f"--- Job {job_id}: FAILED subject {subject_id}. Error: {e} ---", exc_info=True)
    return f"Finished Job {job_id}"

def perform_significance_testing(all_deltas, feature_names, results_dir):
    """Performs Wilcoxon signed-rank test on aggregated deltas and saves a LaTeX table."""
    tex_string = "\\documentclass{article}\n\\usepackage{booktabs}\n\\usepackage{siunitx}\n\\begin{document}\n"
    for band, deltas_matrix in all_deltas.items():
        if deltas_matrix is None or deltas_matrix.size == 0: continue
        n_features = deltas_matrix.shape[1]
        p_values = [wilcoxon(deltas_matrix[:, i], alternative='greater')[1] for i in range(n_features)]
        stats = [wilcoxon(deltas_matrix[:, i], alternative='greater')[0] for i in range(n_features)]
        _, p_vals_corrected = fdrcorrection(p_values, alpha=0.05, method='indep')
        
        band_feature_names = feature_names.get(band, [f"Feature {i+1}" for i in range(n_features)])
        tex_string += f"\\section*{{Statistical Results for {band.replace('_', ' ')} Band}}\n"
        tex_string += "\\begin{tabular}{l S[table-format=7.1] S[table-format=1.4e-2] S[table-format=1.4e-2]}\n\\toprule\n"
        tex_string += "{Feature} & {W-statistic} & {p-value} & {p-value (FDR corrected)} \\\\\n\\midrule\n"
        for i in range(n_features):
            tex_string += f"{band_feature_names[i]} & {stats[i]:.1f} & {p_values[i]:.4e} & {p_vals_corrected[i]:.4e} \\\\\n"
        tex_string += "\\bottomrule\n\\end{tabular}\n"
    tex_string += "\\end{document}\n"
    
    output_file = results_dir / "statistical_summary.tex"
    with open(output_file, "w") as f: f.write(tex_string)
    logging.info(f"✓ Statistical analysis saved to {output_file}")

def _get_reorder_idx(rec, canonical_ch):
    """Helper to get indices for reordering channels to a canonical order."""
    return [rec["channel_names"].index(ch) for ch in canonical_ch]

def aggregate_plot_and_test(results_dir, config, statistic="median", scale_mode="global"):
    """Aggregates results, plots population-level maps, and runs statistical tests."""
    logging.info(f"──► Aggregating results from: {results_dir}")
    files = sorted(results_dir.glob("result_subj_*.pkl"))
    if not files: logging.error("No result files found to aggregate."); return

    all_subject_data = [pickle.load(open(fp, "rb")) for fp in files]
    canonical_ch = all_subject_data[0]["channel_names"]
    info = mne.create_info(ch_names=canonical_ch, sfreq=config["SFREQ"], ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False, on_missing="warn")

    method, method_name = config["method"], config["method"].replace("_", " ").title()
    os.chdir(results_dir)
    reducer = np.mean if statistic == "mean" else np.median

    logging.info(f"Aggregating for method: {method_name}...")
    if method == 'occlusion':
        agg_maps = {b: [np.asarray(r["importance_map"][b])[_get_reorder_idx(r, canonical_ch)] for r in all_subject_data if b in r["importance_map"]] for b in config["FREQ_BANDS"]}
        agg_deltas = {b: [np.asarray(r["deltas"][b])[_get_reorder_idx(r, canonical_ch)] for r in all_subject_data if b in r["deltas"]] for b in config["FREQ_BANDS"]}
        agg_baselines = {b: [r["baseline_aucs"][b] for r in all_subject_data if b in r["baseline_aucs"]] for b in config["FREQ_BANDS"]}
        
        grand_map = {b: reducer(np.stack(lst), axis=0) for b, lst in agg_maps.items() if lst}
        grand_baselines = {b: reducer(lst) for b, lst in agg_baselines.items() if lst}
        plot_spatio_spectral_maps(grand_map, grand_baselines, info, list(config["FREQ_BANDS"].keys()), method_name, statistic, scale_mode)
        final_deltas = {b: np.vstack(lst) for b, lst in agg_deltas.items() if lst}
        perform_significance_testing(final_deltas, {b: canonical_ch for b in config["FREQ_BANDS"]}, results_dir)

    elif method == 'spatial_occlusion':
        maps_list = [np.asarray(r["importance_map"]["Broadband"])[_get_reorder_idx(r, canonical_ch)] for r in all_subject_data]
        deltas_list = [np.asarray(r["deltas"]["Broadband"])[_get_reorder_idx(r, canonical_ch)] for r in all_subject_data]
        baselines_list = [r["baseline_aucs"]["Broadband"] for r in all_subject_data]
        
        grand_map = {"Broadband": reducer(np.stack(maps_list), axis=0)}
        plot_spatial_map(grand_map, reducer(baselines_list), info, method_name, f"Population {statistic.title()}")
        perform_significance_testing({"Broadband": np.vstack(deltas_list)}, {"Broadband": canonical_ch}, results_dir)

    elif method == 'frequency_occlusion':
        maps_dict = {b: [r["importance_map"][b] for r in all_subject_data] for b in config["FREQ_BANDS"]}
        deltas_dict = {b: [r["deltas"][b] for r in all_subject_data] for b in config["FREQ_BANDS"]}
        baselines_list = [r["baseline_aucs"]["baseline_auc"] for r in all_subject_data]
        
        importance_pct_data = {b: [(maps_dict[b][i] / (baseline + 1e-9)) * 100 for i, baseline in enumerate(baselines_list)] for b in config["FREQ_BANDS"]}
        plot_frequency_importance(importance_pct_data, method_name, f"Population {statistic.title()}")
        final_deltas = {b: np.array(v).reshape(-1, 1) for b, v in deltas_dict.items()}
        perform_significance_testing(final_deltas, {b: [b] for b in config["FREQ_BANDS"]}, results_dir)
            
    logging.info("✓ Aggregation, plotting, and testing complete.")

# ==========================================================================================
# 5. MAIN EXECUTION
# ==========================================================================================

def main():
    """Main function to parse arguments and dispatch actions."""
    parser = argparse.ArgumentParser(description="Submit and aggregate interpretability analysis jobs.")
    parser.add_argument("--mode", choices=['submit', 'aggregate'], required=True, help="Operation mode.")
    parser.add_argument("--method", choices=['occlusion', 'spatial_occlusion', 'frequency_occlusion'], 
                        default='occlusion', help="The interpretability method to use.")
    parser.add_argument("--results_dir", type=Path, help="Path to results directory for aggregation.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare jobs but do not submit them.")
    args = parser.parse_args()

    ANALYSIS_CONFIG["method"] = args.method
    
    if args.mode == 'submit':
        checkpoint_paths = sorted(list(CHECKPOINTS_DIR.glob("finetuned_DeepTEPNet_ds_*_subj_*.pt")))
        if not checkpoint_paths:
            logging.error(f"No checkpoints found in {CHECKPOINTS_DIR}. Exiting."); return

        subject_chunks = list(chunk_list(checkpoint_paths, SUBJECTS_PER_JOB))
        logging.info(f"Found {len(checkpoint_paths)} subjects, grouped into {len(subject_chunks)} jobs.")
        logging.info(f"Analysis method to be run: '{args.method}'")
        if args.dry_run:
            logging.info("\n--- DRY RUN: No jobs will be submitted. ---")
            return

        run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_interpret_{args.method}"
        log_folder = BASE_LOG_DIR / run_name
        results_folder = log_folder / "results"
        results_folder.mkdir(exist_ok=True, parents=True)
        
        executor = submitit.AutoExecutor(folder=str(log_folder))
        executor.update_parameters(slurm_partition=SLURM_PARTITION, slurm_time=JOB_TIME,
                                   mem_gb=MEM_GB_PER_JOB, cpus_per_task=CPUS_PER_JOB,
                                   slurm_gres=f"gpu:{GPUS_PER_JOB}")

        logging.info(f"Submitting {len(subject_chunks)} jobs to SLURM...")
        jobs = executor.map_array(run_analysis_for_chunk, subject_chunks, [ANALYSIS_CONFIG]*len(subject_chunks), [results_folder]*len(subject_chunks))
        
        print(f"\n--- All {len(jobs)} jobs submitted successfully! ---")
        print(f"\nTo aggregate results after completion, run:")
        print(f"python {sys.argv[0]} --mode aggregate --method {args.method} --results_dir {results_folder}")

    elif args.mode == 'aggregate':
        if not args.results_dir or not args.results_dir.is_dir():
            logging.error("A valid --results_dir must be provided for 'aggregate' mode."); sys.exit(1)
        aggregate_plot_and_test(args.results_dir, ANALYSIS_CONFIG)

if __name__ == "__main__":
    main()