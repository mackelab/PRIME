#%% ============================================================================
# CELL 1: IMPORTS & SETUP
# ============================================================================
import logging
import pickle
from pathlib import Path
import sys

import mne
import numpy as np
import pandas as pd

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Data Aggregation Function  ---
def aggregate_data(results_dir: Path, method: str, config: dict, statistic: str = "median"):
    """
    Loads and aggregates interpretability data from result files for a given method.
    Converts topoplot data to percentage change.
    """
    logging.info(f"──► Aggregating data for '{method}' from: {results_dir}")
    files = sorted(results_dir.glob("result_subj_*.pkl"))
    if not files:
        logging.error(f"No result_*.pkl files found in {results_dir}. Cannot generate panel.")
        return None

    all_subject_data = []
    for fp in files:
        with open(fp, "rb") as f:
            all_subject_data.append(pickle.load(f))

    canonical_ch = all_subject_data[0]["channel_names"]
    info = mne.create_info(ch_names=canonical_ch, sfreq=config["SFREQ"], ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False, on_missing="warn")

    reducer = np.mean if statistic == "mean" else np.median
    aggregated_payload = {"info": info}

    if method == 'spatial_occlusion':
        percent_maps_list = []
        for rec in all_subject_data:
            reorder_idx = [rec["channel_names"].index(ch) for ch in canonical_ch]
            baseline_auc = rec["baseline_aucs"]["Broadband"]
            percent_map = (np.asarray(rec["importance_map"]["Broadband"]) / (baseline_auc + 1e-9)) * 100
            percent_maps_list.append(percent_map[reorder_idx])
        aggregated_payload["grand_map"] = reducer(np.stack(percent_maps_list), axis=0)

    elif method == 'frequency_occlusion':
        maps_dict = {band: [] for band in config["FREQ_BANDS"]}
        baselines_list = [rec["baseline_aucs"]["baseline_auc"] for rec in all_subject_data]
        for rec in all_subject_data:
            for band in config["FREQ_BANDS"]:
                maps_dict[band].append(rec["importance_map"][band])

        importance_pct_data = {band: [] for band in config["FREQ_BANDS"]}
        for i, baseline_auc in enumerate(baselines_list):
            for band in config["FREQ_BANDS"]:
                absolute_drop = maps_dict[band][i]
                percent_decrease = (absolute_drop / (baseline_auc + 1e-9)) * 100
                importance_pct_data[band].append(percent_decrease)
        aggregated_payload["importance_data"] = importance_pct_data
        aggregated_payload["bands"] = list(config["FREQ_BANDS"].keys())

    elif method == 'occlusion': # Spatio-spectral
        aggregated_percent_maps = {band: [] for band in config["FREQ_BANDS"]}
        for rec in all_subject_data:
            reorder_idx = [rec["channel_names"].index(ch) for ch in canonical_ch]
            for band in config["FREQ_BANDS"]:
                if rec["importance_map"].get(band) is not None:
                    baseline_auc = rec["baseline_aucs"][band]
                    percent_map = (np.asarray(rec["importance_map"][band]) / (baseline_auc + 1e-9)) * 100
                    aggregated_percent_maps[band].append(percent_map[reorder_idx])
        
        aggregated_payload["grand_map"] = {band: reducer(np.stack(lst), axis=0) if lst else None for band, lst in aggregated_percent_maps.items()}
        aggregated_payload["bands"] = list(config["FREQ_BANDS"].keys())

    logging.info(f"✓ Aggregation complete for '{method}'.")
    return aggregated_payload

#%% ============================================================================
# CELL 2: PROCESSING PIPELINE
# ============================================================================
def process_and_save_dataset(name_suffix: str, paths: dict, config: dict):
    """
    Runs the full aggregation and saving pipeline for a given dataset.
    """
    logging.info(f"\n{'='*60}\nProcessing dataset: {name_suffix}\n{'='*60}")
    
    # --- Aggregate Data ---
    spatial_data = aggregate_data(paths['spatial'], 'spatial_occlusion', config, 'median')
    freq_data = aggregate_data(paths['freq'], 'frequency_occlusion', config, 'median')
    spatio_spectral_data = aggregate_data(paths['spatio_spectral'], 'occlusion', config, 'median')

    if not all([spatial_data, freq_data, spatio_spectral_data]):
        logging.error(f"Failed to load data for '{name_suffix}'. Skipping CSV generation.")
        return

    # --- Prepare and Save Data for Panel A (Spatial) ---
    df_spatial = pd.DataFrame({
        'channel': spatial_data['info']['ch_names'],
        'importance_percent': spatial_data['grand_map']
    })
    output_path = f"spatial_occlusion_data{name_suffix}.csv"
    df_spatial.to_csv(output_path, index=False)
    logging.info(f"✓ Saved Panel A data to: {output_path}")

    # --- Prepare and Save Data for Panel B (Frequency) ---
    long_format_data = []
    for band, values in freq_data['importance_data'].items():
        for value in values:
            long_format_data.append({'band': band, 'importance_percent': value})
    df_freq = pd.DataFrame(long_format_data)
    output_path = f"frequency_occlusion_data{name_suffix}.csv"
    df_freq.to_csv(output_path, index=False)
    logging.info(f"✓ Saved Panel B data to: {output_path}")

    # --- Prepare and Save Data for Panel C (Spatio-Spectral) ---
    df_spatio_spectral = pd.DataFrame({'channel': spatio_spectral_data['info']['ch_names']})
    for band in spatio_spectral_data['bands']:
        df_spatio_spectral[band] = spatio_spectral_data['grand_map'][band]
    output_path = f"spatio_spectral_occlusion_data{name_suffix}.csv"
    df_spatio_spectral.to_csv(output_path, index=False)
    logging.info(f"✓ Saved Panel C data to: {output_path}")


#%% ============================================================================
# CELL 3: MAIN EXECUTION
# ============================================================================

# --- Analysis Configuration ---
ANALYSIS_CONFIG = {
    "N_CHANS": 60, "SFREQ": 1000,
    "FREQ_BANDS": {
        'Theta (4-8 Hz)': (4, 8), 'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-25 Hz)': (13, 25), 'Gamma (25-47 Hz)': (25, 47),
    }
}

# --- Paths for the Main Experiment ---
MAIN_PATHS = {
    'spatial': Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/slurm_logs_interpretability/2025-07-08_13-45_interpretability_spatial_occlusion/results"),
    'freq': Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/slurm_logs_interpretability/2025-07-08_06-41_interpretability_frequency_occlusion/results"),
    'spatio_spectral': Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/slurm_logs_interpretability/2025-07-08_09-43_interpretability_occlusion/results")
}

# --- Paths for the P60 Experiment ---
P60_PATHS = {
    'spatial': Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/slurm_logs_interpretability_p60/2025-07-08_13-49_interpretability_spatial_occlusion/results"),
    'freq': Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/slurm_logs_interpretability_p60/2025-07-08_08-03_interpretability_frequency_occlusion/results"),
    'spatio_spectral': Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/slurm_logs_interpretability_p60/2025-07-08_09-47_interpretability_occlusion/results")
}

# --- Run Processing ---
process_and_save_dataset(name_suffix="", paths=MAIN_PATHS, config=ANALYSIS_CONFIG)
process_and_save_dataset(name_suffix="_p60", paths=P60_PATHS, config=ANALYSIS_CONFIG)

logging.info("\nAll CSV data files have been created successfully.")
# %%
