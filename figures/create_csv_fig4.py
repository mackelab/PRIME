#%%
'''
This script processes raw experimental results to generate the specific data files
required for plotting the composite figure (Fig4). It extracts performance metrics from
various experimental configurations and saves them as intermediate CSV files.
'''

import os
import re
from pathlib import Path
import pandas as pd
from collections import OrderedDict

# --- Data Loading and Processing Functions  ---

def find_detailed_results_file(base_path):
    """Recursively searches for 'results_summary_detailed.csv'."""
    files = list(Path(base_path).rglob("results_summary_detailed.csv"))
    return files[0] if files else None

def load_all_configs_performance(results_root, grid_dir, model_name, configs_to_load, dataset):
    """Loads performance data for a single model across specified configurations."""
    all_subject_results = []
    grid_base_dir = Path(results_root) / grid_dir
    print(f"\n--- Scanning for CONFIG comparison data in: {grid_base_dir} ---")
    for config_tag, config_plot_name in configs_to_load.items():
        exp_dir_pattern = f"AlignEval_{model_name}_{dataset}_{config_tag}"
        found_dirs = list(grid_base_dir.glob(f"**/{exp_dir_pattern}"))
        if not found_dirs:
            print(f"Warning: No directory for config '{config_plot_name}' with pattern '{exp_dir_pattern}'")
            continue
        summary_csv_path = find_detailed_results_file(found_dirs[0])
        if summary_csv_path:
            print(f"Found data for '{config_plot_name}' in: {found_dirs[0].name}")
            df = pd.read_csv(summary_csv_path)
            df_processed = df[['subject_id', 'finetuned_roc_auc', 'finetuned_roc_auc_extreme']].copy()
            df_processed['config'] = config_plot_name
            all_subject_results.append(df_processed)
    return pd.concat(all_subject_results, ignore_index=True) if all_subject_results else pd.DataFrame()

def load_performance_by_grid(results_root, model_name, config_tag, dataset, grid_configs, category_col_name, scan_title):
    """Generic function to load performance data by iterating through grid directories."""
    all_subject_results = []
    print(f"\n--- Scanning for {scan_title} data ---")
    for label, grid_dir in grid_configs.items():
        grid_base_dir = Path(results_root) / grid_dir
        exp_dir_pattern = f"AlignEval_{model_name}_{dataset}_{config_tag}"
        found_dirs = list(grid_base_dir.glob(f"**/{exp_dir_pattern}"))
        if not found_dirs:
            print(f"Warning: No directory for '{label}' with pattern '{exp_dir_pattern}' in '{grid_dir}'")
            continue
        summary_csv_path = find_detailed_results_file(found_dirs[0])
        if summary_csv_path:
            print(f"Found data for '{label}' in: {found_dirs[0].name}")
            df = pd.read_csv(summary_csv_path)
            df_processed = df[['subject_id', 'finetuned_roc_auc', 'finetuned_roc_auc_extreme']].copy()
            df_processed[category_col_name] = label
            all_subject_results.append(df_processed)
    return pd.concat(all_subject_results, ignore_index=True) if all_subject_results else pd.DataFrame()

# --- Main Data Generation Function ---

def generate_and_save_data():
    """
    Main function to orchestrate data loading, processing, and saving to CSV.
    """
    try:
        USERNAME = os.environ.get('USER', 'default_user')
        RESULTS_ROOT_DIR = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/results_final")
        if not RESULTS_ROOT_DIR.exists(): raise FileNotFoundError
    except (KeyError, FileNotFoundError):
        RESULTS_ROOT_DIR = Path("./results_final")
        print(f"Warning: Could not find lustre directory. Falling back to local './results_final'")

    # --- Create output directory for generated CSVs ---
    OUTPUT_DATA_DIR = Path("./figure_data_for_reproduction")
    OUTPUT_DATA_DIR.mkdir(exist_ok=True)
    print(f"Data will be saved to: {OUTPUT_DATA_DIR.resolve()}")

    # --- Define constants and paths ---
    GRID_DIR_BASE = "10ms_pp_w50"
    CSV_PATH_PRETRAIN_N45 = RESULTS_ROOT_DIR / "10ms_pp_w50/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F/20250704_060709/results_summary_detailed.csv"
    CSV_PATH_PRETRAIN_P60 = RESULTS_ROOT_DIR / "10ms_pp_w50_p60/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F/20250707_170031/results_summary_detailed.csv"
    CSV_PATH_MEP = RESULTS_ROOT_DIR / "10ms_pp_w50/AlignEval_DeepTEPNet_TMSEEGClassification_FM-Full_A-None_AdaBN-F/AlignEval_DeepTEPNet_TMSEEGClassification_FM-Full_A-None_AdaBN-F/20250709_174256/results_summary_detailed.csv"
    DATASET, PRIME_MODEL_NAME, CONFIG_TAG = "TMSEEGClassificationTEPfree", "DeepTEPNet", "FM-Full_A-None_AdaBN-F"

    # --- Panel A Data: Configurations ---
    configs_to_plot = OrderedDict([("FM-None_A-Eucl_AdaBN-F", "EA"), ("FM-Dec_A-None_AdaBN-F", "Dec-CFT"), ("FM-Full_A-None_AdaBN-F", "CFT"), ("FM-Dec_A-Eucl_AdaBN-F", "EA+\nDec-CFT"),("FM-Full_A-Eucl_AdaBN-F", "EA+\nCFT"), ("FM-DecThr_A-None_AdaBN-F", "Thr-CFT"), ("FM-DecThr_A-Eucl_AdaBN-F", "EA+\nThr-CFT")])
    df_config = load_all_configs_performance(RESULTS_ROOT_DIR, GRID_DIR_BASE, PRIME_MODEL_NAME, configs_to_plot, DATASET)
    if not df_config.empty:
        df_config.to_csv(OUTPUT_DATA_DIR / "panel_a_config.csv", index=False)
        print(f"✅ Successfully saved data for Panel A to {OUTPUT_DATA_DIR / 'panel_a_config.csv'}")

    # --- Panel B Data: Window Size ---
    win_size_configs = OrderedDict([("50 ms", "10ms_pp_w50"), ("100 ms", "10ms_pp_w100 "), ("200 ms", "10ms_pp_w200 "), ("300 ms", "10ms_pp_w300 "), ("400 ms", "10ms_pp_w400 "), ("500 ms", "10ms_pp_w500 ")])
    df_winsize = load_performance_by_grid(RESULTS_ROOT_DIR, PRIME_MODEL_NAME, CONFIG_TAG, DATASET, win_size_configs, 'window_size', 'WINDOW SIZE')
    if not df_winsize.empty:
        df_winsize.to_csv(OUTPUT_DATA_DIR / "panel_b_winsize.csv", index=False)
        print(f"✅ Successfully saved data for Panel B to {OUTPUT_DATA_DIR / 'panel_b_winsize.csv'}")

    # --- Panel C Data: Window Location ---
    win_loc_configs = OrderedDict([("10 ms", "10ms_pp_w50"), ("20 ms", "20ms_pp_w50 "), ("30 ms", "30ms_pp_w50 ")])
    df_winloc = load_performance_by_grid(RESULTS_ROOT_DIR, PRIME_MODEL_NAME, CONFIG_TAG, DATASET, win_loc_configs, 'window_location', 'WINDOW LOCATION')
    if not df_winloc.empty:
        df_winloc.to_csv(OUTPUT_DATA_DIR / "panel_c_winloc.csv", index=False)
        print(f"✅ Successfully saved data for Panel C to {OUTPUT_DATA_DIR / 'panel_c_winloc.csv'}")

    # --- Panel D Data: Calibration Trials ---
    cal_configs_to_load = OrderedDict([("100 Trials", "10ms_pp_w50"), ("200 Trials", "10ms_pp_w50_200cal "), ("300 Trials", "10ms_pp_w50_300cal ")])
    df_cal = load_performance_by_grid(RESULTS_ROOT_DIR, PRIME_MODEL_NAME, CONFIG_TAG, DATASET, cal_configs_to_load, 'calibration_trials', 'CALIBRATION')
    if not df_cal.empty:
        df_cal.to_csv(OUTPUT_DATA_DIR / "panel_d_calibration.csv", index=False)
        print(f"✅ Successfully saved data for Panel D to {OUTPUT_DATA_DIR / 'panel_d_calibration.csv'}")

    # --- Panel E Data: N45 vs P60 Scatter ---
    try:
        df_e1 = pd.read_csv(CSV_PATH_PRETRAIN_N45)[['subject_id', 'finetuned_roc_auc']].rename(columns={'finetuned_roc_auc': 'roc_auc_1'})
        df_e2 = pd.read_csv(CSV_PATH_PRETRAIN_P60)[['subject_id', 'finetuned_roc_auc']].rename(columns={'finetuned_roc_auc': 'roc_auc_2'})
        df_merged_e = pd.merge(df_e1, df_e2, on='subject_id')
        df_merged_e.to_csv(OUTPUT_DATA_DIR / "panel_e_n45_vs_p60.csv", index=False)
        print(f"✅ Successfully saved data for Panel E to {OUTPUT_DATA_DIR / 'panel_e_n45_vs_p60.csv'}")
    except Exception as e:
        print(f"❌ Error generating data for Panel E: {e}")

    # --- Panel F Data: N45 vs MEP Scatter ---
    try:
        df_f1 = pd.read_csv(CSV_PATH_PRETRAIN_N45)[['subject_id', 'finetuned_roc_auc']].rename(columns={'finetuned_roc_auc': 'roc_auc_1'})
        df_f2 = pd.read_csv(CSV_PATH_MEP)[['subject_id', 'finetuned_roc_auc']].rename(columns={'finetuned_roc_auc': 'roc_auc_2'})
        df_merged_f = pd.merge(df_f1, df_f2, on='subject_id')
        df_merged_f.to_csv(OUTPUT_DATA_DIR / "panel_f_n45_vs_mep.csv", index=False)
        print(f"✅ Successfully saved data for Panel F to {OUTPUT_DATA_DIR / 'panel_f_n45_vs_mep.csv'}")
    except Exception as e:
        print(f"❌ Error generating data for Panel F: {e}")

if __name__ == "__main__":
    generate_and_save_data()
# %%
