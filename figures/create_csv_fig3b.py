#%%
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from collections import OrderedDict

def find_detailed_results_file(base_path):
    """
    Recursively searches for 'results_summary_detailed.csv' within a base path.
    Returns the Path object of the first file found, or None.
    """
    files = list(base_path.rglob("results_summary_detailed.csv"))
    if files:
        return files[0]
    return None

def load_all_models_performance(results_root, grid_dir, models_to_load, config_tag, dataset):
    """
    Loads and consolidates performance data for specified models and a single configuration.
    """
    all_subject_results = []
    grid_base_dir = results_root / grid_dir
    print(f"DEBUG: Checking for path -> '{grid_base_dir}'")
    if not grid_base_dir.exists():
        print(f"FATAL: Grid directory not found: {grid_base_dir}")
        return None

    print(f"--- Scanning for MODEL comparison data in: {grid_base_dir} ---")
    
    for model_dir_name, model_plot_name in models_to_load.items():
        exp_dir_pattern = f"AlignEval_{model_dir_name}_{dataset}_{config_tag}"
        found_dirs = list(grid_base_dir.glob(f"**/{exp_dir_pattern}"))
        
        if not found_dirs:
            print(f"Warning: No directory found for model '{model_dir_name}' with pattern '{exp_dir_pattern}'")
            continue
            
        exp_dir = found_dirs[0]
        print(f"Found data for '{model_plot_name}' in: {exp_dir.name}")
        
        summary_csv_path = find_detailed_results_file(exp_dir)
        if not summary_csv_path:
            print(f"  -> Warning: 'results_summary_detailed.csv' not found in {exp_dir}")
            continue

        try:
            df = pd.read_csv(summary_csv_path)
            if 'finetuned_roc_auc' not in df.columns or 'finetuned_roc_auc_extreme' not in df.columns:
                print(f"  -> Warning: Required roc_auc columns not found in {summary_csv_path}")
                continue

            df_processed = df[['subject_id', 'finetuned_roc_auc', 'finetuned_roc_auc_extreme']].copy()
            df_processed['model'] = model_plot_name
            all_subject_results.append(df_processed)
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            print(f"  -> Could not process file {summary_csv_path}: {e}")

    if not all_subject_results:
        print("Error: No valid experiment data was loaded for model comparison!")
        return None
    
    return pd.concat(all_subject_results, ignore_index=True)

def load_all_configs_performance(results_root, grid_dir, model_name, configs_to_load, dataset):
    """
    Loads and consolidates performance data for a single model across specified configurations.
    """
    all_subject_results = []
    grid_base_dir = results_root / grid_dir
    if not grid_base_dir.exists():
        print(f"FATAL: Grid directory not found: {grid_base_dir}")
        return None

    print(f"\n--- Scanning for CONFIG comparison data in: {grid_base_dir} ---")
    
    for config_tag, config_plot_name in configs_to_load.items():
        exp_dir_pattern = f"AlignEval_{model_name}_{dataset}_{config_tag}"
        found_dirs = list(grid_base_dir.glob(f"**/{exp_dir_pattern}"))
        
        if not found_dirs:
            print(f"Warning: No directory found for config '{config_plot_name}' with pattern '{exp_dir_pattern}'")
            continue
            
        exp_dir = found_dirs[0]
        print(f"Found data for '{config_plot_name}' in: {exp_dir.name}")
        
        summary_csv_path = find_detailed_results_file(exp_dir)
        if not summary_csv_path:
            print(f"  -> Warning: 'results_summary_detailed.csv' not found in {exp_dir}")
            continue

        try:
            df = pd.read_csv(summary_csv_path)
            if 'finetuned_roc_auc' not in df.columns or 'finetuned_roc_auc_extreme' not in df.columns:
                print(f"  -> Warning: Required roc_auc columns not found in {summary_csv_path}")
                continue
                
            df_processed = df[['subject_id', 'finetuned_roc_auc', 'finetuned_roc_auc_extreme']].copy()
            df_processed['config'] = config_plot_name
            all_subject_results.append(df_processed)
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            print(f"  -> Could not process file {summary_csv_path}: {e}")

    if not all_subject_results:
        print("Error: No valid experiment data was loaded for config comparison!")
        return None
    
    return pd.concat(all_subject_results, ignore_index=True)

def load_performance_by_grid(results_root, model_name, config_tag, dataset, grid_configs, category_col_name, scan_title):
    """
    Generic function to load performance data by iterating through a dictionary of grid directories.
    """
    all_subject_results = []
    print(f"\n--- Scanning for {scan_title} data ---")
    
    for label, grid_dir in grid_configs.items():
        grid_base_dir = results_root / grid_dir
        if not grid_base_dir.exists():
            print(f"FATAL: Grid directory not found: {grid_base_dir}")
            continue

        exp_dir_pattern = f"AlignEval_{model_name}_{dataset}_{config_tag}"
        found_dirs = list(grid_base_dir.glob(f"**/{exp_dir_pattern}"))
        
        if not found_dirs:
            print(f"Warning: No directory found for '{label}' with pattern '{exp_dir_pattern}' in '{grid_dir}'")
            continue
            
        exp_dir = found_dirs[0]
        print(f"Found data for '{label}' in: {exp_dir.name}")
        
        summary_csv_path = find_detailed_results_file(exp_dir)
        if not summary_csv_path:
            print(f"  -> Warning: 'results_summary_detailed.csv' not found in {exp_dir}")
            continue

        try:
            df = pd.read_csv(summary_csv_path)
            if 'finetuned_roc_auc' not in df.columns or 'finetuned_roc_auc_extreme' not in df.columns:
                print(f"  -> Warning: Required roc_auc columns not found in {summary_csv_path}")
                continue
                
            df_processed = df[['subject_id', 'finetuned_roc_auc', 'finetuned_roc_auc_extreme']].copy()
            df_processed[category_col_name] = label
            all_subject_results.append(df_processed)
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            print(f"  -> Could not process file {summary_csv_path}: {e}")

    if not all_subject_results:
        print(f"Error: No valid experiment data was loaded for {scan_title}!")
        return None
    
    return pd.concat(all_subject_results, ignore_index=True)

def run_model_comparison_data_gen(results_root, grid_dir, dataset):
    """Runs the model comparison analysis and saves the data to CSV."""
    print("\n" + "="*50)
    print("### GENERATING MODEL COMPARISON DATA ###")
    print("="*50)

    config_tag = "FM-Full_A-None_AdaBN-F"
    models_to_load = OrderedDict([
        ("DeepTEPNet", "PRIME"), ("EEGNetv4", "EEGNet"), ("DeepConvNet", "Deep\nConvNet"),
        ("ATCNet", "ATCNet"), ("ShallowConvNet", "Shallow\nConvNet"),
        ("Ablation_NoS4", "PRIME\nw/o S4"), ("Ablation_ConvInsteadOfS4", "PRIME\nw/ Conv"),
    ])
    
    df_perf = load_all_models_performance(results_root, grid_dir, models_to_load, config_tag, dataset)
    
    if df_perf is not None and not df_perf.empty:
        output_filename = "model_comparison_data.csv"
        df_perf.to_csv(output_filename, index=False)
        print(f"\n✅ Model comparison data saved to {output_filename}")


def main():
    """Main execution function."""
    try:
        USERNAME = os.environ.get('USER', 'default_user') 
        REPO_ROOT = Path(f"/mnt/lustre/work/macke/{USERNAME}/repos/eegjepa")
        RESULTS_ROOT_DIR = REPO_ROOT / "EDAPT_neurips/EDAPT_TMS/results_final"
        if not RESULTS_ROOT_DIR.exists():
            print(f"Warning: Default results directory not found at {RESULTS_ROOT_DIR}")
            print("Please ensure RESULTS_ROOT_DIR is set correctly.")
            RESULTS_ROOT_DIR = Path("./results_final")
            # Create a dummy directory for demonstration if it doesn't exist
            RESULTS_ROOT_DIR.mkdir(exist_ok=True)
    except Exception:
        RESULTS_ROOT_DIR = Path("./results_final")
        RESULTS_ROOT_DIR.mkdir(exist_ok=True)

    GRID_DIR = "10ms_pp_w50"
    DATASET = "TMSEEGClassificationTEPfree"
    
    # --- Execution ---
    run_model_comparison_data_gen(RESULTS_ROOT_DIR, GRID_DIR, DATASET)

if __name__ == "__main__":
    main()
# %%