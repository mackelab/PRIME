#%%

import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# --- Helper function  ---
def to_binary(y, thresh=0.5):
    """Convert continuous labels to 0/1, returning None if only one class is present."""
    y_bin = (y >= thresh).astype(int)
    if y_bin.min() == y_bin.max():
        return None
    return y_bin

def generate_roc_over_time_data(paths, labels, output_path, window_size=100):
    """
    Calculates sliding window ROC-AUC for multiple data paths and saves the aggregated results to a CSV file.
    """
    print("--- Generating data for ROC-over-time plot ---")
    max_trials = 900
    all_conditions_df = []

    for path, label in zip(paths, labels):
        subject_files = {}
        search_pattern = os.path.join(path, '**', '*.npz')
        for f in glob.glob(search_pattern, recursive=True):
            match = re.search(r'subj_(\d+)_fold_(\d+)', os.path.basename(f))
            if match:
                subject_id = int(match.group(1))
                if subject_id not in subject_files:
                    subject_files[subject_id] = []
                subject_files[subject_id].append(f)

        if not subject_files:
            print(f"Warning: No subject data found in {path}. Skipping.")
            continue

        all_subject_curves = []
        print(f"Processing data for '{label}' from: {path}")
        for subject_id in tqdm(sorted(subject_files.keys()), desc=f"Subjects for '{label}'"):
            fold_curves = []
            for f in subject_files[subject_id]:
                with np.load(f) as data:
                    preds = data['predictions']
                    actuals = data['actual_values']

                if len(preds) > max_trials:
                    preds, actuals = preds[:max_trials], actuals[:max_trials]
                if len(preds) < window_size:
                    continue

                curve = []
                for t in range(window_size, len(preds) + 1):
                    labels_window = to_binary(actuals[t - window_size:t])
                    preds_window = preds[t - window_size:t]
                    roc = roc_auc_score(labels_window, preds_window) if labels_window is not None else np.nan
                    curve.append(roc)

                if curve:
                    fold_curves.append(curve)

            if fold_curves:
                max_len = max(len(c) for c in fold_curves)
                padded = [np.pad(c, (0, max_len - len(c)), 'constant', constant_values=np.nan) for c in fold_curves]
                all_subject_curves.append(np.nanmean(padded, axis=0))

        if not all_subject_curves:
            print(f"Warning: Could not compute any valid ROC curves for '{label}'. Skipping.")
            continue

        max_len_overall = max(len(c) for c in all_subject_curves)
        final_curves = np.array([np.pad(c, (0, max_len_overall - len(c)), 'constant', constant_values=np.nan) for c in all_subject_curves])

        median_curve = np.nanpercentile(final_curves, 50, axis=0)
        lower_quantile = np.nanpercentile(final_curves, 25, axis=0)
        upper_quantile = np.nanpercentile(final_curves, 75, axis=0)
        
        valid_indices = ~np.isnan(median_curve)
        
        df = pd.DataFrame({
            'trial_index': np.arange(window_size, window_size + len(median_curve[valid_indices])),
            'median_roc_auc': median_curve[valid_indices],
            'lower_quantile_roc_auc': lower_quantile[valid_indices],
            'upper_quantile_roc_auc': upper_quantile[valid_indices],
            'condition': label
        })
        all_conditions_df.append(df)

    final_df = pd.concat(all_conditions_df, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    print(f"Successfully saved ROC over time data to {output_path}")

def generate_calibration_data(csv_path_pretrain, csv_path_no_pretrain, output_path):
    """
    Loads, processes, and combines calibration/fine-tuning data from two sources
    and saves the result to a single CSV file.
    """
    print("\n--- Generating data for calibration violin plot ---")
    # 1. Load and process standard (pretrained) data
    try:
        df_pretrain = pd.read_csv(csv_path_pretrain)
    except FileNotFoundError:
        print(f"Error: Pretrained CSV file not found at {csv_path_pretrain}")
        return

    value_vars_pre = [
        'pre_calib_roc_auc', 'post_calib_roc_auc', 'finetuned_roc_auc',
        'pre_calib_roc_auc_extreme', 'post_calib_roc_auc_extreme', 'finetuned_roc_auc_extreme'
    ]
    df_long_pre = df_pretrain.melt(id_vars=['subject_id'], value_vars=value_vars_pre, var_name='metric', value_name='roc_auc')
    df_long_pre['variant'] = np.where(df_long_pre['metric'].str.contains('_extreme'), 'Extreme', 'All')
    df_long_pre['condition'] = df_long_pre['metric'].str.replace('_roc_auc_extreme', '').str.replace('_roc_auc', '')
    df_long_pre['condition'] = df_long_pre['condition'].replace({
        'pre_calib': 'PRE-ZS', 'post_calib': 'PRE-CAL', 'finetuned': 'PRE-FT'
    })

    # 2. Load and process "no pretrain" (from scratch) data
    try:
        df_no_pretrain = pd.read_csv(csv_path_no_pretrain)
    except FileNotFoundError:
        print(f"Error: 'No Pretrain' CSV file not found at {csv_path_no_pretrain}")
        df_combined = df_long_pre
    else:
        value_vars_no_pre = [
            'post_calib_roc_auc', 'finetuned_roc_auc',
            'post_calib_roc_auc_extreme', 'finetuned_roc_auc_extreme'
        ]
        df_long_no_pre = df_no_pretrain.melt(id_vars=['subject_id'], value_vars=value_vars_no_pre, var_name='metric', value_name='roc_auc')
        df_long_no_pre['variant'] = np.where(df_long_no_pre['metric'].str.contains('_extreme'), 'Extreme', 'All')
        df_long_no_pre['condition'] = df_long_no_pre['metric'].str.replace('_roc_auc_extreme', '').str.replace('_roc_auc', '')
        df_long_no_pre['condition'] = df_long_no_pre['condition'].replace({
            'post_calib': 'SS-CAL', 'finetuned': 'SS-FT'
        })
        df_combined = pd.concat([df_long_pre, df_long_no_pre], ignore_index=True)

    # 3. Save the combined dataframe
    df_combined.to_csv(output_path, index=False)
    print(f"Successfully saved calibration data to {output_path}")

def generate_scatter_data(path, output_path):
    """
    Creates a scatter plot of predicted vs. actual values from all subjects.
    """
    print("\n--- Generating data for predicted-vs-actual scatter plot ---")
    all_predictions, all_actuals = [], []
    search_pattern = os.path.join(path, '**', '*.npz')
    file_list = glob.glob(search_pattern, recursive=True)

    if not file_list:
        print(f"Error: No .npz files found in {path}")
        return

    print(f"Loading data for scatter plot from: {path}")
    for f in tqdm(file_list, desc="Loading files for scatter"):
        with np.load(f) as data:
            all_predictions.append(data['predictions'])
            all_actuals.append(data['actual_values'])

    if not all_predictions:
        print("Error: No data could be loaded for the scatter plot.")
        return

    df = pd.DataFrame({
        'predicted_values': np.concatenate(all_predictions),
        'actual_values': np.concatenate(all_actuals)
    })
    
    df.to_csv(output_path, index=False)
    print(f"Successfully saved scatter plot data to {output_path}")

def main():
    """Main function to generate and save the data."""
    # --- IMPORTANT: Source data paths from the original script ---
    PRIME_PATH = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/results_final/10ms_pp_w50/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F"
    RANDOM_PATH = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/results/10ms_pp_w50_shuffled/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F"
    CSV_PATH_PRETRAIN = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/results_final/10ms_pp_w50/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F/20250704_060709/results_summary_detailed.csv"
    CSV_PATH_NO_PRETRAIN = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/results_final/10ms_pp_w50_no_pretrain/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F_no_pretrain/AlignEval_DeepTEPNet_TMSEEGClassificationTEPfree_FM-Full_A-None_AdaBN-F_no_pretrain/20250704_060839/results_summary_detailed.csv"
    
    # --- Define output paths for the generated CSV files ---
    OUTPUT_DIR = "reproduction_data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    OUTPUT_CSV_ROC = os.path.join(OUTPUT_DIR, "roc_over_time_data.csv")
    OUTPUT_CSV_CALIBRATION = os.path.join(OUTPUT_DIR, "calibration_data.csv")
    OUTPUT_CSV_SCATTER = os.path.join(OUTPUT_DIR, "predicted_vs_actual_data.csv")

    # --- Run data generation functions ---
    generate_roc_over_time_data(
        paths=[PRIME_PATH, RANDOM_PATH],
        labels=['PRIME', 'Random control'],
        output_path=OUTPUT_CSV_ROC,
        window_size=100
    )
    
    generate_calibration_data(
        csv_path_pretrain=CSV_PATH_PRETRAIN,
        csv_path_no_pretrain=CSV_PATH_NO_PRETRAIN,
        output_path=OUTPUT_CSV_CALIBRATION
    )
    
    generate_scatter_data(
        path=PRIME_PATH,
        output_path=OUTPUT_CSV_SCATTER
    )
    
    print("\nAll data generation complete.")

if __name__ == "__main__":
    main()
# %%
