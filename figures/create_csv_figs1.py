#%%
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Loading Function  ---

def load_all_latency_data(base_dir, dipole_dir, model_csv_path):
    """
    Loads and prepares all latency data required for the multi-panel plot.
    """
    logging.info("--- Loading Preprocessing & TEP Latencies ---")
    
    key_steps = {
        'pre': ['1. Reconstruct Bad Channels', '2.2 Apply Filter', '2.3 Mean Subtraction', '3.1 Global MAD Check', '3.2 Local MAD Check'],
        'post': ['2. Pulse Artifact Interpolation 1', '3. Reconstruct Bad Channels', '4. Set EEG Reference', '5. ICA Ocular Check', '6. Apply ICA', '7. Re-baseline & Crop', '8. Apply SOUND', '9. Apply SSP-SIR'],
    }
    
    # 1. Load Pre-stimulus and Post-stimulus latencies
    search_pattern = os.path.join(base_dir, 'sub-*', '*_preprocessing_info.npz')
    subject_files = glob.glob(search_pattern)
    subject_files = [f for f in subject_files if '_rep' not in os.path.basename(f)]
    pre_total_latencies, post_total_latencies = [], []
    
    for file_path in subject_files:
        with np.load(file_path, allow_pickle=True) as data:
            times = data['processing_times'].item()
            for trial_info in times.get('pre', {}).get('all', []):
                trial_total = sum(latency for step, latency in trial_info.items() if step in key_steps['pre'])
                if trial_total > 0:
                    pre_total_latencies.append(trial_total)
            for trial_info in times.get('post', {}).get('all', []):
                trial_total = sum(latency for step, latency in trial_info.items() if step in key_steps['post'])
                if trial_total > 0:
                    post_total_latencies.append(trial_total)

    logging.info(f"Found and calculated latencies for {len(pre_total_latencies)} pre-stimulus trials.")
    logging.info(f"Found and calculated latencies for {len(post_total_latencies)} post-stimulus trials.")

    # 2. Load and combine Post-stimulus TEP latencies
    dipole_search_pattern = os.path.join(dipole_dir, 'sub-*', '*_response_extraction_info.npz')
    dipole_files = glob.glob(dipole_search_pattern)
    tep_free_ori_latencies = []
    for file_path in dipole_files:
        with np.load(file_path, allow_pickle=True) as data:
            if 'dipoles_free_ori_extraction_times' in data:
                tep_free_ori_latencies.extend(data['dipoles_free_ori_extraction_times'] * 1000)
    
    min_trials = min(len(post_total_latencies), len(tep_free_ori_latencies))
    combined_tep_latencies = (np.array(post_total_latencies[:min_trials]) + 
                              np.array(tep_free_ori_latencies[:min_trials]))
    logging.info(f"Found and combined {len(combined_tep_latencies)} full TEP processing trials.")

    # 3. Load Model latencies from CSV 
    logging.info("\n--- Loading Model Latencies ---")
    try:
        df = pd.read_csv(model_csv_path)
        logging.info(f"✅ Successfully loaded model latency data from: {model_csv_path}")
        model_latencies = {
            'pred_gpu': df[df['device'] == 'cuda']['pred_latency_ms_median'].dropna().values,
            'pred_cpu': df[df['device'] == 'cpu']['pred_latency_ms_median'].dropna().values,
            'adapt_gpu': df[df['device'] == 'cuda']['finetune_epoch_ms_median'].dropna().values,
            'adapt_cpu': df[df['device'] == 'cpu']['finetune_epoch_ms_median'].dropna().values
        }
    except FileNotFoundError:
        # This is the fix: Raise a clear error instead of silently continuing.
        error_message = (
            f"\n\n========================================================================\n"
            f"FATAL ERROR: The model latency CSV file was not found at the specified path:\n"
            f"'{model_csv_path.resolve()}'\n\n"
            f"Please ensure this path is correct relative to where you are running the script, "
            f"or provide an absolute path.\n"
            f"========================================================================\n"
        )
        raise FileNotFoundError(error_message)

    # 4. Collate all data
    return {
        'pre_stimulus': np.array(pre_total_latencies),
        'tep_processing': np.array(combined_tep_latencies),
        **model_latencies
    }

def save_data_to_csv(latency_data, output_path):
    """Converts the latency data dictionary to a long-format DataFrame and saves it."""
    long_format_list = []
    for category, values in latency_data.items():
        if values is not None and len(values) > 0:
            for value in values:
                long_format_list.append({'category': category, 'latency_ms': value})
    
    df = pd.DataFrame(long_format_list)
    df.to_csv(output_path, index=False)
    logging.info(f"✅ All latency data saved successfully to {output_path}")


def main():
    """Main function to load data and save it to a single CSV."""
    # --- IMPORTANT: Please verify these paths are correct for your system ---
    base_data_directory = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/preprocessing/data_processed_final_pre_ica_False_final_v4")
    dipole_data_directory = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/preprocessing/dipoles_with_calibration_final_v4")
    model_latency_file = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/results_latency/collated_latency_summary_per_subject.csv")
    output_dir = Path("./latency_figure_final")
    output_dir.mkdir(exist_ok=True)
    
    latency_data = load_all_latency_data(base_data_directory, dipole_data_directory, model_latency_file)
    
    output_csv_file = output_dir / "latency_data_for_plotting.csv"
    save_data_to_csv(latency_data, output_csv_file)


if __name__ == "__main__":
    main()
# %%
