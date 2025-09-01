#%%
"""
This script computes the preprocessing latencies for pre- and post-stimulus EEG processing.
"""
import numpy as np
import glob
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def sanitize_filename(name):
    """Removes characters that are invalid for filenames."""
    name = name.replace("'", "").replace(" ", "_").replace(":", "")
    return "".join(c for c in name if c.isalnum() or c in ('_', '.')).rstrip()

def analyze_detailed_latencies(base_dir, dipole_base_dir, output_plot_dir):
    """
    Loads and combines TEP latencies, calculates detailed statistics, and generates
    histograms for the main summary categories.
    """
    output_plot_dir = Path(output_plot_dir)
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"📊 Saving plots to: {output_plot_dir.resolve()}")

    # --- Part 1: Analyze original preprocessing latencies ---
    search_pattern = os.path.join(base_dir, 'sub-*', '*_preprocessing_info.npz')
    subject_files = glob.glob(search_pattern)
    subject_files = [f for f in subject_files if '_rep' not in os.path.basename(f)]
    if not subject_files: print(f"No subject files found: {search_pattern}")
    else: print(f"Analyzing {len(subject_files)} subject files for main preprocessing.")
    detailed_latencies = {'pre': defaultdict(list), 'post': defaultdict(list), 'emg': defaultdict(list)}
    pre_total_latencies, post_total_latencies, emg_total_latencies = [], [], []
    key_steps = {
        'pre': ['1. Reconstruct Bad Channels', '2.2 Apply Filter', '2.3 Mean Subtraction', '3.1 Global MAD Check', '3.2 Local MAD Check'],
        'post': ['2. Pulse Artifact Interpolation 1', '3. Reconstruct Bad Channels', '4. Set EEG Reference', '5. ICA Ocular Check', '6. Apply ICA', '7. Re-baseline & Crop', '8. Apply SOUND', '9. Apply SSP-SIR'],
        'emg': ['2. Apply EMG Filter', '3. Pre-Innervation & MEP Check']
    }
    for file_path in subject_files:
        try:
            with np.load(file_path, allow_pickle=True) as data:
                processing_times = data['processing_times'].item()
                for stage in ['pre', 'post', 'emg']:
                    for trial_time_info in processing_times.get(stage, {}).get('all', []):
                        current_trial_total = 0
                        for step, latency in trial_time_info.items():
                            if isinstance(latency, (int, float)) and latency > 0:
                                detailed_latencies[stage][step].append(latency)
                                if step in key_steps[stage]: current_trial_total += latency
                        if current_trial_total > 0:
                            if stage == 'pre': pre_total_latencies.append(current_trial_total)
                            elif stage == 'post': post_total_latencies.append(current_trial_total)
                            elif stage == 'emg': emg_total_latencies.append(current_trial_total)
        except Exception as e: print(f"Could not process file {file_path}: {e}")

    # --- Part 2: Analyze TEP extraction latencies ---
    dipole_search_pattern = os.path.join(dipole_base_dir, 'sub-*', '*_response_extraction_info.npz')
    dipole_files = glob.glob(dipole_search_pattern)
    dipole_files = [f for f in dipole_files if '_rep' not in os.path.basename(f)]
    if not dipole_files: print(f"No dipole files found: {dipole_search_pattern}")
    else: print(f"Analyzing {len(dipole_files)} subject files for TEP extraction.")
    tep_free_ori_latencies = []
    for file_path in dipole_files:
        try:
            with np.load(file_path, allow_pickle=True) as data:
                if 'dipoles_free_ori_extraction_times' in data: tep_free_ori_latencies.extend(data['dipoles_free_ori_extraction_times'] * 1000)
        except Exception as e: print(f"Could not process dipole file {file_path}: {e}")

    # --- Part 2.5: Combine TEP latencies for a full analysis pipeline ---
    n_tep_extraction = len(post_total_latencies)
    n_dipole_fitting = len(tep_free_ori_latencies)
    min_trials = 0
    
    if n_tep_extraction > 0 and n_dipole_fitting > 0:
        if n_tep_extraction != n_dipole_fitting:
            print(f"\n⚠️ Warning: Mismatched trial counts for TEP stages.")
            print(f"  - TEP Extraction trials: {n_tep_extraction}")
            print(f"  - TEP Dipole Fitting trials: {n_dipole_fitting}")
            min_trials = min(n_tep_extraction, n_dipole_fitting)
            print(f"  Combining the first {min_trials} trials from each stage for a total.")
        else:
            min_trials = n_tep_extraction
        
        # Combine the latencies by adding them element-wise for the common trials
        combined_tep_latencies = (np.array(post_total_latencies[:min_trials]) + 
                                  np.array(tep_free_ori_latencies[:min_trials]))
    else:
        combined_tep_latencies = []
        print("\n⚠️ Warning: Could not combine TEP latencies as one of the stages has no data.")


    # --- Part 3: Print stats and generate SPECIFIC plots ---
    summaries_to_plot = [
        "Pre-stimulus Total",
        "Post-stimulus Full TEP Analysis", 
        "Post-stimulus MEP Total"
    ]
    
    def print_and_plot_stats(title, latencies, unit='ms', output_dir=None, plot=False):
        # (Function is unchanged)
        if not latencies.any():
            print(f"\n--- {title} ---\nNo latency data available.")
            return
        latencies = np.array(latencies)
        print(f"\n--- {title} (from {len(latencies)} trials) ---")
        print(f"  Mean            : {np.mean(latencies):.3f} {unit}")
        print(f"  Std Dev         : {np.std(latencies):.3f} {unit}")
        print(f"  Median          : {np.median(latencies):.3f} {unit}")
        print(f"  Min             : {np.min(latencies):.3f} {unit}")
        print(f"  99th Percentile : {np.percentile(latencies, 99):.3f} {unit}")
        print(f"  Max (worst case): {np.max(latencies):.3f} {unit}")
        if plot and output_dir:
            plt.figure(figsize=(10, 6))
            bins = min(len(latencies) // 10, 100) if len(latencies) > 20 else 15
            plt.hist(latencies, bins=bins, color='skyblue', edgecolor='black')
            plt.title(f'Latency Distribution for {title}', fontsize=16)
            plt.xlabel(f'Latency ({unit})', fontsize=12)
            plt.ylabel('Number of Trials', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(latencies):.2f}')
            plt.axvline(np.median(latencies), color='darkorange', linestyle='-', linewidth=2, label=f'Median: {np.median(latencies):.2f}')
            plt.axvline(np.percentile(latencies, 99), color='purple', linestyle=':', linewidth=2.5, label=f'99th percentile: {np.percentile(latencies, 99):.2f}')
            plt.legend()
            filepath = Path(output_dir) / (sanitize_filename(title) + '.pdf')
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            plt.close()

    # --- Updated Summary Report ---
    # This dictionary now holds the final data sets for reporting
    summary_data = {
        "Pre-stimulus Total": np.array(pre_total_latencies),
        "Post-stimulus Full TEP Analysis": combined_tep_latencies,
        "Post-stimulus MEP Total": np.array(emg_total_latencies)
    }

    print(f"\n\n{'='*25} SUMMARY LATENCY ANALYSIS {'='*25}")
    for title, data in summary_data.items():
        should_plot = title in summaries_to_plot
        print_and_plot_stats(title, data, unit='ms', output_dir=output_plot_dir, plot=should_plot)

if __name__ == '__main__':
    base_data_directory = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/preprocessing/data_processed_final_pre_ica_False_final_v4")
    dipole_data_directory = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/preprocessing/dipoles_with_calibration_final_v4")
    plot_output_directory = Path("./latency_analysis_plots_summaries_only")
    
    analyze_detailed_latencies(base_data_directory, dipole_data_directory, plot_output_directory)

# %%