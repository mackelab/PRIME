import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List
import logging
import submitit

# --- Basic Configuration ---
# Configure logging for better output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Core Paths and Settings ---
# Get username for constructing paths
USER = os.environ.get('USER', 'default_user')
REPO_DIR = Path(f"/mnt/lustre/work/macke/{USER}/repos/eegjepa")
SCRIPT_TO_RUN = REPO_DIR / "EDAPT_neurips/EDAPT_TMS/preprocessing/single_trial_dipole_amplitude_slurm.py"

# --- Data and Logging Directories ---
# Input directory containing the processed EEG data (e.g., 'sub-001', 'sub-002')
PROCESSED_EEG_DIR = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/preprocessing/data_processed_final_pre_ica_False_final_v4")
# Base directory for SLURM log files for this script
BASE_LOG_DIR = REPO_DIR / "slurm_logs_dipole"

# --- SLURM Job Configuration ---
SLURM_PARTITION = "cpu-galvani"  # Verify your cluster's partition name
MEM_GB_PER_JOB = 32              # Memory per job in GB
CPUS_PER_JOB = 1                 # Number of CPUs per job
SUBJECTS_PER_JOB = 2             # Number of subjects to process sequentially in one job
JOB_TIME = "0-08:00:00"          # Job time limit (D-HH:MM:SS), e.g., 8 hours

# --- Utility and Job Execution Functions ---

def discover_subjects(data_dir: Path) -> List[str]:
    """Scans the data directory and returns a sorted list of subject IDs."""
    subjects = []
    if not data_dir.is_dir():
        logging.error(f"Data directory not found at {data_dir}")
        return []
    for subject_dir in data_dir.iterdir():
        if subject_dir.is_dir():
            subjects.append(subject_dir.name)
    return sorted(subjects)

def chunk_list(data: list, size: int):
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

def run_sequential_dipole_fitting(subject_chunk: List[str]):
    """
    Executes the dipole fitting script sequentially for a chunk of subjects.
    This function is what runs on the SLURM node.
    """
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    logging.info(f"--- SLURM Job {job_id} starting ---")
    logging.info(f"Processing {len(subject_chunk)} subjects in this job: {subject_chunk}")

    for i, subject_id in enumerate(subject_chunk):
        logging.info(f"--- [{i+1}/{len(subject_chunk)}] Processing subject: {subject_id} ---")

        command = [
            "python", "-u", str(SCRIPT_TO_RUN),
            "--subject", subject_id
        ]
        logging.info(f"  - Executing command: {' '.join(command)}")

        try:
            # Run the script and capture output
            subprocess.run(
                command,
                check=True,         # This will raise CalledProcessError on failure
                capture_output=True,
                text=True,
                env=os.environ.copy()
            )
            logging.info(f"  - SUCCESSFULLY finished processing {subject_id}")

        except subprocess.CalledProcessError as e:
            # This is the crucial part: Log the actual error from the failing script
            logging.error(f"--- SCRIPT FAILED for subject {subject_id} ---")
            logging.error(f"--- Captured STDOUT ---\n{e.stdout}")
            logging.error(f"--- Captured STDERR ---\n{e.stderr}")
            # Re-raise the exception to ensure the SLURM job is marked as failed
            raise e

    return f"Finished job {job_id}, successfully processed {len(subject_chunk)} subjects."

# --- Main Execution Block ---

def main():
    """Main function to discover subjects and submit SLURM jobs."""
    parser = argparse.ArgumentParser(description="Submit sequential dipole fitting jobs to SLURM.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job configurations and commands without submitting to SLURM."
    )
    cli_args = parser.parse_args()

    all_subjects = discover_subjects(PROCESSED_EEG_DIR)
    if not all_subjects:
        logging.info("No subjects found in the specified directory. Exiting.")
        return

    subject_chunks = list(chunk_list(all_subjects, SUBJECTS_PER_JOB))
    total_jobs = len(subject_chunks)

    logging.info(f"Discovered {len(all_subjects)} subjects.")
    logging.info(f"Grouping into {total_jobs} jobs with up to {SUBJECTS_PER_JOB} subjects each.")

    if cli_args.dry_run:
        print("\n--- DRY RUN: Job Configurations (not submitting) ---")
        for i, chunk in enumerate(subject_chunks[:3]): # Show first 3 jobs as an example
            print(f"\n--- Job {i+1}/{total_jobs} would process: {chunk} ---")
            for subject_id in chunk:
                print(f"  - Command: python {SCRIPT_TO_RUN} --subject {subject_id}")
        sys.exit(0)

    # Setup the submitit executor
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_folder = BASE_LOG_DIR / f"{date_str}_dipole_run"
    executor = submitit.AutoExecutor(folder=str(log_folder))

    executor.update_parameters(
        slurm_partition=SLURM_PARTITION,
        slurm_time=JOB_TIME,
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=CPUS_PER_JOB,
        mem_gb=MEM_GB_PER_JOB,
    )

    logging.info(f"\nSubmitting {total_jobs} jobs to SLURM...")
    
    # Initialize a list to hold tuples of (name, job_object)
    jobs_with_names = [] 
    with executor.batch():
        for i, chunk in enumerate(subject_chunks):
            job_name = f"dipole_batch_{i+1}"
            job = executor.submit(run_sequential_dipole_fitting, chunk)
            # Append the tuple to the list
            jobs_with_names.append((job_name, job)) 
            logging.info(f"  - Queued Job {i+1}/{total_jobs}: {job_name}")

    print(f"\n--- All {total_jobs} jobs submitted successfully! ---")
    # Iterate through the list of tuples to access both name and job
    for name, job in jobs_with_names:
        print(f"  - Job Name: {name}, SLURM ID: {job.job_id}")
    print(f"\nSLURM logs will be stored in: {log_folder}")

if __name__ == "__main__":
    main()