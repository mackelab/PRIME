# %%
"""
Datasets Module for TMS-EEG Classification.

Core Components:
- `EEGDataset`: A PyTorch-compatible Dataset class.
- `load_cached_pretrain_data`: Main function to load, preprocess, align, and
  concatenate data from multiple subjects.
- Caching Paradigms: `CachingTMSEEGClassification`, `CachingTMSEEGClassificationTEP`,
  and `CachingTMSEEGClassificationTEPfree` handle the specifics of data
  extraction and caching for different TMS-EEG data types.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from moabb.datasets.base import BaseDataset
from torch.utils.data import Dataset

# --- Local Imports ---
import utils
from TMS_EEG_moabb import (
    TMSEEGClassification,
    TMSEEGClassificationTEP,
    TMSEEGClassificationTEPfree,
    TMSEEGDataset,
    TMSEEGDatasetTEP,
    TMSEEGDatasetTEPfree,
)
from tta_wrapper import (
    PYRIEMANN_AVAILABLE,
    _apply_alignment_transform_np,
    _compute_alignment_transform_np,
    _compute_reference_covariance_np,
    _compute_trial_covariances_np,
)

# --- Module-level Logger Setup ---
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# %%
# ----------------------------------------------------------------------------
# PARADIGM AND DATASET CONFIGURATION
# ----------------------------------------------------------------------------

PARADIGM_DATA = {
    "CUSTOM_CLS": {
        "datasets": [
            "TMSEEGClassification",
            "TMSEEGClassificationTEP",
            "TMSEEGClassificationTEPfree",
        ],
        "class_map": {
            "TMSEEGClassification": TMSEEGDataset,
            "TMSEEGClassificationTEP": TMSEEGDatasetTEP,
            "TMSEEGClassificationTEPfree": TMSEEGDatasetTEPfree,
        },
        "specs": {
            "TMSEEGClassification": dict(sr=1000, sec=0.995, n_cls=2),
            "TMSEEGClassificationTEP": dict(sr=1000, sec=0.995, n_cls=2),
            "TMSEEGClassificationTEPfree": dict(sr=1000, sec=0.995, n_cls=2),
        },
    },
}

# %%
class EEGDataset(Dataset):
    """A PyTorch-compatible dataset for EEG epochs."""

    def __init__(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        channel_names: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset.
        """
        self.epochs = epochs
        self.labels = labels
        self.channel_names = channel_names

    def __len__(self) -> int:
        """Return the total number of trials in the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a single trial and its associated label.
        """
        sample = {
            "epoch": self.epochs[index],
            "label": self.labels[index],
        }
        if self.channel_names is not None:
            sample["channel_names"] = self.channel_names
        return sample


# %%
# DATA LOADING AND MANAGEMENT FUNCTIONS

def get_subject_list_for_datasets(
    dataset_names: List[str], data_root: str
) -> List[int]:
    """
    Retrieves a sorted list of subject IDs for a given TMS-EEG dataset.
    Note: This function is designed to operate on a single dataset at a time.
    """
    # Configure MNE data paths to prevent race conditions in parallel environments
    os.environ["MNE_DATA"] = data_root
    os.environ["MNE_DATASETS_MOABB_PATH"] = data_root

    if not dataset_names or len(dataset_names) != 1:
        log.error("This function requires a list with exactly one dataset name.")
        return []

    dataset_name = dataset_names[0]
    paradigm_info = PARADIGM_DATA.get("CUSTOM_CLS")

    if not paradigm_info or dataset_name not in paradigm_info["class_map"]:
        log.error(f"Dataset '{dataset_name}' is not configured in PARADIGM_DATA.")
        return []

    dataset_class = paradigm_info["class_map"][dataset_name]

    try:
        dataset_instance = dataset_class()
        subjects = sorted(dataset_instance.subject_list)
        log.info(f"Found {len(subjects)} subjects in dataset '{dataset_name}'.")
        return subjects
    except Exception as e:
        log.error(f"Failed to retrieve subjects for '{dataset_name}': {e}")
        return []


def load_cached_pretrain_data(
    dataset_names: List[str],
    subject_ids: List[int],
    paradigm_kwargs: Dict[str, Any],
    data_root: str,
    args: Any,
    apply_trial_ablation: bool = False,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[int],
    Optional[int],
    Optional[List[str]],
    Optional[np.ndarray],
]:
    """ 
    Loads, processes, and optionally aligns EEG data for pre-training.
    """
    if not dataset_names or len(dataset_names) != 1:
        log.error("Function requires a list with exactly one dataset name.")
        return None, None, None, None, None, None

    # --- Setup and Configuration ---
    dataset_name = dataset_names[0]
    os.environ["MNE_DATA"] = data_root
    os.environ["MNE_DATASETS_MOABB_PATH"] = data_root

    paradigm_class_map = {
        "TMSEEGClassification": CachingTMSEEGClassification,
        "TMSEEGClassificationTEP": CachingTMSEEGClassificationTEP,
        "TMSEEGClassificationTEPfree": CachingTMSEEGClassificationTEPfree,
    }
    paradigm_class = paradigm_class_map.get(dataset_name)

    if paradigm_class is None:
        log.error(f"No caching paradigm found for dataset: {dataset_name}")
        return None, None, None, None, None, None

    # --- Data Loading ---
    effective_paradigm_kwargs = paradigm_kwargs.copy()
    if apply_trial_ablation and hasattr(args, "num_trials_per_subject"):
        num_trials = args.num_trials_per_subject
        effective_paradigm_kwargs["num_trials_per_subject"] = num_trials
        log.info(f"Applying trial ablation: {num_trials} trials per subject.")

    try:
        paradigm = paradigm_class(**effective_paradigm_kwargs)
        dataset_class = PARADIGM_DATA["CUSTOM_CLS"]["class_map"][dataset_name]
        dataset = dataset_class()
    except Exception as e:
        log.error(f"Failed to instantiate paradigm or dataset: {e}", exc_info=True)
        return None, None, None, None, None, None

    available_subjects = dataset.subject_list
    subjects_to_load = [s for s in subject_ids if s in available_subjects]
    if not subjects_to_load:
        log.warning(f"No requested subjects found in dataset {dataset_name}.")
        return None, None, None, None, None, None

    log.info(f"Loading data for {len(subjects_to_load)} subjects from {dataset_name}.")
    epochs_data, y_values, _ = paradigm.get_data(
        dataset=dataset, subjects=subjects_to_load
    )

    if epochs_data is None or epochs_data.size == 0:
        log.warning(f"No data returned from paradigm for {dataset_name}.")
        return None, None, None, None, None, None

    # --- Label and Data Shape Processing ---
    labels_numeric = np.array(y_values).astype(np.float32)
    log.info(f"Using probabilistic float labels (dtype: {labels_numeric.dtype}).")
    _, n_channels, n_timepoints = epochs_data.shape

    if hasattr(paradigm, "channels") and paradigm.channels:
        channel_names = paradigm.channels
    else:
        channel_names = [f"Ch{i+1}" for i in range(n_channels)]

    all_epochs_list = [epochs_data]
    all_labels_list = [labels_numeric]

    # --- Euclidean Alignment (EA) Processing ---
    use_alignment = (
        hasattr(args, "use_tta")
        and args.use_tta
        and hasattr(args, "alignment_type")
        and args.alignment_type != "none"
    )

    if not use_alignment:
        log.info("Skipping Euclidean Alignment.")
        concatenated_epochs = np.concatenate(all_epochs_list)
        concatenated_labels = np.concatenate(all_labels_list)
        return (
            concatenated_epochs,
            concatenated_labels,
            n_channels,
            n_timepoints,
            channel_names,
            None,
        )

    log.info(f"Applying per-subject alignment (type: {args.alignment_type})")
    global_backrot_np = None
    do_backrot = getattr(args, "ea_backrotation", False)

    # 1. Compute global back-rotation matrix if requested
    if do_backrot:
        all_trial_covs = [
            _compute_trial_covariances_np(_subj_epochs, args.alignment_cov_epsilon)
            for _subj_epochs in all_epochs_list
            if _subj_epochs.size
        ]
        if all_trial_covs:
            trial_covs_np = np.concatenate(all_trial_covs, axis=0)
            align_type = args.alignment_type
            if align_type == "riemannian" and not PYRIEMANN_AVAILABLE:
                align_type = "euclidean"

            Sigma_global_np = _compute_reference_covariance_np(
                trial_covs_np, alignment_type=align_type
            )
            eps_glob = max(
                args.alignment_cov_epsilon, np.trace(Sigma_global_np) * 1e-6
            )
            Sigma_global_np += eps_glob * np.eye(Sigma_global_np.shape[0])
            eigvals, eigvecs = np.linalg.eigh(Sigma_global_np)
            global_backrot_np = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # 2. Apply alignment to each subject's data block
    aligned_epochs_list = []
    # Note: Currently, data is loaded as a single block. This loop supports
    # future extensions where data might be loaded per-subject.
    for subject_epochs_np in all_epochs_list:
        if subject_epochs_np.size == 0:
            continue
        try:
            # Subject-level whitening: Σ_s^{-½}
            source_trial_covs = _compute_trial_covariances_np(
                subject_epochs_np, args.alignment_cov_epsilon
            )
            Sigma_s_np = _compute_reference_covariance_np(
                source_trial_covs, alignment_type=args.alignment_type
            )
            R_s_neg_half = _compute_alignment_transform_np(
                Sigma_s_np, args.alignment_transform_epsilon
            )
            aligned_subject_epochs = _apply_alignment_transform_np(
                subject_epochs_np, R_s_neg_half
            )

            # Optional back-rotation: Σ_global^{+½}
            if do_backrot and global_backrot_np is not None:
                aligned_subject_epochs = _apply_alignment_transform_np(
                    aligned_subject_epochs, global_backrot_np
                )
            aligned_epochs_list.append(aligned_subject_epochs)

        except Exception as e:
            log.warning(f"Alignment failed; using original data. Error: {e}")
            aligned_epochs_list.append(subject_epochs_np)

    log.info("Pretraining alignment complete.")

    # --- Final Concatenation ---
    try:
        concatenated_epochs = (
            np.concatenate(aligned_epochs_list) if aligned_epochs_list else np.array([])
        )
        concatenated_labels = (
            np.concatenate(all_labels_list) if all_labels_list else np.array([])
        )
        log.info(
            f"Final Shapes: Epochs {concatenated_epochs.shape}, Labels {concatenated_labels.shape}"
        )
    except ValueError as e_concat:
        log.error(f"Failed to concatenate aligned data: {e_concat}", exc_info=True)
        return None, None, None, None, None, None

    return (
        concatenated_epochs,
        concatenated_labels,
        n_channels,
        n_timepoints,
        channel_names,
        global_backrot_np,
    )


# %%
# CUSTOM CACHING PARADIGM CLASSES
class BaseCachingParadigm:
    """A base class for caching paradigms to reduce code duplication."""

    def __init__(
        self,
        data_type: str,
        num_trials_per_subject: Optional[int] = None,
        **kwargs,
    ):
        """Initializes the base caching paradigm."""
        self._data_type = data_type
        self._num_trials_per_subject = num_trials_per_subject
        self.paradigm_kwargs = kwargs  # Store kwargs for caching

    def _get_cache_path(self, dataset: BaseDataset, subject: int) -> Path:
        """Constructs a unique file path for the cache based on parameters."""
        param_dict = self.paradigm_kwargs.copy()
        param_dict["num_trials"] = (
            str(self._num_trials_per_subject)
            if self._num_trials_per_subject is not None
            else "all"
        )
        param_str = "_".join(f"{k}={v}" for k, v in sorted(param_dict.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        cache_dir = utils.CACHE_ROOT_DIR / dataset.code / f"subject_{subject:03d}"
        return cache_dir / f"params_{param_hash}.npz"

    def get_data(
        self, dataset: BaseDataset, subjects: Optional[List[int]] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Retrieves data, using cache if available, otherwise computes and saves.
        This method is intended to be called by the inheriting child class.
        """
        if subjects is None:
            subjects = dataset.subject_list

        all_X, all_y, all_meta = [], [], []

        for subject in subjects:
            cache_path = self._get_cache_path(dataset, subject)

            if cache_path.exists():
                log.info(
                    f"Loading cached {self._data_type} data for Subj {subject} from {cache_path}"
                )
                cached_data = np.load(cache_path, allow_pickle=True)
                X, y, metadata = (
                    cached_data["X"],
                    cached_data["y"],
                    pd.DataFrame(cached_data["metadata"].item()),
                )
            else:
                log.info(
                    f"Cache miss for {self._data_type} Subj {subject}. Computing data."
                )
                X, y, metadata = super().get_data(dataset=dataset, subjects=[subject])

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, X=X, y=y, metadata=metadata.to_dict())
                log.info(
                    f"Saved computed {self._data_type} data for Subj {subject} to {cache_path}"
                )

            if self._num_trials_per_subject is not None and len(
                X
            ) > self._num_trials_per_subject:
                log.info(
                    f"Slicing {self._data_type} data for Subj {subject} to {self._num_trials_per_subject} trials."
                )
                X = X[: self._num_trials_per_subject]
                y = y[: self._num_trials_per_subject]
                metadata = metadata.iloc[: self._num_trials_per_subject]

            all_X.append(X)
            all_y.append(y)
            all_meta.append(metadata)

        final_X = np.concatenate(all_X) if all_X else np.array([])
        final_y = np.concatenate(all_y) if all_y else np.array([])
        final_meta = (
            pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
        )

        return final_X, final_y, final_meta


class CachingTMSEEGClassification(BaseCachingParadigm, TMSEEGClassification):
    """Caching wrapper for the TMSEEGClassification (MEP) paradigm."""

    def __init__(self, num_trials_per_subject: Optional[int] = None, **kwargs):
        super(CachingTMSEEGClassification, self).__init__(
            data_type="MEP",
            num_trials_per_subject=num_trials_per_subject,
            **kwargs,
        )
        # Initialize the actual paradigm logic
        TMSEEGClassification.__init__(self, **kwargs)


class CachingTMSEEGClassificationTEP(BaseCachingParadigm, TMSEEGClassificationTEP):
    """Caching wrapper for the TMSEEGClassificationTEP (TEP) paradigm."""

    def __init__(self, num_trials_per_subject: Optional[int] = None, **kwargs):
        super(CachingTMSEEGClassificationTEP, self).__init__(
            data_type="TEP",
            num_trials_per_subject=num_trials_per_subject,
            **kwargs,
        )
        # Initialize the actual paradigm logic
        TMSEEGClassificationTEP.__init__(self, **kwargs)


class CachingTMSEEGClassificationTEPfree(
    BaseCachingParadigm, TMSEEGClassificationTEPfree
):
    """Caching wrapper for the TMSEEGClassificationTEPfree paradigm."""

    def __init__(self, num_trials_per_subject: Optional[int] = None, **kwargs):
        super(CachingTMSEEGClassificationTEPfree, self).__init__(
            data_type="TEPfree",
            num_trials_per_subject=num_trials_per_subject,
            **kwargs,
        )
        # Initialize the actual paradigm logic
        TMSEEGClassificationTEPfree.__init__(self, **kwargs)


# %%
# MODULE EXPORTS

__all__ = [
    "EEGDataset",
    "get_subject_list_for_datasets",
    "load_cached_pretrain_data",
    "CachingTMSEEGClassification",
    "CachingTMSEEGClassificationTEP",
    "CachingTMSEEGClassificationTEPfree",
    "PARADIGM_DATA",
]
# %%
