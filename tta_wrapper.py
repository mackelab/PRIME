#%%
"""
Test-time adaptation wrapper for EEG classification models
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import eigh

try:
    from pyriemann.utils.mean import mean_riemann  # type: ignore

    PYRIEMANN_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYRIEMANN_AVAILABLE = False

log = logging.getLogger(__name__)

# Utility functions (unchanged)
def _compute_trial_covariances_np(trials_np: np.ndarray, cov_epsilon: float = 1e-6) -> np.ndarray:
    """Return per‑trial covariance matrices with Tikhonov regularisation."""
    n_trials, n_channels, _ = trials_np.shape
    covs = np.empty((n_trials, n_channels, n_channels))
    for i, trial in enumerate(trials_np):
        cov = np.cov(trial)
        trace = np.trace(cov)
        cov += max(cov_epsilon, trace * 1e-6) * np.eye(n_channels)
        covs[i] = cov
    return covs


def _compute_reference_covariance_np(trial_covs_np: np.ndarray, *, alignment_type: str = "euclidean") -> np.ndarray:
    """Mean covariance across trials using Euclidean or Riemannian metric."""
    if alignment_type == "riemannian":
        if not PYRIEMANN_AVAILABLE:
            raise ImportError("pyriemann required for Riemannian alignment")
        return mean_riemann(trial_covs_np)

    if alignment_type == "euclidean" or alignment_type in {"none", None}:
        return np.mean(trial_covs_np, axis=0)

    raise ValueError(f"Unsupported alignment type: {alignment_type}")


def _compute_alignment_transform_np(cov_np: np.ndarray,
                                    eps_cfg: float = 0.,
                                    eps_scale: float = 1e-6) -> np.ndarray:
    """
    Return Σ^{-½} with a guaranteed shrinkage ε·I  (ε = max(eps_cfg, trace*eps_scale)).
    """
    eps_auto = np.trace(cov_np) * eps_scale
    eps = max(eps_cfg, eps_auto)

    cov_shrunk = cov_np + eps * np.eye(cov_np.shape[0], dtype=cov_np.dtype)
    eigvals, eigvecs = np.linalg.eigh(cov_shrunk)
    eigvals_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    return eigvecs @ eigvals_inv_sqrt @ eigvecs.T


def _apply_alignment_transform_np(trials_np: np.ndarray, transform_np: np.ndarray) -> np.ndarray:
    """Whiten a batch of trials → (n_trials, n_channels, n_times)."""
    return np.einsum("jk,ikm->ijm", transform_np, trials_np, dtype=trials_np.dtype)


# AdaBN helper (– disabled unless args.use_adabn == True)
def _set_adabn_status(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.track_running_stats = not enabled



# Main wrapper
class TTAWrapper(nn.Module):
    """Parameter‑free test‑time adaptation via Euclidean Alignment."""

    def __init__(self, model: nn.Module, args: Any, *, sr_hz: Optional[float] = None, global_backrot_matrix_np: Optional[np.ndarray] = None):
        super().__init__()
        self.wrapped_model = model
        self.args = args
        self.sr_hz = sr_hz
        self.device = next(model.parameters()).device
        self.decision_criterion = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Sliding window of trial covariances (default 50) for EA
        window = getattr(args, "tta_cov_buffer_size", 50)
        self._cov_buffer: deque[np.ndarray] = deque(maxlen=window)

        # Runtime state
        self.reference_cov_np: Optional[np.ndarray] = None
        self.alignment_transform_torch: Optional[torch.Tensor] = None
        
        # --- Internal flag to override fine-tuning mode ---
        self._force_full_update = False


        # ---  Store the global back-rotation matrix as a torch tensor ---
        self.global_backrot_torch: Optional[torch.Tensor] = None
        # Check the flag in args to ensure back-rotation is actually enabled for this run
        if global_backrot_matrix_np is not None and getattr(args, "ea_backrotation", False):
            self.global_backrot_torch = (
                torch.from_numpy(global_backrot_matrix_np)
                    .float()
                    .to(self.device, non_blocking=True)
            )
            log.info("TTAWrapper initialized with a global back-rotation matrix.")

        if self.args.alignment_type == "riemannian" and not PYRIEMANN_AVAILABLE:
            log.warning("Riemannian alignment requested but pyriemann not available – falling back to Euclidean.")
            self.args.alignment_type = "euclidean"


    @torch.no_grad()
    def init_alignment_from_calibration(self, calib_np: np.ndarray) -> None:
        """Compute first whitening matrix from the calibration block."""
        covs = _compute_trial_covariances_np(calib_np, self.args.alignment_cov_epsilon)
        self._cov_buffer.extend(covs)
        self._recompute_transform()

    @torch.no_grad()
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass without updating adaptation statistics."""
        x_aligned = self._apply_current_alignment_torch(x)
        logits = self.wrapped_model(x_aligned, **kwargs)
        return logits - self.decision_criterion


    @torch.no_grad()
    def adapt_alignment(self, trial_np: np.ndarray) -> None:
        """Update whitening matrix with a single *unlabelled* trial."""
        if not self.args.use_tta or self.args.alignment_type in {None, "none"}:
            return
        cov = _compute_trial_covariances_np(trial_np[None, ...], self.args.alignment_cov_epsilon)[0]
        self._cov_buffer.append(cov)
        self._recompute_transform()

    # Backward‑compat convenience – now just *predict then adapt*
    @torch.no_grad()
    def update_tta_statistics_and_predict(self, x_raw: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: N802 (legacy name)
        logits = self.predict(x_raw, **kwargs)
        self.adapt_alignment(x_raw.detach().cpu().numpy())
        return logits


    # Internal helpers
    def _recompute_transform(self) -> None:
        """
        Update the whitening transform P^{-1/2} with an EMA-smoothed
        reference covariance.  Requires self.args.alignment_ref_ema_beta
        in (0,1].  If beta == 1 we get the old 'simple mean' behaviour.
        """
        if not self._cov_buffer:
            return

        # 1)  Instantaneous covariance estimate using your sliding window
        cov_inst_np = np.mean(self._cov_buffer, axis=0)
        # 2)  Exponential moving average
        beta = getattr(self.args, "alignment_ref_ema_beta", 1.0)
        if self.reference_cov_np is None:
            # first update → initialise EMA
            self.reference_cov_np = cov_inst_np
        else:
            # EMA smoothing
            self.reference_cov_np = (
                beta * self.reference_cov_np + (1.0 - beta) * cov_inst_np
            )
        # 3)  Recompute whitening matrix from the *smoothed* covariance
        transform_np = _compute_alignment_transform_np(
            self.reference_cov_np,
            self.args.alignment_transform_epsilon
        )
        # 4)  Cache as torch tensor on the right device
        self.alignment_transform_torch = (
            torch.from_numpy(transform_np)
                .float()
                .to(self.device, non_blocking=True)
        )

    def _apply_current_alignment_torch(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.args.use_tta
            and self.args.alignment_type not in {None, "none"}
            and self.alignment_transform_torch is not None
        ):
            # Step 1: Whiten the data with the subject-specific transform
            x_whitened = torch.einsum("jk,bkm->bjm", self.alignment_transform_torch, x)

            # Step 2: Apply global back-rotation if available
            if self.global_backrot_torch is not None:
                # This applies Σ_global^{+1/2} to the whitened data
                return torch.einsum("jk,bkm->bjm", self.global_backrot_torch, x_whitened)
            
            return x_whitened # Return whitened data if no back-rotation
        return x


    # Boiler‑plate wrappers to expose underlying model 

    def forward(self, x: torch.Tensor, *, apply_tta: bool = False, **kwargs) -> torch.Tensor:  # still used by some scripts
        kwargs.pop('is_finetuning_batch', None)

        if apply_tta:
            # self.predict will now receive the "cleaned" kwargs and will
            # pass them to the wrapped model.
            return self.predict(x, **kwargs)
        
        # Pass the "cleaned" kwargs to the wrapped model.
        return self.wrapped_model(x, **kwargs)
    

    def train(self, mode: bool = True): 
        super().train(mode)
        self.wrapped_model.train(mode)
        if mode:
            _set_adabn_status(self.wrapped_model, enabled=False)
        return self

    def eval(self):  
        super().eval()
        self.wrapped_model.eval()
        _set_adabn_status(self.wrapped_model, enabled=False)
        return self

    # Expose parameters conditionally (kept from original implementation)
    def parameters(self, recurse: bool = True):
        return self._select_params()

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for n, p in self._select_params(named=True, prefix=prefix):
            yield n, p

    # Helpers for selective fine‑tuning (decision‑only / full)
    def _select_params(self, *, named: bool = False, prefix: str = ""):
        # --- Respect the override flag ---
        finetune_setting = getattr(self.args, "finetune_mode", "full")
        
        # Determine the effective mode based on the override flag
        effective_mode = "full" if self._force_full_update else finetune_setting

        # --- Handle decision_criterion_only mode ---
        if effective_mode == "decision_criterion_only":
            if named:
                yield f"{prefix}decision_criterion", self.decision_criterion
            else:
                yield self.decision_criterion
            return

        if effective_mode == "decision_only":
            # last learnable conv/linear layer
            for module_name, module in reversed(list(self.wrapped_model.named_modules())):
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    if any(p.requires_grad for p in module.parameters()):
                        if named:
                            for n, p in module.named_parameters(prefix=f"{prefix}{module_name}."):
                                yield n, p
                        else:
                            yield from module.parameters()
                        return
            # fall‑back → whole model
            log.warning("'decision_only' mode specified, but no suitable layer found. Falling back to full model update.")
        
        # This block now handles 'full' mode, the fallback, and the temporary override
        if named:
            yield from self.wrapped_model.named_parameters(prefix=f"{prefix}")
        else:
            yield from self.wrapped_model.parameters()

    def enable_full_model_update(self, enabled: bool):
        """
        Force the wrapper to yield all model parameters, overriding 'finetune_mode'.
        """
        self._force_full_update = enabled
        status = "ENABLED" if enabled else "DISABLED"
        log.info(f"Full model parameter update override is now {status}.")

