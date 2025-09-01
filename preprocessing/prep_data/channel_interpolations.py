#%%
import numpy as np
from mne.bem import _check_origin
from mne._fiff.pick import pick_types
from warnings import warn
from mne.channels.interpolation import _make_interpolation_matrix

#modified functions from https://github.com/mne-tools/mne-python/blob/maint/1.9/mne/channels/channels.py and https://github.com/mne-tools/mne-python/blob/maint/1.9/mne/channels/interpolation.py

def custom_get_interpolation_matrix(inst, exclude=None, ecog=False):
    if exclude is None:
        exclude = list()
    bads_idx = np.zeros(len(inst.ch_names), dtype=bool)
    goods_idx = np.zeros(len(inst.ch_names), dtype=bool)

    picks = pick_types(inst.info, meg=False, eeg=not ecog, ecog=ecog, exclude=exclude)
    inst.info._check_consistency()
    bads_idx[picks] = [inst.ch_names[ch] in inst.info["bads"] for ch in picks]

    if len(picks) == 0 or bads_idx.sum() == 0:
        return

    goods_idx[picks] = True
    goods_idx[bads_idx] = False

    pos = inst._get_channel_positions(picks)

    # Make sure only EEG channels are used
    bads_idx_pos = bads_idx[picks]
    goods_idx_pos = goods_idx[picks]

    # test spherical fit
    origin = _check_origin("auto", inst.info)

    distance = np.linalg.norm(pos - origin, axis=-1)
    distance = np.mean(distance / np.mean(distance))
    if np.abs(1.0 - distance) > 0.1:
        warn(
            "Your spherical fit is poor, interpolation results are "
            "likely to be inaccurate."
        )

    pos_good = pos[goods_idx_pos] - origin
    pos_bad = pos[bads_idx_pos] - origin
    print(f"Computing interpolation matrix from {len(pos_good)} sensor positions")
    interpolation = _make_interpolation_matrix(pos_good, pos_bad)
    return interpolation, goods_idx, bads_idx

def apply_channel_interpolation(inst, interpolation_info):
    interpolation = interpolation_info['interpolation_matrix']
    goods_idx = interpolation_info['goods_idx']
    bads_idx = interpolation_info['bads_idx']
    inst._data[..., bads_idx, :] = np.matmul(
        interpolation, inst._data[..., goods_idx, :])