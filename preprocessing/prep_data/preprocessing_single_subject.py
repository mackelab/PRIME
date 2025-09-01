#%%
import mne #for useful functions and structures
import numpy as np #for useful linear algebra functions
#import neat methods from scipy
import scipy
from scipy.stats import median_abs_deviation
from scipy.stats import zscore
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings
import matplotlib.pyplot as plt

from ica_calibrator import *
from ssp_sir_python import *
from sound_modified import *
from channel_interpolations import *
import time
import argparse
import os
import sys
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*ICA\.apply\(\) was baseline-corrected.*"
)

def butter_filter(data, cutoff, btype, fs, order, pad_time):
    nyquist = fs/2 #nyquist rate
    if isinstance(cutoff, list):
        if len(cutoff)==2:
            cutoff_normalized = [cutoff[0]/nyquist, cutoff[1]/nyquist] #for bandpass and bandstop filters (two frequency edges)
        else:
            raise ValueError("cutoff should be a single value or a list of length 2.")
    else:
        cutoff_normalized = cutoff/nyquist #low-pass or high-pass style filter has only one edge frequency
    b,a = butter(order, cutoff_normalized, btype=btype, analog=False) #get the filter coefficients
    filtered_data = apply_filter_to_data(data, [b, a], pad_time, fs)
    return filtered_data, [b, a] #return the filtered data without pad samples

def apply_filter_to_data(data, coefs, pad_time, fs):
    pad_samples = int(pad_time*fs) #number of samples to pad
    padded_data = np.pad(data, ((0,0),(0,0),(pad_samples,pad_samples)), mode='reflect') #apply padding to the data in the time domain (both sides)
    filtered_padded_data = filtfilt(coefs[0], coefs[1], padded_data, padlen=None) #apply filter to the data
    return filtered_padded_data[:,:,pad_samples:-pad_samples]

#----------------------------- Finding and reconstructing bad channels EEG from epoched data ----------------------------- ----------------------------- 
def get_bad_channels_epoched(epochs, z_score_thresh_mad, z_score_thresh_power, fmin_fmax, z_score_autocorr, z_score_auc):
    bad_channels = get_bad_channels_epoched_run(epochs.copy().set_eeg_reference('average', projection=False, verbose=False), z_score_thresh_mad, z_score_thresh_power,fmin_fmax, z_score_autocorr, z_score_auc)
    if len(bad_channels) > 0: #if one or more bad channels were found then re-run the channel rejection
        while True:
            epochs_without_bad_channels = epochs.copy().drop_channels(bad_channels).set_eeg_reference('average', projection=False, verbose=False) #do not analyze the dropped channels anymore
            #get new bad channels
            bad_channels_new = get_bad_channels_epoched_run(epochs_without_bad_channels, z_score_thresh_mad, z_score_thresh_power,fmin_fmax, z_score_autocorr, z_score_auc)
            if len(bad_channels_new) == 0:
                break
            bad_channels = list(np.union1d(bad_channels, bad_channels_new)) #list of all bad channels
    return bad_channels

def get_bad_channels_epoched_run(epochs, z_score_thresh_mad, z_score_thresh_power, fmin_fmax, z_score_autocorr, z_score_auc):
    data = epochs.get_data(copy=True) # get the epoched data
    n_trials, n_channels, _ = data.shape #number of trials and channels
    bad_channel_indices = np.array([]) #initialize an empty array for later unions with bad channels detected by various metrics

    if z_score_thresh_mad:
        data_flattened = data.swapaxes(0, 1).reshape(n_channels, -1)
        #get the median absolute deviation values (more robust and not as sensitive to outliers compared to standard deviation)
        mad_values = median_abs_deviation(data_flattened, axis=1)
        #z-score the values and get the absolute values
        mad_z_scores = zscore(mad_values)
        #compare absolute z-scored values to the threshold
        bad_channel_indices_deviation = np.where((mad_z_scores > z_score_thresh_mad[1]) | (mad_z_scores < z_score_thresh_mad[0]))[0]
        print_out_channel_rejection_info(epochs, bad_channel_indices_deviation, "median absolute deviation")
        bad_channel_indices = np.union1d(bad_channel_indices, bad_channel_indices_deviation)

    if z_score_thresh_power:
        # check if channel has a lot of high-frequency power
        # get the power spectral densities
        trials_channels_psds, _ = get_psds(data,epochs.info['sfreq'],fmin_fmax[0],fmin_fmax[1])
        channel_psds = np.mean(trials_channels_psds,axis=0) #average psds for each channel
        channel_psds = np.mean(channel_psds,axis=-1) #average psds for each channel
        psd_z_scores = zscore(channel_psds) #z-scored average psds
        #which channels go over the threshold?
        bad_channel_indices_power = np.where(psd_z_scores > z_score_thresh_power)[0]
        # combine the information of bad channels
        print_out_channel_rejection_info(epochs, bad_channel_indices_power, "high frequency power")
        bad_channel_indices = np.union1d(bad_channel_indices,bad_channel_indices_power)

    if z_score_autocorr:
        #detect flat channels
        # compute autocorrelation the mean autocorrelation for each channel
        ch_autocorrelations = [np.mean([autocorr(data[trial_ind, ch_ind,:],lag=1)
                                for trial_ind in range(n_trials)]) for ch_ind in range(n_channels)]
        autocorrelation_z_scores = np.abs(zscore(ch_autocorrelations)) #z-scored mean autocorrelations
        #which channels go over the threshold?
        bad_channel_indices_autocorrelation = np.where(autocorrelation_z_scores > z_score_autocorr)[0]
        print_out_channel_rejection_info(epochs, bad_channel_indices_autocorrelation, "autocorrelation")
        # combine the information of bad channels
        bad_channel_indices = np.union1d(bad_channel_indices,bad_channel_indices_autocorrelation)
    
    if z_score_auc:
        #check if a channel is bad by calculating area under the curve distribution
        #this should be only used for post-stimulus intervals, where large decay artifacts can substantially contaminate the signal
        evoked_data = np.mean(data, axis=0)
        auc_values = [np.trapz(np.abs(evoked_data[ch_ind,:])) for ch_ind in range(n_channels)]
        auc_z_scores = zscore(auc_values) #z-scored aucs
        #which channels go over the threshold?
        bad_channel_indices_auc = np.where(auc_z_scores > z_score_auc)[0]
        # combine the information of bad channels
        print_out_channel_rejection_info(epochs, bad_channel_indices_auc, "area under the curve")
        bad_channel_indices = np.union1d(bad_channel_indices,bad_channel_indices_auc)
    bad_channel_indices = bad_channel_indices.astype(int)
    # get the names of the bad channels
    bad_channels = [epochs.ch_names[bad_ch_ind] for bad_ch_ind in bad_channel_indices]
    return bad_channels

def autocorr(data, lag=1):
    #compute autocorrelation for a given lag
    return np.corrcoef(data[:-lag],data[lag:])[0,1]

def get_psds(data, sfreq, fmin, fmax):
    #get the power spectral densities for all channels and trials
    psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq, fmin=fmin, fmax=fmax, verbose=False)
    return psds, freqs

def print_out_channel_rejection_info(structure, rejected_channel_indices, type_of_rejection):
    if len(rejected_channel_indices) > 0 and type_of_rejection:
        print(f"Rejected channels {[structure.ch_names[bad_ch_ind] for bad_ch_ind in rejected_channel_indices]} due to deviation in {type_of_rejection}.")
    if not type_of_rejection and len(rejected_channel_indices) == 0:
        print(f"Rejected channels {[structure.ch_names[bad_ch_ind] for bad_ch_ind in rejected_channel_indices]}.")

def get_bad_channels_pre(epochs, rejection_options, filter_options):
    epochs_data = epochs.get_data(copy=True) #get the epoched data
    epochs_data, epochs_pre_filter = butter_filter(epochs_data, filter_options['cutoff'], filter_options['btype'], epochs.info['sfreq'], filter_options['order'], filter_options['pad_time'])
    epochs_pre_filtered = mne.EpochsArray(epochs_data, epochs.info, epochs.events, tmin=epochs.times[0]) #re-create the epochs object for channel rejection
    bad_channels_pre = get_bad_channels_epoched(epochs_pre_filtered, rejection_options['z_score_thresh_mad'],
                                                 rejection_options['z_score_thresh_power'],rejection_options['fmin_fmax'], rejection_options['z_score_thresh_autocorr'], False)
    return bad_channels_pre, epochs_pre_filter

def get_bad_channels_post(epochs, rejection_options, post_range):
    #apply baseline correction using a post-stimulus time (responses should not be on here or artifacts, such as recharging artifacts)
    if post_range:
        epochs_post_true = epochs.copy().crop(post_range[0], post_range[1])
    else:
        epochs_post_true = epochs.copy()
    bad_channels_post = get_bad_channels_epoched(epochs_post_true, False, False, False, False, rejection_options['z_score_thresh_auc']) #reject channels in post-stimulus period based on area under the curve
    return bad_channels_post

def reconstruct_bad_channels(epochs, bad_channels, interpolation_info):
    epochs.info['bads'] = bad_channels #set the bad channel information to info
    if interpolation_info is None:
        interpolation_matrix, goods_idx, bads_idx = custom_get_interpolation_matrix(epochs, exclude=None, ecog=False)
        interpolation_info = {'interpolation_matrix':interpolation_matrix, 'goods_idx':goods_idx, 'bads_idx':bads_idx}
    apply_channel_interpolation(epochs, interpolation_info) #interpolate bad channels
    epochs.info['bads'] = [] #reset bad channels after interpolation

    return epochs, interpolation_info

#---------------- Functions for finding bad trials from pre- or post-stimulus EEG data -------------------------------------------------------------
def find_bad_trials(epochs, global_zscore_threshold, local_zscore_threshold, psd_trial_threshold, psd_freq_range):
    bad_trials, good_trial_stats = find_bad_trials_run(epochs, global_zscore_threshold, local_zscore_threshold, psd_trial_threshold, psd_freq_range, np.array([]))
    if len(bad_trials) > 0:
        while True:
            bad_trials_new, good_trial_stats = find_bad_trials_run(epochs, global_zscore_threshold, local_zscore_threshold, psd_trial_threshold, psd_freq_range, bad_trials)
            if len(bad_trials_new) == 0:
                break
            else:
                bad_trials = np.union1d(bad_trials, bad_trials_new)
    return list(bad_trials.astype(int)), good_trial_stats

def find_bad_trials_run(epochs, global_zscore_threshold, local_zscore_threshold, psd_trial_threshold, psd_freq_range, current_bad_trials):
    data =  epochs.get_data(copy=True) #data is n_trials x n_channels x n_samples
    n_trials = data.shape[0] #number of all trials
    trial_indices = np.arange(0, n_trials, 1).astype(int)
    good_trial_indices = np.array([ind for ind in trial_indices if ind not in current_bad_trials])
    data = data[good_trial_indices,:,:] #only the good trial indices

    if psd_trial_threshold:
        psds, _ = get_psds(data, epochs.info['sfreq'],psd_freq_range[0],psd_freq_range[1])
        psds_trials = np.mean(psds, axis=(1,2)) #mean across frequencies and channels
        z_score_psd_trials_adjusted = []
        z_score_psd_trials = zscore(psds_trials)

    mad_trials = median_abs_deviation(data, axis=(1,2)) #median abs deviation across channels and samples
    z_score_global_trials = zscore(mad_trials)


    mad_trials_channels = median_abs_deviation(data, axis=2) #median abs deviation across samples
    z_score_local_trials = zscore(mad_trials_channels, axis=1)
    z_score_global_trials_adjusted = []
    z_score_local_trials_adjusted = []
    good_ind = 0
    for i in trial_indices:
        if i in good_trial_indices:
            z_score_global_trials_adjusted.append(z_score_global_trials[good_ind])
            z_score_local_trials_adjusted.append(z_score_local_trials[good_ind])
            if psd_trial_threshold:
                z_score_psd_trials_adjusted.append(z_score_psd_trials[good_ind])
            good_ind += 1
        else: #do not re-reject bad trials
            z_score_global_trials_adjusted.append(0)
            z_score_local_trials_adjusted.append(np.zeros(z_score_local_trials.shape[1]))
            if psd_trial_threshold:
                z_score_psd_trials_adjusted.append(0)
    
    if len(z_score_local_trials_adjusted):                          # at least one good trial left
        z_score_local_trials_adjusted = np.vstack(z_score_local_trials_adjusted)
    else:                                                           # corner case: nothing to stack
        z_score_local_trials_adjusted = np.zeros(
            (len(trial_indices), mad_trials_channels.shape[1])
        )

    z_score_global_trials_adjusted = np.asarray(z_score_global_trials_adjusted)
    if psd_trial_threshold:
        z_score_psd_trials_adjusted = np.asarray(z_score_psd_trials_adjusted)


    # check which trials exceed global_zscore_threshold
    bad_trials_local = np.where(np.any(np.abs(z_score_local_trials_adjusted) > local_zscore_threshold, axis=1))[0]
    bad_trials = np.where((np.array(z_score_global_trials_adjusted) > global_zscore_threshold[1]) | (np.array(z_score_global_trials_adjusted) < global_zscore_threshold[0]))[0]

    bad_trials_mad_current = np.where((z_score_global_trials > global_zscore_threshold[1]) | (z_score_global_trials < global_zscore_threshold[0]))[0]
    good_indices_mad = np.array([index for index in range(len(mad_trials)) if index not in bad_trials_mad_current])
    good_trial_stats = {'mads': mad_trials[good_indices_mad], 'mads_std': np.std(mad_trials[good_indices_mad]), 'mads_mean': np.mean(mad_trials[good_indices_mad])} #compute statistics of good trials

    bad_trials = np.union1d(bad_trials, bad_trials_local) #add the bad "globally and locally" detected bad trials
    if psd_trial_threshold:
        #compute statistics of good trials
        bad_trials_psd = np.where(np.array(z_score_psd_trials_adjusted) > psd_trial_threshold)[0]
        bad_trials_psd_current = np.where(z_score_psd_trials > psd_trial_threshold)[0]
        good_indices_psd = np.array([index for index in range(len(psds_trials)) if index not in bad_trials_psd_current])
        good_trial_stats['psds'] = psds_trials[good_indices_psd] #accepted PSDs of the trials
        good_trial_stats['psds_std'] = np.std(psds_trials[good_indices_psd])
        good_trial_stats['psds_mean'] = np.mean(psds_trials[good_indices_psd])

        bad_trials = np.union1d(bad_trials, bad_trials_psd)

    return bad_trials, good_trial_stats


def get_bad_ocular_trials(ica, epochs, tmin, tmax, ocular_artifact_indices, z_thresh):
    ica_source_tcs = ica.get_sources(epochs) #get the ica sources
    ica_tcs = ica_source_tcs.get_data(copy=True) #the the ica source time courses

    #check if tmin and tmax have been defined to something else than None and set them to values that wont cause problems
    if tmax is None:
        tmax = np.inf
    if tmin is None:
        tmin = -np.inf

    time_indices_of_interest = np.where((ica_source_tcs.times >= tmin) & (ica_source_tcs.times <= tmax))[0] #time indices to check for ocular artifacts for
    bad_trials = np.array([]) #init array for bad trials
    ic_thresholds = {} #store distribution information of time courses here

    for comp_ind in ocular_artifact_indices: #go through the ocular artifact indices
        abs_ica_tcs = np.abs(ica_tcs[:, comp_ind, :])          # shape: (n_trials, n_times)
        n_trials, n_times = abs_ica_tcs.shape                  # cache dims once
        ocular_comp_ts = abs_ica_tcs.ravel()                       # 1-D view, no copy
        z_scored = zscore(ocular_comp_ts).reshape(n_trials, n_times) #z-scored and reshaped time courses (n_good_trials x n_times)

        assert z_scored.shape == (n_trials, n_times), (
            f"Shape mismatch after z-scoring: got {z_scored.shape}, "
            f"expected {(n_trials, n_times)}"
        )

        z_scores_in_range = z_scored[:, time_indices_of_interest] #z-scores within the time range of interest (n_good_trials x n_times_in_time_range_of_interest (=len(time_indices_of_interest)))
        median_z_scores = np.median(z_scores_in_range, axis=1) #check whether the median z-score exceeds the threshold
        #adjust the size of median_z_scores to match the number of trials for rejection
        bad_trial_inds_now = np.where(median_z_scores > z_thresh)[0]
        ic_thresholds[comp_ind] = {'std':np.std(ocular_comp_ts), 'mean': np.mean(ocular_comp_ts), 'time_indices_of_interest':time_indices_of_interest} #the maximum accepted median
        bad_trials = np.union1d(bad_trials, bad_trial_inds_now)

    return list(bad_trials.astype(int)), ic_thresholds #return the bad ocular trials and the number of iterations the method took to converge


def drop_trials_from_structs(epochs_objects, bad_trials):
    for epochs in epochs_objects:
        epochs.drop(bad_trials) #directly drop bad trials
    return epochs_objects

#--------------------------------------------------------------- Functions for checking bad trials for EMG ------------------------------------------------------------------------
def get_bad_trials_emg(emg_epochs, pre_innervation_options, ptp_options, line_freq):
    full_emg_times = emg_epochs.times #all times in emg
    full_emg_data = emg_epochs.get_data(copy=True) #all emg epochs data that is left
    pre_innervation_time_indices = np.where((full_emg_times >= pre_innervation_options['tmin']) & (full_emg_times <= pre_innervation_options['tmax']))[0] #potential pre-innervation time indices in the full times
    peak_to_peak_time_indices = np.where((full_emg_times >= ptp_options['tmin']) & (full_emg_times <= ptp_options['tmax']))[0] #potential peak-to-peak time indices in the full times
    pre_stim_times = full_emg_times[pre_innervation_time_indices] #get the times in the pre-innervation window
    n_emg_channels = full_emg_data.shape[1] #number of channels
    emg_prep_times = {'full_emg_times':full_emg_times, 'pre_innervation_time_indices':pre_innervation_time_indices, 'peak_to_peak_time_indices':peak_to_peak_time_indices, 'pre_stim_times':pre_stim_times}
    n_trials = full_emg_data.shape[0] #number of trials
    bad_trials = [] #initialize list for bad trials
    channels_ptps = [] #peak to peaks for all channels for all trials
    channels_pre_innervations = [] #pre-innervation info for all channels for all trials
    for emg_channel_ind in range(n_emg_channels):
        #init lists for channel peak-to-peaks and pre-innervation info
        channel_ptps = []
        channel_pre_innervations = []
        for trial_ind in range(n_trials): #go through all trials
            has_pre_innervation, valid_peak_to_peak, ptp, full_emg_data = prep_one_emg_trial(full_emg_data, trial_ind, emg_channel_ind, pre_innervation_time_indices,
                                                                               line_freq, pre_stim_times, full_emg_times, pre_innervation_options,peak_to_peak_time_indices, ptp_options, emg_epochs.info['sfreq'])
            
            if has_pre_innervation or valid_peak_to_peak is False: #then note the trial as bad
                bad_trials.append(trial_ind)
            channel_ptps.append(ptp) #add channel peak-to-peak information for the trial
            channel_pre_innervations.append(has_pre_innervation) #add channel pre-innervation information for the trial
        channels_ptps.append(channel_ptps) #add channel peak-to-peak information across trials
        channels_pre_innervations.append(channel_pre_innervations) #add channel pre-innervation information across trials
    bad_trials = np.unique(bad_trials) #remove duplicates because bad trials can be detected from both channels
    emg_epochs = mne.EpochsArray(full_emg_data, info=emg_epochs.info,  events=emg_epochs.events, tmin=emg_epochs.times[0]) #re-create the epochs structure with the sine fit-corrected data
    return bad_trials, emg_epochs, channels_ptps, channels_pre_innervations, emg_prep_times

def prep_one_emg_trial(full_emg_data, trial_ind, emg_channel_ind, pre_innervation_time_indices, line_freq, pre_stim_times, full_emg_times, pre_innervation_options,peak_to_peak_time_indices, ptp_options, sfreq):
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=scipy.optimize.OptimizeWarning)
        for harmonic_ind in [1, 2]: #fit line_freq first, remove the fit from the data and then do the same with second harmonic on the residual
            try:
                data_pre = full_emg_data[trial_ind, emg_channel_ind, pre_innervation_time_indices] #pre-stim emg data for the current trial
                sine_model_to_fit = fit_sine_wave(line_freq=line_freq*harmonic_ind) #initialize fit with specific freq (50 Hz in Europe and 60 Hz in the US) or its 2nd harmonic
                sine_predicted = get_line_freq_sine(data_pre, pre_stim_times, full_emg_times, sine_model_to_fit) #get the line_freq harmonic sine fit to the pre-innervation time window
                full_emg_data[trial_ind, emg_channel_ind, :] -= sine_predicted #subtract the estimated sine wave from each respective time point
            except scipy.optimize.OptimizeWarning:
                if full_emg_data.shape[0] > 1:
                    print(f"Optimizing prolems for EMG data sine fitting for channel {emg_channel_ind}, trial {trial_ind}, line_freq {line_freq}, harmonic {harmonic_ind} (fit not used)")
                else:
                    print(f"Optimizing prolems for EMG data sine fitting for channel {emg_channel_ind}, line_freq {line_freq}, harmonic {harmonic_ind} (fit not used)")
                break #don't continue sine fitting if this arises
    

    #now check that should the trial be rejected or not
    has_pre_innervation = check_pre_innervation(full_emg_data[trial_ind, emg_channel_ind, pre_innervation_time_indices], pre_innervation_options['threshold']) #check if the trial has pre-innervation


    #check for a valid MEP
    valid_peak_to_peak, ptp = check_peak_to_peak(full_emg_data[trial_ind, emg_channel_ind, peak_to_peak_time_indices], ptp_options['min_ptp_height'],
                                                    ptp_options['prominence'], ptp_options['min_distance'], sfreq, ptp_options['check_ptp'])
    
    return has_pre_innervation, valid_peak_to_peak, ptp, full_emg_data

def check_peak_to_peak(emg_trial_data, min_ptp_height, prominence, min_distance, sfreq, check_ptp):
    if check_ptp is True:
        min_distance_in_samples = min_distance * sfreq
        #find positive and negative peaks from the data
        peaks_pos, _ = find_peaks(emg_trial_data, prominence=prominence, distance=min_distance_in_samples) #positive peaks
        peaks_neg, _ = find_peaks(-emg_trial_data, prominence=prominence, distance=min_distance_in_samples) #negative peaks

        if len(peaks_pos) < 1 or len(peaks_neg) < 1: #check that at least one positive and one negative peak have been found
            return False, False #two peaks were not found
        
        #highest peaks from both positive and negative
        max_pos = np.max(emg_trial_data[peaks_pos])
        min_neg = np.min(emg_trial_data[peaks_neg])
        ptp = max_pos - min_neg #peak-to-peak of the data
        if ptp < min_ptp_height: #check if the peak-to-peak threshold is exceeded
            return False, False
    else:
        max_pos = np.max(emg_trial_data)
        min_neg = np.min(emg_trial_data)
        ptp = max_pos - min_neg #peak-to-peak of the data (min-max difference in this case)
    
    return True, ptp #return True and the true peak-to-peak when the trial is valid!


def check_pre_innervation(emg_trial_data, pre_innervation_threshold):
    min_max_diff = np.max(emg_trial_data) - np.min(emg_trial_data) #min-max difference of the data
    has_pre_innervation = True if min_max_diff > pre_innervation_threshold else False #pre-innervation is deemed to be True or False depending on if the threshold is exceeded
    return has_pre_innervation


def get_line_freq_sine(data_pre, pre_stim_times, full_emg_times, sine_model_to_fit):
    p0 = [np.std(data_pre)*np.sqrt(2), 0, np.mean(data_pre)] #initial guesses for amplitude, phase, and offset, respectively
    popt, _  = curve_fit(sine_model_to_fit, pre_stim_times, data_pre, p0=p0) #optimize model paramaters by fitting a line_freq sine function to the pre-stimulus (pre-innervation) window
    sine_predicted = sine_model_to_fit(full_emg_times, *popt) #fit sine wave using the model parameters such that it extends the whole emg window
    return sine_predicted


def fit_sine_wave(line_freq):
    def sine_model(t, A, phi, C):
        return A*np.sin(2*np.pi*line_freq*t + phi) + C
    return sine_model

def get_better_emg_channel(emg_epochs, ptp_options, filter_options_emg):
    emg_data = emg_epochs.get_data(copy=True) #get the emg data
    #apply filtering to the whole data (only highpass)
    emg_data, emg_filter = butter_filter(emg_data, filter_options_emg['cutoff'], filter_options_emg['btype'], emg_epochs.info['sfreq'], filter_options_emg['order'], filter_options_emg['pad_time'])
    emg_epochs = mne.EpochsArray(emg_data, info=emg_epochs.info,  events=emg_epochs.events, tmin=emg_epochs.times[0])
    emg_epochs_data_in_range = emg_epochs.copy().crop(ptp_options['tmin'], ptp_options['tmax']).get_data(copy=True)
    min_data = np.min(emg_epochs_data_in_range, axis=2)
    max_data = np.max(emg_epochs_data_in_range, axis=2)
    diff_data = max_data - min_data
    average_diffs = np.mean(diff_data, axis=0)
    better_channel_index = np.argmax(average_diffs)
    picked_channel = emg_epochs.ch_names[better_channel_index] #get the channel name
    return picked_channel, emg_filter


def preprocess_prestim_and_poststim(epochs, epochs_emg, pre_range_times, post_range_times, emg_times, channel_rejection_options, ica_options, trial_rejection_options,
                                     emg_trial_rejection_options, sound_options, ssp_sir_options, L, filter_options, filter_options_emg,
                                       line_freq, resample_to, n_trials_goal, use_ica_on_pre, emg_filter):
    
    preprocessing_info = {} #initialize dict for pre-processing information
    pre_timerange_ica = ica_options['pre_timerange']
    pre_range = pre_range_times['pre_range_final']
    #extract the post-stimulus time range information from the dict
    post_range = post_range_times['post_range']
    post_range_baseline = post_range_times['baseline']
    pulse_artifact_time1 = post_range_times['pulse_artifact_time1']
    pulse_artifact_time2 = post_range_times['pulse_artifact_time2']
    post_range_reject = post_range_times['post_range_reject']

    #crop epochs separately for pre- and post-time ranges
    epochs_emg.crop(emg_times[0], emg_times[1])
    epochs_pre = epochs.copy().crop(pre_range[0], pre_range[1])
    epochs_pre_ica = epochs.copy().crop(pre_timerange_ica[0], pre_timerange_ica[1])
    epochs_post = epochs.copy().crop(post_range[0], post_range[1])
    del epochs # not needed anymore because pre- and post-stim for EEG are preprocessed separately

    epochs_pre.resample(resample_to, method='polyphase') #downsample data
    epochs_pre_ica.resample(resample_to, method='polyphase') #downsample data
    epochs_post.resample(resample_to, method='polyphase') #downsample data
    epochs_emg.resample(resample_to, method='polyphase') #downsample data

    epochs_post.apply_baseline(post_range_baseline) #baseline correct the post-stimulus data
    epochs_post = mne.preprocessing.fix_stim_artifact(epochs_post, tmin=pulse_artifact_time1[0], tmax=pulse_artifact_time1[1], mode='window') #reconstruct pulse artifact window
    #detect bad channels separately from pre- and post-stimulus periods, interpolate them using MNE, and then set EEG to average reference
    bad_channels_pre, pre_stim_filter = get_bad_channels_pre(epochs_pre, channel_rejection_options['pre'], filter_options)
    preprocessing_info['pre_stim_filter'] = pre_stim_filter #save the filter used for pre-stim
    bad_channels_post = get_bad_channels_post(epochs_post, channel_rejection_options['post'], post_range_reject)
    bad_channels = list(np.union1d(bad_channels_pre, bad_channels_post)) #combine information on bad channels

    #update bad channel information to the preprocessing info dictionary
    preprocessing_info['channels_before_rejection'] = epochs_post.ch_names
    preprocessing_info['bad_channels_pre'] = bad_channels_pre
    preprocessing_info['bad_channels_post'] = bad_channels_post
    preprocessing_info['bad_channels'] = bad_channels
    
    if bad_channels: #interpolate bad channels if any were found
        epochs_pre, interpolation_info = reconstruct_bad_channels(epochs_pre, bad_channels, interpolation_info=None)
        epochs_pre_ica, _ = reconstruct_bad_channels(epochs_pre_ica, bad_channels, interpolation_info=interpolation_info)
        epochs_post, _ = reconstruct_bad_channels(epochs_post, bad_channels,interpolation_info=interpolation_info)
    else:
        interpolation_info = None
    preprocessing_info['channel_interpolation_info'] = interpolation_info
    
    #Run ICA only on pre-stim that has been prepared for icalabel and apply the ica filters to the pre-stim and post-stim
    epochs_pre_ica_data = epochs_pre_ica.get_data(copy=True)
    epochs_pre_ica_data, epochs_pre_ica_filter = butter_filter(epochs_pre_ica_data, ica_options['filtering']['cutoff'], 'bandpass', epochs_pre_ica.info['sfreq'], ica_options['filtering']['order_bandpass'], ica_options['filtering']['pad_time_bandpass'])
    preprocessing_info['epochs_pre_ica_filter'] = epochs_pre_ica_filter #save the filter used for pre-stim
    #create an mne.Epochs structure of the filtered data (note that the info structure is now inconsistent due to filtering outside of MNE Python)
    epochs_pre_ica_filtered = mne.EpochsArray(epochs_pre_ica_data, info=epochs_pre_ica.info, events=epochs_pre_ica.events, tmin=epochs_pre_ica.times[0])

    #set EEGs reference to average
    epochs_post.set_eeg_reference('average', projection=False, verbose=False)
    epochs_pre_ica_filtered.set_eeg_reference('average', projection=False, verbose=False)

    #get the optimal number of ICS to use in ICA
    n_components = get_number_of_components(epochs_pre_ica_filtered.get_data(copy=True), ica_options['pc_threshold'])
    #Get the ica structure fitted to epochs_pre that has been filtered with n_components
    ica, components_excluded, ic_label_dict = get_ica(epochs_pre_ica_filtered, n_components, None, ica_options['bad_component_thresholds'], ica_options['n_min_comps_to_reject'], ica_options['thresh_min_comps_to_reject'])
    preprocessing_info['ica_comps_excluded'] = components_excluded #save the excluded components and their labels
    preprocessing_info['ic_label_dict'] = ic_label_dict #iclabel dictionary
    del epochs_pre_ica_filtered #not needed anymore after ICA calibration, save memory 

    #reference and filter epochs_pre
    epochs_pre_data = epochs_pre.get_data(copy=True)
    epochs_pre_data = apply_filter_to_data(epochs_pre_data, pre_stim_filter, filter_options['pad_time'], epochs_pre.info['sfreq']) #corrected order of arguments
    epochs_pre = mne.EpochsArray(epochs_pre_data, info=epochs_pre.info, events=epochs_pre.events, tmin=epochs_pre.times[0])
    if use_ica_on_pre:
        epochs_pre.set_eeg_reference('average', projection=False, verbose=False)
    del epochs_pre_data #save memory by deleting

    if use_ica_on_pre:
        bad_ocular_trials_pre, ocular_thresholds_pre = get_bad_ocular_trials(ica, epochs_pre, trial_rejection_options['ocular']['pre_timerange_min'], None, components_excluded['eye blink'], trial_rejection_options['ocular']['z_thresh'])
        preprocessing_info['bad_trials_ocular_pre'] = bad_ocular_trials_pre
        preprocessing_info['ocular_thresholds_pre'] = ocular_thresholds_pre
    bad_ocular_trials_post, ocular_thresholds_post = get_bad_ocular_trials(ica, epochs_post, trial_rejection_options['ocular']['post_timerange'][0],
                                                    trial_rejection_options['ocular']['post_timerange'][1], components_excluded['eye blink'], trial_rejection_options['ocular']['z_thresh'])
    
    #combine and save information of trials with ocular artifacts near the TMS pulse
    if use_ica_on_pre:
        bad_ocular_trials = list(np.union1d(bad_ocular_trials_pre, bad_ocular_trials_post).astype(int))
    else:
        bad_ocular_trials = bad_ocular_trials_post
    preprocessing_info['bad_trials_ocular_post'] = bad_ocular_trials_post
    preprocessing_info['ocular_thresholds_post'] = ocular_thresholds_post
    preprocessing_info['bad_trials_ocular'] = bad_ocular_trials
    preprocessing_info['trials_before_ocular_rejection'] = epochs_pre.get_data(copy=True).shape[0]

    #now that the bad trials (regarding ocular artifacts) have been identified, suppress the bad components with ica
    if use_ica_on_pre:
        ica.apply(epochs_pre) #apply ica to epochs_pre

    ica.apply(epochs_post) #apply ica to epochs_post

    #drop the bad ocular trials
    epochs_pre, epochs_post, epochs_emg = drop_trials_from_structs([epochs_pre, epochs_post, epochs_emg], bad_ocular_trials)

    #polish the pre-stim period with filtering, cropping, downsampling, and average referencing
    epochs_pre.set_eeg_reference('average', projection=False, verbose=False) #set pre-stimulus EEG reference to average

    #find bad trials from the pre-stim period
    bad_trials_pre, good_trial_stats_pre = find_bad_trials(epochs_pre, trial_rejection_options['pre']['global_zscore_threshold'],
                                                            trial_rejection_options['pre']['local_zscore_threshold'], False, False)
    preprocessing_info['trials_before_pre_eeg_rejection'] = epochs_pre.get_data(copy=True).shape[0]
    preprocessing_info['bad_trials_pre'] = bad_trials_pre
    preprocessing_info['good_trial_stats_pre'] = good_trial_stats_pre #add statistics on accepted good trials from pre-stim
    
    epochs_pre, epochs_post, epochs_emg = drop_trials_from_structs([epochs_pre, epochs_post, epochs_emg], bad_trials_pre)


    # Apply SOUND and SSP-SIR to the post-stimulus periods only
    # Apply SOUND first
    epochs_post.apply_baseline(post_range_baseline).set_eeg_reference('average',projection=False, verbose=False) #baseline correct the data after ICA has been applied and set to average reference
    mintime = epochs_post.times[epochs_post.times > 0][0]
    preprocessing_info['post_mintime'] = mintime #save the min time that is larger than 0
    epochs_post.crop(preprocessing_info['post_mintime'], None) #crop to a valid post-stimulus-only time range

    epochs_post_data = epochs_post.get_data(copy=True) #get the post-stimulus data
    evoked_post_data = np.mean(epochs_post_data, axis=0) #mean across trials
    n_channels = evoked_post_data.shape[0] #number of channels
    # get the SOUND filter
    sound_filter, sound_sigmas, n_iters_sound, sound_convergence_reached = sound(evoked_post_data.T, 0, np.ones((n_channels, 1)), n_channels, L, 
                            sound_options['max_iters'], sound_options['lambda'], sound_options['convergence_tol'], sound_options['fixed_max_iters']) #get the SOUND filter
    
    for trial_ind in range(epochs_post_data.shape[0]): #apply SOUND for single trials
        epochs_post_data[trial_ind,:,:] = np.matmul(sound_filter, epochs_post_data[trial_ind,:,:])
        
    epochs_post = mne.EpochsArray(epochs_post_data, epochs_post.info, events=epochs_post.events, tmin=epochs_post.times[0]) #re-create post-stimulus epochs structure
    epochs_post.set_eeg_reference('average', projection=False, verbose=False) #set to average reference again
    #save SOUND information
    preprocessing_info['sound_filter'] = sound_filter #save the SOUND filter
    preprocessing_info['n_iters_sound'] = n_iters_sound #save the number of iterations used in SOUND
    preprocessing_info['sound_convergence_reached'] = sound_convergence_reached #save if SOUND converged within the number of trials or not
    preprocessing_info['sound_sigmas'] = sound_sigmas #save the sigmas from SOUND

    # Apply SSP-SIR next
    # get the filter kernel and the artifact suppression matrix P
    epochs_post_data = epochs_post.get_data(copy=True)
    evoked_post_data = np.mean(epochs_post_data, axis=0) #mean across trials
  
    data_corrected, artifact_topographies, data_suppressed, filter_kernel, P, M, PL, sir_projmat_suppr, sir_projmat_orig, n_pcs_removed = ssp_sir_to_average(
        evoked_post_data, L, epochs_post.info['sfreq'], ssp_sir_options['timerange'], method=ssp_sir_options['method'])
   
    epochs_post_data = ssp_sir_trials(epochs_post_data, P, sir_projmat_suppr, sir_projmat_orig, filter_kernel) #apply SSP-SIR for single trials
    #re-create the post-stimulus epochs structure
    epochs_post = mne.EpochsArray(epochs_post_data, epochs_post.info, events=epochs_post.events, tmin=epochs_post.times[0]) #re-create post-stimulus epochs structure
    #save SSP--SIR information
    preprocessing_info['sspsir_suppression_matrix_P'] = P
    preprocessing_info['sspsir_suppression_matrix_PL'] = PL
    preprocessing_info['sspsir_filter_kernel'] = filter_kernel
    preprocessing_info['sspsir_M'] = M
    preprocessing_info['sspsir_n_pcs_removed'] = n_pcs_removed
    preprocessing_info['sspsir_data_corrected_ave'] = data_corrected
    preprocessing_info['sspsir_artifact_topographies'] = artifact_topographies
    preprocessing_info['sspsir_data_suppressed_ave'] = data_suppressed
    preprocessing_info['sspsir_sir_projmat_suppr'] = sir_projmat_suppr
    preprocessing_info['sspsir_sir_projmat_orig'] = sir_projmat_orig

    #interpolate again the artifact time window (can be different)
    epochs_post = mne.preprocessing.fix_stim_artifact(epochs_post, tmin=preprocessing_info['post_mintime'], tmax=pulse_artifact_time2[1], mode='window')  #reconstruct a longer window
    epochs_post.set_eeg_reference('average', projection=False, verbose=False) #set the EEG to average reference

    #find bad trials from the post-stimulus period and drop them from the pre- and post-stimulus epochs
    bad_trials_post, good_trial_stats_post = find_bad_trials(epochs_post.copy().crop(post_range_reject[0],post_range_reject[1]), trial_rejection_options['post']['global_zscore_threshold'],
                                       trial_rejection_options['post']['local_zscore_threshold'], False, False)
    preprocessing_info['trials_before_post_eeg_rejection'] = epochs_post.get_data(copy=True).shape[0]
    preprocessing_info['bad_trials_post'] = bad_trials_post
    preprocessing_info['good_trial_stats_post'] = good_trial_stats_post
    #drop the same trials from all structures
    epochs_pre, epochs_post, epochs_emg = drop_trials_from_structs([epochs_pre, epochs_post, epochs_emg], bad_trials_post)
    
    #pre-process EMG data, highpass filtering and bad trial rejections
    emg_data = epochs_emg.get_data(copy=True) #get the emg data
    #apply filtering to the whole data (only highpass)
    emg_data = apply_filter_to_data(emg_data, emg_filter, filter_options_emg['pad_time'], epochs_emg.info['sfreq'])
    epochs_emg = mne.EpochsArray(emg_data, info=epochs_emg.info,  events=epochs_emg.events, tmin=epochs_emg.times[0])
    bad_trials_emg, epochs_emg, channels_ptps, pre_innervations, emg_prep_times = get_bad_trials_emg(epochs_emg, emg_trial_rejection_options['pre_innervation_options'], emg_trial_rejection_options['ptp_options'], line_freq) #check bad trials from EMG
    #save information on EMG bad trials
    preprocessing_info['emg_filter'] = emg_filter
    preprocessing_info['emg_prep_times'] = emg_prep_times
    preprocessing_info['bad_trials_emg'] = bad_trials_emg
    preprocessing_info['bad_trials_emg_peak_to_peak'] = [[trial for trial in range(len(channel_ptps)) if channel_ptps[trial] is False] for channel_ptps in channels_ptps]
    preprocessing_info['bad_trials_emg_pre_innervation_trials'] = [[trial for trial in range(len(channel_pre_innervation)) if channel_pre_innervation[trial] is True] for channel_pre_innervation in pre_innervations]
    preprocessing_info['trials_before_emg_rejection'] = epochs_emg.get_data(copy=True).shape[0]
    #drop the same trials from all structures
    epochs_pre, epochs_post, epochs_emg = drop_trials_from_structs([epochs_pre, epochs_post, epochs_emg], bad_trials_emg)
    channels_ptps = np.array([[ptp for index, ptp in enumerate(channel_ptps) if index not in bad_trials_emg] for channel_ptps in channels_ptps])
    preprocessing_info['n_trials_left'] = epochs_emg.get_data(copy=True).shape[0]

    if preprocessing_info['n_trials_left'] < n_trials_goal:
        return n_trials_goal - preprocessing_info['n_trials_left']

    return epochs_pre, epochs_post, epochs_emg, ica, channels_ptps, preprocessing_info

def get_z_scores_of_val(val, mean, std):
    return (val - mean)/std #return the z-score of the value given an existing distribution

def preprocess_pre_trial_with_calibrated_data(epoch_pre, preprocessing_info, trial_rejection_options, ica, filter_options, ica_options, use_ica_on_pre):
    """
    Preprocesses a single pre-stimulus trial using calibrated parameters,
    with detailed timing for each step.
    """
    print("\n--- Starting single trial pre-stimulus processing ---")
    performance_times = {}
    total_start_time = time.perf_counter()

    # Step 1: Reconstruct Bad Channels
    if preprocessing_info['bad_channels']:
        start_time = time.perf_counter()
        epoch_pre, _ = reconstruct_bad_channels(
            epoch_pre,
            preprocessing_info['bad_channels'],
            preprocessing_info['channel_interpolation_info']
        )
        duration = (time.perf_counter() - start_time) * 1e3
        performance_times['1. Reconstruct Bad Channels'] = duration
        print(f"Step 1 'Reconstruct Bad Channels': {duration:.3f}ms")

    # Step 2: ICA Processing (Not Implemented)
    if use_ica_on_pre:
        raise NotImplementedError("Applying ICA to pre-stimulus in single-trials is not currently implemented")

    # Step 3a: Get Data
    epoch_pre_data = epoch_pre.get_data(copy=True)

    # Step 3b: Apply Filter
    start_time = time.perf_counter()
    epoch_pre_data = apply_filter_to_data(
        epoch_pre_data,
        preprocessing_info['pre_stim_filter'],
        filter_options['pad_time'],
        epoch_pre.info['sfreq']
        
    )
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['2.2 Apply Filter'] = duration
    print(f"Step 2.2 'Apply Filter': {duration:.3f}ms")

    # Step 3c: Mean Subtraction
    start_time = time.perf_counter()
    epoch_pre_data -= np.mean(epoch_pre_data, axis=1)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['2.3 Mean Subtraction'] = duration
    print(f"Step 2.3 'Mean Subtraction': {duration:.3f}ms")

    # Step 4: Global MAD Check
    start_time = time.perf_counter()
    mad_trial = median_abs_deviation(epoch_pre_data, axis=(1, 2))[0]
    z_mad = get_z_scores_of_val(
        mad_trial,
        preprocessing_info['good_trial_stats_pre']['mads_mean'],
        preprocessing_info['good_trial_stats_pre']['mads_std']
    )
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['3.1 Global MAD Check'] = duration
    print(f"Step 3.1 'Global MAD Check': {duration:.3f}ms")
    if z_mad < trial_rejection_options['pre']['global_zscore_threshold'][0] or z_mad > trial_rejection_options['pre']['global_zscore_threshold'][1]:
        print(f"Rejected trial due to global MAD z-score {z_mad:.2f} outside threshold {trial_rejection_options['pre']['global_zscore_threshold']}")
        performance_times['Total (rejected)'] = (time.perf_counter() - total_start_time) * 1e3
        print(f"--- Total trial processing time (rejected): {performance_times['Total (rejected)'] / 1e3:.4f}s ---")
        return False, performance_times

    # Step 5: Local MAD Check
    start_time = time.perf_counter()
    mad_trials_channels = median_abs_deviation(epoch_pre_data, axis=2)
    z_score_local_trial = zscore(mad_trials_channels, axis=1)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['3.2 Local MAD Check'] = duration
    print(f"Step 3.2 'Local MAD Check': {duration:.3f}ms")
    if np.any(np.abs(z_score_local_trial) > trial_rejection_options['pre']['local_zscore_threshold']):
        print(f"Rejected trial due to local MAD z-score min/max {np.min(z_score_local_trial):.2f}/{np.max(z_score_local_trial):.2f} outside ±{trial_rejection_options['pre']['local_zscore_threshold']}")
        performance_times['Total (rejected)'] = (time.perf_counter() - total_start_time) * 1e3
        print(f"--- Total trial processing time (rejected): {performance_times['Total (rejected)'] / 1e3:.4f}s ---")
        return False, performance_times

    # Step 6: Recreate EpochsArray
    #start_time = time.perf_counter()
    epoch_pre = mne.EpochsArray(
        epoch_pre_data,
        info=epoch_pre.info,
        events=epoch_pre.events,
        tmin=epoch_pre.times[0],
        verbose=False
    )

    # Final total time
    total_duration = (time.perf_counter() - total_start_time) * 1e3
    performance_times['Total (accepted)'] = total_duration
    print(f"--- Total trial processing time (accepted): {total_duration / 1e3:.4f}s ---")

    return epoch_pre, performance_times



def preprocess_post_trial_with_calibrated_data(epoch_post, preprocessing_info, trial_rejection_options, ica, post_range_baseline, pulse_artifact_time1, pulse_artifact_time2, L, post_range_reject):
    print("\n--- Starting single trial post-stimulus processing ---")
    performance_times = {}
    total_start_time = time.perf_counter()

    # Step 1: Baseline Correction
    start_time = time.perf_counter()
    epoch_post.apply_baseline(post_range_baseline)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['1. Baseline Correction'] = duration
    print(f"Step 1 'Baseline Correction': {duration:.3f}ms")

    # Step 2: Interpolate Pulse Artifact (First Window)
    start_time = time.perf_counter()
    epoch_post = mne.preprocessing.fix_stim_artifact(epoch_post, tmin=pulse_artifact_time1[0], tmax=pulse_artifact_time1[1], mode='window')
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['2. Pulse Artifact Interpolation 1'] = duration
    print(f"Step 2 'Pulse Artifact Interpolation 1': {duration:.3f}ms")

    # Step 3: Reconstruct Bad Channels
    if preprocessing_info['bad_channels']:
        start_time = time.perf_counter()
        epoch_post, _ = reconstruct_bad_channels(epoch_post, preprocessing_info['bad_channels'], preprocessing_info['channel_interpolation_info'])
        duration = (time.perf_counter() - start_time) * 1e3
        performance_times['3. Reconstruct Bad Channels'] = duration
        print(f"Step 3 'Reconstruct Bad Channels': {duration:.3f}ms")

    # Step 4: Set EEG Reference
    start_time = time.perf_counter()
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['4. Set EEG Reference'] = duration
    print(f"Step 4 'Set EEG Reference': {duration:.3f}ms")

    # Step 5: Check ICA for Ocular Artifact
    start_time = time.perf_counter()
    ica_source_tc = ica.get_sources(epoch_post)
    ica_tc = ica_source_tc.get_data(copy=True)
    for comp_ind in preprocessing_info['ocular_thresholds_post']:
        comp_ind_dict = preprocessing_info['ocular_thresholds_post'][comp_ind]
        z_score_comp = get_z_scores_of_val(np.abs(ica_tc[0, comp_ind, comp_ind_dict['time_indices_of_interest']]), comp_ind_dict['mean'], comp_ind_dict['std'])
        if np.median(z_score_comp) > trial_rejection_options['ocular']['z_thresh']:
            print(f"Rejected trial due to near-TMS ocular activity in post: {np.median(z_score_comp)} exceeded threshold {trial_rejection_options['ocular']['z_thresh']}")
            performance_times['Total (rejected)'] = (time.perf_counter() - total_start_time) * 1e3
            print(f"--- Total trial processing time (rejected): {performance_times['Total (rejected)'] / 1e3:.4f}s ---")
            return False, performance_times
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['5. ICA Ocular Check'] = duration
    print(f"Step 5 'ICA Ocular Check': {duration:.3f}ms")

    # Step 6: Apply ICA
    start_time = time.perf_counter()
    ica.apply(epoch_post)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['6. Apply ICA'] = duration
    print(f"Step 6 'Apply ICA': {duration:.3f}ms")

    # Step 7: Re-baseline and Crop
    start_time = time.perf_counter()
    epoch_post.apply_baseline(post_range_baseline).set_eeg_reference('average', projection=False, verbose=False)
    epoch_post.crop(preprocessing_info['post_mintime'], None)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['7. Re-baseline & Crop'] = duration
    print(f"Step 7 'Re-baseline & Crop': {duration:.3f}ms")

    # Step 8: Apply SOUND
    start_time = time.perf_counter()
    epoch_post_data = epoch_post.get_data(copy=True)
    epoch_post_data = np.matmul(preprocessing_info['sound_filter'], epoch_post_data)
    epoch_post = mne.EpochsArray(epoch_post_data, epoch_post.info, events=epoch_post.events, tmin=epoch_post.times[0], verbose=False)
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['8. Apply SOUND'] = duration
    print(f"Step 8 'Apply SOUND': {duration:.3f}ms")

    # Step 9: Apply SSP-SIR
    start_time = time.perf_counter()
    epoch_post_data = epoch_post.get_data(copy=True)
    epoch_post_data = ssp_sir_single_trial(epoch_post_data[0,:,:], preprocessing_info['sspsir_suppression_matrix_P'], preprocessing_info['sspsir_sir_projmat_suppr'],
                    preprocessing_info['sspsir_sir_projmat_orig'], preprocessing_info['sspsir_filter_kernel'])
    epoch_post = mne.EpochsArray(epoch_post_data.reshape(1, epoch_post_data.shape[0], epoch_post_data.shape[1]), epoch_post.info, events=epoch_post.events, tmin=epoch_post.times[0])
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['9. Apply SSP-SIR'] = duration
    print(f"Step 9 'Apply SSP-SIR': {duration:.3f}ms")

    # Step 10: Interpolate Pulse Artifact (Second Window)
    start_time = time.perf_counter()
    epoch_post = mne.preprocessing.fix_stim_artifact(epoch_post, tmin=preprocessing_info['post_mintime'], tmax=pulse_artifact_time2[1], mode='window')
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['10. Pulse Artifact Interpolation 2'] = duration
    print(f"Step 10 'Pulse Artifact Interpolation 2': {duration:.3f}ms")

    # Step 11: Global MAD Check
    start_time = time.perf_counter()
    epoch_post_data = epoch_post.copy().crop(post_range_reject[0], post_range_reject[1]).get_data(copy=True)
    mad_trial = median_abs_deviation(epoch_post_data, axis=(1,2))[0]
    z_mad = get_z_scores_of_val(mad_trial, preprocessing_info['good_trial_stats_post']['mads_mean'], preprocessing_info['good_trial_stats_post']['mads_std'])
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['11. Global MAD Check'] = duration
    print(f"Step 11 'Global MAD Check': {duration:.3f}ms")
    if z_mad < trial_rejection_options['post']['global_zscore_threshold'][0] or z_mad > trial_rejection_options['post']['global_zscore_threshold'][1]:
        print(f"Rejected trial due to median absolute deviation in post: {z_mad} exceeded thresholds {trial_rejection_options['post']['global_zscore_threshold']}")
        performance_times['Total (rejected)'] = (time.perf_counter() - total_start_time) * 1e3
        print(f"--- Total trial processing time (rejected): {performance_times['Total (rejected)'] / 1e3:.4f}s ---")
        return False, performance_times

    # Step 12: Local MAD Check
    start_time = time.perf_counter()
    mad_trials_channels = median_abs_deviation(epoch_post_data, axis=2)
    z_score_local_trial = zscore(mad_trials_channels, axis=1)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['12. Local MAD Check'] = duration
    print(f"Step 12 'Local MAD Check': {duration:.3f}ms")
    if np.any(np.abs(z_score_local_trial) > trial_rejection_options['post']['local_zscore_threshold']):
        print(f"Rejected trial due to local median absolute deviation in post: {np.min(z_score_local_trial)} or {np.max(z_score_local_trial)} exceeded thresholds ±{trial_rejection_options['post']['local_zscore_threshold']}")
        performance_times['Total (rejected)'] = (time.perf_counter() - total_start_time) * 1e3
        print(f"--- Total trial processing time (rejected): {performance_times['Total (rejected)'] / 1e3:.4f}s ---")
        return False, performance_times

    # Final total time
    total_duration = (time.perf_counter() - total_start_time) * 1e3
    performance_times['Total (accepted)'] = total_duration
    print(f"--- Total trial processing time (accepted): {total_duration / 1e3:.4f}s ---")

    return epoch_post, performance_times



def preprocess_emg_trial(epoch_emg, filter_options_emg, emg_trial_rejection_options, preprocessing_info, line_freq):
    """
    Preprocess a single EMG trial and determine MEP validity, with timing for each major step.
    """
    import time
    print("\n--- Starting single trial EMG processing ---")
    performance_times = {}
    total_start_time = time.perf_counter()

    # Step 1: Extract time windows and raw data
    full_emg_times = preprocessing_info['emg_prep_times']['full_emg_times']
    pre_stim_times = full_emg_times[preprocessing_info['emg_prep_times']['pre_innervation_time_indices']]
    emg_data = epoch_emg.get_data(copy=True)

    # Step 2: Apply EMG Filter
    start_time = time.perf_counter()
    emg_data = apply_filter_to_data(emg_data, preprocessing_info['emg_filter'], filter_options_emg['pad_time'], epoch_emg.info['sfreq'])
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['2. Apply EMG Filter'] = duration
    print(f"Step 2 'Apply EMG Filter': {duration:.3f}ms")

    # Step 3: Pre-Innervation and MEP Check
    start_time = time.perf_counter()
    has_pre_innervation, valid_peak_to_peak, ptp, emg_data = prep_one_emg_trial(
        emg_data,
        0,
        0,
        preprocessing_info['emg_prep_times']['pre_innervation_time_indices'],
        line_freq,
        pre_stim_times,
        full_emg_times,
        emg_trial_rejection_options['pre_innervation_options'],
        preprocessing_info['emg_prep_times']['peak_to_peak_time_indices'],
        emg_trial_rejection_options['ptp_options'],
        epoch_emg.info['sfreq']
    )
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['3. Pre-Innervation & MEP Check'] = duration
    print(f"Step 3 'Pre-Innervation & MEP Check': {duration:.3f}ms")

    # Step 4: Recreate EpochsArray
    epoch_emg = mne.EpochsArray(emg_data, info=epoch_emg.info, events=epoch_emg.events, tmin=epoch_emg.times[0], verbose=False)

    # Final Step: Return decision
    total_duration = (time.perf_counter() - total_start_time) * 1e3
    if has_pre_innervation or not valid_peak_to_peak:
        print(f"Rejected EMG trial: Pre-innervation = {has_pre_innervation}, Valid PTP = {valid_peak_to_peak}")
        performance_times['Total (rejected)'] = total_duration
        print(f"--- Total EMG trial processing time (rejected): {total_duration / 1e3:.4f}s ---")
        return False, False, performance_times
    else:
        performance_times['Total (accepted)'] = total_duration
        print(f"--- Total EMG trial processing time (accepted): {total_duration / 1e3:.4f}s ---")
        return epoch_emg, ptp, performance_times
    


def run_subject_processing(site_id: str, subject_id: str):
    """
    Main logic to preprocess data for a single subject.
    """
    # Paths and settings
    all_data_path = Path("/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/preprocessing/data_epoched/raw_eeglab_and_block_idents")
    use_ica_on_pre = False
    save_results = True
    processed_path = Path(f"/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/preprocessing/data_processed_final_pre_ica_{use_ica_on_pre}_final_v4")

    # Time ranges
    pre_range = [-0.505, -0.005]
    post_range = [-0.03, 0.1]
    post_range_baseline = [-0.025, -0.015]
    post_range_reject = [0.02, 0.06]
    pulse_artifact_time1 = [-0.014, 0.014]
    pulse_artifact_time2 = [None, 0.015]
    post_range_times = {'post_range': post_range, 'baseline': post_range_baseline, 'pulse_artifact_time1': pulse_artifact_time1, 'post_range_reject': post_range_reject, 'pulse_artifact_time2': pulse_artifact_time2}
    pre_range_times = {'pre_range_final': pre_range}
    emg_times = [-0.5, 0.2]

    # Frequency ranges
    freq_range_suspect_channels = [30, 47]
    freq_range_suspect_trials = [30, 47]
    
    # Channel rejection options
    channel_rejection_options_pre = {'z_score_thresh_mad': [-3, 3], 'z_score_thresh_power': 5, 'fmin_fmax': freq_range_suspect_channels, 'z_score_thresh_autocorr': 4}
    channel_rejection_options_post = {'z_score_thresh_auc': 3}
    channel_rejection_options = {'pre': channel_rejection_options_pre, 'post': channel_rejection_options_post}

    # ICA options
    ica_options = {'pc_threshold': 0.99, 'bad_component_thresholds': {'eye blink': 0.9}, 'n_min_comps_to_reject': {'eye blink': 2}, 'thresh_min_comps_to_reject': {'eye blink': 0.7}, 'pre_timerange': [-1.1, -0.005], 'filtering': {'order_bandpass': 2, 'order_bandstop': 2, 'pad_time_bandpass': 0.5, 'cutoff': [1, 100]}}

    # SOUND and SSP-SIR options
    sound_options = {'max_iters': 10, 'lambda': 0.01, 'convergence_tol': 1e-9, 'fixed_max_iters': False}
    ssp_sir_options = {'timerange': ['automatic', 50], 'method': ['threshold', 0.99]}

    # Trial rejection options
    trial_rejection_options = {'ocular': {'z_thresh': 2, 'pre_timerange_min': -0.1, 'post_timerange': [0.015, None]}, 'pre': {'global_zscore_threshold': [-8, 4], 'local_zscore_threshold': 5, 'psd_zscore_threshold': False, 'psd_freq_range': freq_range_suspect_trials}, 'post': {'global_zscore_threshold': [-np.inf, 6], 'local_zscore_threshold': 5}}

    # EMG trial rejection options
    pre_innervation_options = {'tmin': -0.2, 'tmax': -0.015, 'threshold': 50 * 1e-6}
    ptp_options = {'tmin': 0.02, 'tmax': 0.05, 'min_ptp_height': 50 * 1e-6, 'min_distance': 0.005, 'prominence': 10 * 1e-6, 'check_ptp': False}
    emg_trial_rejection_options = {'pre_innervation_options': pre_innervation_options, 'ptp_options': ptp_options}

    # Channel and forward model setup
    common_channels = ['AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8']
    montage_name = 'standard_1005'
    montage = mne.channels.make_standard_montage(montage_name)
    forward = mne.read_forward_solution(r"/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/preprocessing/extracted_files/subjects_dir_fsaverage/fsaverage/fsaverage-fwd.fif")
    L = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0)
    channel_order = forward.ch_names
    
    # Filter options
    filter_options = {'cutoff': [2, 47], 'btype': 'bandpass', 'order': 2, 'pad_time': 0.1}
    filter_options_emg = {'cutoff': 2, 'btype': 'highpass', 'order': 4, 'pad_time': 0.5}
    
    # General parameters
    line_freq = 50
    resample_to = 1000
    n_trials_goal = 100

    # --- Processing logic for a single subject ---
    print(f"--- Starting processing for site: {site_id}, subject: {subject_id} ---")
    start_time_total = time.time()

    subject_path = all_data_path / site_id / subject_id
    subject_path_processed = processed_path / subject_id

    if not subject_path.exists():
        print(f"ERROR: Subject directory not found at {subject_path}. Skipping.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(subject_path_processed, exist_ok=True)

    epochs = mne.read_epochs_eeglab(os.path.join(subject_path, f'{subject_id}_task-tep_all_eeg.set'))
    emg_channel_names = [ch_name for ch_name in epochs.ch_names if 'emg' in ch_name.lower() or 'apb' in ch_name.lower() or 'fdi' in ch_name.lower()]
    emg_epochs = epochs.copy().pick(emg_channel_names)
    epochs.pick(common_channels)
    epochs.reorder_channels(channel_order)
    epochs.set_montage(None)
    epochs.set_montage(montage)

    block_identifiers = scipy.io.loadmat(os.path.join(subject_path, f'{subject_id}_block_identifiers.mat'), simplify_cells=True)['block_identifiers_trials']

    if channel_order != epochs.ch_names:
        raise ValueError(f"Wrong channel order {channel_order} vs {epochs.ch_names}")

    n_trials_use = n_trials_goal + 25
    check_emg_epochs = emg_epochs.copy()[0:n_trials_use].crop(emg_times[0], emg_times[1]).resample(resample_to, method='polyphase')
    picked_channel, emg_filter = get_better_emg_channel(check_emg_epochs, ptp_options, filter_options_emg)
    emg_epochs.pick(picked_channel)
    calibration_processing_times = []
    trials_in_calibration = []
    while True:
        start = time.time()
        epochs_now = epochs[0:n_trials_use]
        emg_epochs_now = emg_epochs[0:n_trials_use]
        out = preprocess_prestim_and_poststim(epochs_now, emg_epochs_now, pre_range_times, post_range_times, emg_times,
                                                    channel_rejection_options, ica_options, trial_rejection_options, emg_trial_rejection_options, sound_options,
                                                    ssp_sir_options, L, filter_options, filter_options_emg, line_freq, resample_to, n_trials_goal, use_ica_on_pre, emg_filter)
        
        end = time.time()
        calibration_processing_times.append(end - start)
        trials_in_calibration.append(n_trials_use)
        if isinstance(out, int):
            n_trials_use += out
            print(f"Adjusting n_trials_use to {n_trials_use}")
        else:
            break
    
    processing_times = {'pre': {'good': [], 'bad': [], 'all': []}, 'post': {'good': [], 'bad': [], 'all': []}, 'emg': {'good': [], 'bad': [], 'all': []},
                        'calibration_processing_times': calibration_processing_times, 'trials_in_calibration': trials_in_calibration, 'n_calibration_iterations': len(trials_in_calibration)}
    
    epochs_pre_calibration, epochs_post_calibration, epochs_emg_calibration, ica, channels_ptps, preprocessing_info = out

    preprocessing_info['n_trials_calibration'] = len(epochs_pre_calibration)
    preprocessing_info['used_emg_channel'] = picked_channel
    preprocessing_info['n_trials_used_in_calibration'] = n_trials_use
    preprocessing_info['n_trials_original'] = len(epochs)
    
    epochs_pre_list = [epochs_pre_calibration]
    epochs_post_list = [epochs_post_calibration]
    epochs_emg_list = [epochs_emg_calibration]
    bad_trials_calibrated = []
    ch_ptps = []
    preprocessing_info['n_trials_to_process_with_calibrated_filters'] = len(epochs) - n_trials_use
    
    for trial_ind in range(n_trials_use, len(epochs)):
        print(f"Processing trial {trial_ind}...")
        epoch = epochs[trial_ind]
        epoch_emg = emg_epochs[trial_ind]

        epoch_pre_now = epoch.copy().crop(pre_range_times['pre_range_final'][0], pre_range_times['pre_range_final'][1]).resample(resample_to, method='polyphase')
        start_pre = time.perf_counter()
        epoch_pre, processing_times_pre = preprocess_pre_trial_with_calibrated_data(
            epoch_pre_now, preprocessing_info, trial_rejection_options, ica, filter_options, ica_options, use_ica_on_pre
        )
        end_pre = time.perf_counter()
        processing_times['pre']['all'].append(processing_times_pre)

        epoch_post_now = epoch.copy().crop(post_range_times['post_range'][0], post_range_times['post_range'][1]).resample(resample_to, method='polyphase')
        start_post = time.perf_counter()
        epoch_post, processing_times_post = preprocess_post_trial_with_calibrated_data(
            epoch_post_now, preprocessing_info, trial_rejection_options, ica, post_range_baseline, pulse_artifact_time1, pulse_artifact_time2, L, post_range_reject
        )
        end_post = time.perf_counter()
        processing_times['post']['all'].append(processing_times_post)

        epoch_emg.crop(emg_times[0], emg_times[1]).resample(resample_to, method='polyphase')
        start_emg = time.perf_counter()
        epoch_emg, mep_ptp, processing_times_emg = preprocess_emg_trial(
            epoch_emg, filter_options_emg, emg_trial_rejection_options, preprocessing_info, line_freq
        )
        end_emg = time.perf_counter()
        processing_times['emg']['all'].append(processing_times_emg)

        if epoch_pre and epoch_post and epoch_emg:
            epochs_pre_list.append(epoch_pre)
            epochs_post_list.append(epoch_post)
            epochs_emg_list.append(epoch_emg)
            ch_ptps.append(mep_ptp)
            processing_times['pre']['good'].append(end_pre - start_pre)
            processing_times['post']['good'].append(end_post - start_post)
            processing_times['emg']['good'].append(end_emg - start_emg)
        else:
            bad_trials_calibrated.append(trial_ind)
            processing_times['pre']['bad'].append(end_pre - start_pre)
            processing_times['post']['bad'].append(end_post - start_post)
            processing_times['emg']['bad'].append(end_emg - start_emg)

    preprocessing_info['bad_trials_calibrated'] = bad_trials_calibrated
    preprocessing_info['processing_times'] = processing_times
    epochs_pre = mne.concatenate_epochs(epochs_pre_list)
    epochs_post = mne.concatenate_epochs(epochs_post_list)
    epochs_emg = mne.concatenate_epochs(epochs_emg_list)
    
    channels_ptps = np.append(channels_ptps, ch_ptps)

    # Metadata handling
    bad_keys = ['bad_trials_ocular', 'bad_trials_pre', 'bad_trials_post', 'bad_trials_emg']
    block_idents = block_identifiers[:n_trials_use]
    for key in bad_keys:
        current_good_trials = np.array([index for index in range(len(block_idents)) if index not in preprocessing_info[key]])
        block_idents = block_idents[current_good_trials]
    good_trials_calibrated = np.array([index for index in range(n_trials_use, len(epochs)) if index not in preprocessing_info['bad_trials_calibrated']])
    block_idents2 = block_identifiers[good_trials_calibrated]
    block_identifiers = np.concatenate((block_idents, block_idents2))
    preprocessing_info['block_identifiers'] = block_identifiers
    preprocessing_info['n_trials_final'] = len(epochs_pre)

    # --- Saving results and plots ---
    if save_results:
        # Save epoch structures
        for epochs_structure, name_attr in zip([epochs_pre, epochs_post, epochs_emg], ['pre', 'post', 'emg']):
            filename = f"{subject_id}_{name_attr}.fif"
            filepath = os.path.join(subject_path_processed, filename)
            epochs_structure.save(filepath, overwrite=True)
        
        # Save other data
        ica.save(os.path.join(subject_path_processed, f'{subject_id}_ica.fif'), overwrite=True)
        np.savez(os.path.join(subject_path_processed, f'{subject_id}_preprocessing_info.npz'), **preprocessing_info)
        np.save(os.path.join(subject_path_processed, f'{subject_id}_block_identifiers.npy'), block_identifiers)
        np.save(os.path.join(subject_path_processed, f'{subject_id}_MEPs.npy'), channels_ptps)
        
        # Save plots instead of showing them
        try:
            fig = epochs_post.average().plot_joint(show=False)
            fig.savefig(os.path.join(subject_path_processed, f'{subject_id}_post_joint.png'))
            plt.close(fig)

            fig, axs = plt.subplots()
            axs.plot(channels_ptps * 1e6)
            axs.set_title(f'MEP PTP values for {subject_id}')
            axs.set_xlabel('Trial')
            axs.set_ylabel('PTP (\u00b5V)')
            fig.savefig(os.path.join(subject_path_processed, f'{subject_id}_mep_ptp.png'))
            plt.close(fig)

            fig = epochs_emg.average().crop(-0.01, 0.1).plot(show=False)
            fig.savefig(os.path.join(subject_path_processed, f'{subject_id}_emg_avg.png'))
            plt.close(fig)

        except Exception as e:
            print(f"Could not generate plots for {subject_id}. Error: {e}")

    # --- Cleanup ---
    print(f"--- Finished processing for {subject_id}. Total time: {time.time() - start_time_total:.2f}s ---")
    del epochs, emg_epochs, epochs_pre, epochs_post, epochs_emg
    del epochs_pre_list, epochs_post_list, epochs_emg_list
    del epochs_pre_calibration, epochs_post_calibration, epochs_emg_calibration
    del preprocessing_info, processing_times, ica, channels_ptps, ch_ptps
    gc.collect()

# %%
def main():
    """
    Parses command-line arguments and runs the subject processing pipeline.
    """
    # The argparse module is already imported at the top of your script.
    parser = argparse.ArgumentParser(description="Run preprocessing for a single subject.")
    parser.add_argument("--site", required=True, type=str, help="The site identifier for the subject.")
    parser.add_argument("--subject", required=True, type=str, help="The subject identifier.")
    
    args = parser.parse_args()
    
    print(f"Received arguments: site='{args.site}', subject='{args.subject}'")
    
    # Call the main function that contains all the processing logic
    run_subject_processing(site_id=args.site, subject_id=args.subject)

if __name__ == "__main__":
    main()