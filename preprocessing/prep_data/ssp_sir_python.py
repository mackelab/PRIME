import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import svd
from scipy.stats import zscore
"""
Translated from https://github.com/nigelrogasch/TESA/blob/master/tesa_sspsir.m by Oskari Ahola (oskari.ahola@aalto.fi) with modifications

%                     [1] Mutanen, T. P., Kukkonen, M., Nieminen, J. O., Stenroos, M., Sarvas, J.,
%                     & Ilmoniemi, R. J. (2016). Recovering TMS-evoked EEG responses masked by
%                     muscle artifacts. Neuroimage, 139, 157-166.
%
%                     [2] Biabani, M, Fornito, A, Mutanen, T. P., Morrow, J, & Rogasch, N. C.(2019).
%                     Characterizing and minimizing the contribution of sensory inputs to TMS-evoked
%                     potentials.Brain Stimulation, 12(6):1537-1552.
"""


def ssp_sir_to_average(data, L, fs, timerange, high_cutoff=100, order=2, method=['threshold',0.9]):
    print("Performing SSP-SIR to the average response...")
    if timerange[0] == 'automatic': #timerange = ['automatic',window_length]
        n_channels = data.shape[0] #number of channels in the data
        # get data that is over 100 Hz to estimate muscle artifacts
        b, a = butter(order,high_cutoff/(fs/2),btype='high',analog=False) #digital (analog=False) 2nd order butterworth filter coefficients
        data_high = filtfilt(b,a,data,axis=-1, padlen=data.shape[1]-1) #data high-pass filtered to over "high_cutoff" frequencies

        #use a 50-ms sliding window approach to estimate the muscle artifact
        data_high_squared = data_high**2
        window_size = np.round(timerange[1]*fs/1000).astype(int) #this always corresponds to window_length ms in number of samples, e.g. for fs = 5 kHz, window_size is 250 for a window length of 50

        filter_kernel = np.array([np.convolve(data_high_squared[i, :], np.ones(window_size), mode='same') / window_size for i in range(n_channels)])
        filter_kernel = np.sum(filter_kernel,axis=0)/n_channels
        filter_kernel = filter_kernel / np.max(filter_kernel)
        filter_kernel = np.sqrt(filter_kernel)
        filter_kernel = filter_kernel.reshape(1,-1)
        filter_kernel = np.tile(filter_kernel, (n_channels, 1))
        U, singular_values, _ = svd(filter_kernel * data_high, full_matrices=False)


    #remove N principal components that in total explain at least pc_percentile components
    n_pcs_to_remove = get_n_pcs(singular_values, method)
    print(f'Removing {n_pcs_to_remove} principal components')

    M = np.linalg.matrix_rank(data) - n_pcs_to_remove #number of singular values to use

    artifact_topographies = U[:,:n_pcs_to_remove] #the n_pcs_to_remove topographies that are classified as artifactual
    
    P = np.eye(n_channels) - np.matmul(artifact_topographies,np.transpose(artifact_topographies)) #artifact suppression matrix

    # Suppress the artifacts (apply the artifact suppression matrix to the data)
    data_suppressed = np.matmul(P, data)

    # Apply SIR (source-informed reconstruction) for the suppressed data
    PL = np.matmul(P,L)
    sir_projmat_suppr =  get_sir_projmat(L, PL, M)
    suppr_data_sir = get_sir_data(data_suppressed, sir_projmat_suppr)

    # Apply SIR (source-informed reconstruction) for the original data
    sir_projmat_orig =  get_sir_projmat(L, L, M)
    orig_data_sir = get_sir_data(data, sir_projmat_orig)

    #reconstruct the data
    data_corrected = get_data_corrected(filter_kernel, suppr_data_sir, orig_data_sir)

    return data_corrected, artifact_topographies, data_suppressed, filter_kernel, P, M, PL, sir_projmat_suppr, sir_projmat_orig, n_pcs_to_remove

def ssp_sir_trials(data_all, P, sir_projmat_suppr, sir_projmat_orig, filter_kernel):
    #apply SSP-SIR across all trials
    data_corrected = np.zeros_like(data_all)
    #go through each trial and apply SSP-SIR to the trial and save the result
    for i in range(data_all.shape[0]):
        data_corrected[i,:,:] = ssp_sir_single_trial(data_all[i,:,:], P, sir_projmat_suppr, sir_projmat_orig, filter_kernel)
    return data_corrected

def ssp_sir_single_trial(trial_data, P, sir_projmat_suppr, sir_projmat_orig, filter_kernel):
    # apply SSP-SIR for a single trial
    data_suppressed = np.matmul(P,trial_data) #artifact-suppressed data
    suppr_data_sir = get_sir_data(data_suppressed, sir_projmat_suppr) #get the suppressed data that has been "source-informed-reconstructed"
    orig_data_sir = get_sir_data(trial_data, sir_projmat_orig) #get the original data that has been "source-informed-reconstructed"
    data_corrected = get_data_corrected(filter_kernel, suppr_data_sir, orig_data_sir) #the corrected data
    return data_corrected

def get_data_corrected(filter_kernel, suppr_data_sir, orig_data_sir):
    data_corrected = filter_kernel * suppr_data_sir + orig_data_sir - filter_kernel * orig_data_sir
    return data_corrected

def get_sir_projmat(L, PL_or_L, M):
    tau_proj = np.matmul(L, np.transpose(PL_or_L))
    U, singular_values, Vt = svd(tau_proj) #singular value decomposition on tau_proj
    S = np.diag(singular_values) #diagonal matrix of the singular values
    S_inv = np.zeros(shape=S.shape) #initialize inverse matrix for singular values
    S_inv[:M,:M] = np.diag(1/np.diag(S[:M,:M])) #pick M singular values
    tau_inv = np.matmul(np.matmul(np.transpose(Vt),S_inv),np.transpose(U))
    sir_projmat = np.matmul(np.matmul(L,np.transpose(PL_or_L)),tau_inv)
    return sir_projmat


def get_sir_data(data, sir_projmat):
    data_sir = np.matmul(sir_projmat, data)
    return data_sir

def get_n_pcs(singular_values, method):
    if method[0] == 'threshold':
        n_pcs_to_remove = 0
        component_percentile_sum = 0 #is zero when zero components are considered removed
        total_sum_squared = np.sum(singular_values**2)
        while component_percentile_sum < method[1]:
            n_pcs_to_remove += 1
            component_percentile_sum = (np.sum(singular_values[:n_pcs_to_remove]**2))/total_sum_squared
        return n_pcs_to_remove
    elif method[0] == 'zscore':
        singular_values_z_scored = zscore(singular_values)
        n_pcs_to_remove = len(np.where(singular_values_z_scored > method[1])[0])
        return n_pcs_to_remove

