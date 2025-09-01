# ------------------- NB! -----------------------------------
# This is a beta version of the online Sound algorithm and is not yet fully validated.
# The given lead field matrix LFM_Aalto_ReftepPP.csv only works with the Aalto Reftep++ 64-ch channel configuration. 
# If your channel configuration is different, please make your own lead field.
# If you are planning to use this code for experiments, please contact Tuomas Mutanen and Matilda Makkonen.
# The optimal parameters might be different for your use case.
# ------------------- NB! -----------------------------------

# Authors: Tuomas Mutanen, Olli-Pekka Kahilakoski, Matilda Makkonen, Johanna Metsomaa
 
import multiprocessing
import time

import numpy as np

def sound(eeg_samples, baseline_correction, sigmas, num_of_channels, lfm, iterations, lambda0, convergence_boundary, fixed_iters=False):
    # If there are no channels, return an empty filter.
    if num_of_channels == 0:
        return np.identity(0), lambda0, np.identity(0), np.identity(0), np.identity(0)

    # Actual baseline correction for Sound data buffer
    eeg_samples = eeg_samples - baseline_correction

    # Smooth sigmas update coeff:
    sigmas_update_coeff = 0.05

    # Performs the SOUND algorithm for a given data.

    data = eeg_samples.T

    start = time.time()

    n0, _ = data.shape
    data = np.reshape(data, (n0, -1))

    LL = lfm @ lfm.T
    dn = np.empty((iterations, 1)) # Empty vector for convergences

    #################### Run beamformer SOUND #####################################################
    # See Metsomaa et al. 2024 Brain Topography for equations

    dataCov = np.matmul(data, data.T) / data.shape[1] # Estimate the data covariance matrix as sample covariance
    
    # Estimate the neuronal covariance
    LL = lfm @ (lfm.T)
    regularization_term = lambda0*np.trace(LL) / num_of_channels
    LL_reg = LL / regularization_term

    # Save the previous sigma values before the new iteration:
    sigmas_prev_update = np.copy(sigmas)

    # Iterate over channels
    n_iters = iterations
    convergence_reached = False
    for k in range(iterations):
        # Save the previous sigma values
        sigmas_old = np.copy(sigmas)
        # Update noise estimate values
        GAMMA = np.linalg.pinv(LL_reg + np.diagflat(np.square(sigmas))) # Eq. 18 in Metsomaa et al. 2024
        sigmas = [(GAMMA[:, i] / GAMMA[i, i]).T @ (dataCov @ (GAMMA[:, i] / GAMMA[i, i])) for i in range(num_of_channels)] # Eq. 20 in Metsomaa et al. 2024
        
        # Following and storing the convergence of the algorithm
        max_noise_estimate_change = np.max(np.abs(sigmas_old - sigmas) / sigmas_old)
        #print("Output: Max noise estimate change = {}".format(max_noise_estimate_change))
        if max_noise_estimate_change < convergence_boundary and not fixed_iters: # terminates the iteration if the convergence boundary is reached
            print("Output: Convergence reached after {} iterations!".format(k+1))
            n_iters = k+1
            convergence_reached = True
            break

    # Make sure sigmas is a numpy array:
    sigmas = np.array(sigmas, dtype=np.float32)
    sigmas = np.expand_dims(sigmas,axis = 1)
    # Change sigmas smoothly:
    sigmas = sigmas_update_coeff*sigmas + (1 - sigmas_update_coeff)*sigmas_prev_update

    # Final data correction based on the final noise-covariance estimate.
    # Calculates matrices needed for SOUND spatial filter (for other functions)
    #SOUND_filter = LL @ np.linalg.inv(LL + regularization_term*C_noise)
    W = np.diag(1.0 / np.squeeze(sigmas))
    WL = np.matmul(W, lfm)
    WLLW = np.matmul(WL, WL.T)
    C = (WLLW + lambda0 * np.trace(WLLW) / num_of_channels * np.eye(num_of_channels))
    SOUND_filter = np.matmul(lfm, np.matmul(WL.T, np.linalg.solve(C, W)))

    # find the best-quality channel
    best_ch = np.argmin(sigmas)
    # Calculate the relative error in the best channel caused by SOUND overcorrection:
    rel_err = np.linalg.norm(SOUND_filter[best_ch,:]@data - data[best_ch,:])/np.linalg.norm(data[best_ch,:])
    print("Output: Relative error in best channel = {}".format(rel_err))

    end = time.time()
    print("Output: SOUND update time = {:.1f} ms".format(10 ** 3 * (end-start)))

    return SOUND_filter, sigmas, n_iters, convergence_reached
