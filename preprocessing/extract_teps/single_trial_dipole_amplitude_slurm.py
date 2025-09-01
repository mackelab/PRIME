#%%
"""
This script calculates the dipole amplitude for each time point in the evoked object.
"""
import mne 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time 
import pandas as pd
import argparse


def dipoles_for_times(evoked, forward, tmin, tmax):
	evoked = evoked.copy().crop(tmin, tmax) #crop the data to the time range of interest
	data_measured = evoked.data #data in the evoked object
	L = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0) #leadfield in average reference
	data_measured = data_measured - np.mean(data_measured, axis=0) #data ensured to be in average reference (if already not)
	n_source_locations_x_n_orientations = L.shape[1] #number of sources
	best_dipole_per_time = [] #initialize a list for optimal dipoles for each time point
	  
	for time_index, time in enumerate(evoked.times): #go through the time range of interest

		best_r2, best_dipole_moment, best_pos_index, best_data_predicted = init_dipole_params() #initialize dipole parameters
		best_r2_default = best_r2

		for position_index, position_now in enumerate(np.arange(0, n_source_locations_x_n_orientations, 3)): #go through the source locations

			leadfield_at_pos = L[:,position_now:position_now+3] #leadfield of the current position

			dipole_moment = np.matmul(np.linalg.pinv(leadfield_at_pos), data_measured[:, time_index]) #Q = pinv(L(r))y = dipole_moment

			data_predicted = np.matmul(leadfield_at_pos, dipole_moment) #y_predicted = L(r)Q

			r2_now = r2_score(data_measured[:, time_index], data_predicted) #coefficient of determination of the dipole fit

			if r2_now > best_r2: #then update the best dipole statistics
				best_r2 = r2_now #update the best r2 that must be exceeded to update the best dipole statistic
				best_pos_index = position_index #update the best position index
				best_data_predicted = data_predicted #update the best predicted topography (forward-modeled dipole)
				best_dipole_moment = dipole_moment #update the best dipole moment
		if best_r2 == best_r2_default:
			raise ValueError("Bad R2 score, did not find a plausible dipole")
		#compute useful information for the best dipole now that all candidates have been considered
		amplitude = np.linalg.norm(best_dipole_moment) #dipole amplitude is the norm of the dipole moment
		orientation = best_dipole_moment / amplitude #unit-length dipole orientation is the dipole moment divided by amplitude
		position = forward['source_rr'][best_pos_index] #position of the dipole in the source space

		#store the statistics of the best dipole to a dictionary
		best_dipole_statistics = {'amplitude': amplitude, 'orientation':orientation, 'position': position, 'time':time,
							'position_index':best_pos_index, 'moment': best_dipole_moment, 'r2':best_r2, 'data_predicted': best_data_predicted}

		best_dipole_per_time.append(best_dipole_statistics) #store the best dipole of the current time to the list

	return best_dipole_per_time #return the list of dipoles in dictionaries for each time point


def determine_optimal_time_range(dipoles, min_window_size, max_window_size, window_size_exponent):
	n_times = len(dipoles)
	amplitudes = np.array([dipole['amplitude'] for dipole in dipoles])
	positions = np.array([dipole['position'] for dipole in dipoles])
	orientations = np.array([dipole['orientation'] for dipole in dipoles])
	if max_window_size is None:
		max_window_size = n_times

	# --- 1. Pre-computation loop ---
	window_data = []
	for window_size in range(min_window_size, max_window_size + 1):
		for start in range(n_times - window_size + 1):
			end = start + window_size

			#Amplitude
			amplitude_score = np.min(amplitudes[start:end])
            
            # Position stability
			positions_in_window = positions[start:end]
			mean_pos = np.mean(positions_in_window, axis=0)
			pos_stability = np.mean(np.linalg.norm(positions_in_window - mean_pos, axis=1))

            # Orientation stability
			orientations_in_window = orientations[start:end]
			mean_ori = np.mean(orientations_in_window, axis=0)
			mean_ori /= np.linalg.norm(mean_ori)
			dot_products = np.dot(orientations_in_window, mean_ori)
			ori_stability = np.mean(np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0))))
            
			window_data.append({
                'start': start, 'end': end, 'size': window_size,
                'pos_stability': pos_stability, 'ori_stability': ori_stability, 'min_amplitude':amplitude_score, 'time_range': (dipoles[start]['time'], dipoles[end - 1]['time']),
            })

    # --- 2. Convert to DataFrame and Rank --
	df = pd.DataFrame(window_data)
    # ranks in descending or ascending order
	df['min_amplitude_rank'] = df['min_amplitude'].rank(method='average', ascending=False)
	df['size_rank'] = df['size'].rank(method='average', ascending=True)
	df['pos_rank'] = df['pos_stability'].rank(method='average', ascending=True)
	df['ori_rank'] = df['ori_stability'].rank(method='average', ascending=True)

	df['combined_deviation_rank'] = df['pos_rank'] + df['ori_rank'] #combined deviation rank

    # --- 3. Calculate final segment score ---
    # Score rewards window size and amplitude and penalizes high (bad) combined deviation rank
	df['final_score'] = (df['size_rank']**(window_size_exponent)) / (df['combined_deviation_rank'] + df['min_amplitude_rank'])

    # --- 4. Find the best segment ---
	best_segment_row = df.loc[df['final_score'].idxmax()]
	start, end = int(best_segment_row['start']), int(best_segment_row['end'])

	time_range = (dipoles[start]['time'], dipoles[end - 1]['time'])
	dipoles_in_time_range = dipoles[start:end]
    
	return time_range, dipoles_in_time_range, df


def fit_dipoles_to_single_trials(epochs, forward, position_index, orientation, tmin, tmax):
	epochs_cropped = epochs.copy().crop(tmin, tmax) #epochs in the time range of interest
	epochs_data = epochs_cropped.get_data(copy=True) #epoched data in the time range of interest
	n_trials = epochs_data.shape[0] #number of trials in the data
	L = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0) #leadfield in average reference
	position_now = position_index*3
	leadfield_at_pos = L[:,position_now:position_now+3] #leadfield of the current position

	dipoles_for_trials = [] #initialize a list for adding dipoles for each trial to

	if orientation is not None:
		orientation_is_fixed = True
		leadfield_in_ori = np.matmul(leadfield_at_pos, orientation) #project to orientation
		leadfield_at_pos_pinv = None #no pinv needed
	else:
		orientation_is_fixed = False
		leadfield_in_ori = None
		leadfield_at_pos_pinv = np.linalg.pinv(leadfield_at_pos)

	extraction_times = [] #init list for saving extraction times

	for trial_index in range(n_trials):
		#simulate getting a single-trial epoch
		epoch_now = epochs_cropped[trial_index]
		trial_data = epoch_now.get_data(copy=True)[0,:,:] #data n_channels x n_times for the single trial data
		start_time = time.perf_counter()
		best_y_predicted, best_amplitude, ori, best_dipole_moment, best_time, best_r2, best_data_measured = get_single_trial_dipole(trial_data, trial_index, epoch_now.times, orientation,
																												 leadfield_in_ori, leadfield_at_pos, leadfield_at_pos_pinv)

		true_amplitude = np.abs(best_amplitude) if orientation is not None else best_amplitude #save magnitude and scalar amplitude separately if needed

		dipole_info = {'amplitude':true_amplitude, 'dipole':best_dipole_moment, 'time': best_time, 'position': forward['source_rr'][position_index],
				  'position_index': position_index, 'position_now': position_now, 'orientation': ori, 'r2': best_r2, 'orientation_is_fixed':orientation_is_fixed,
					'y_predicted': best_y_predicted, 'y_measured': best_data_measured, 'trial_index':trial_index,
					 'leadfield_at_pos':leadfield_at_pos, 'leadfield_at_pos_pinv':leadfield_at_pos_pinv, 'leadfield_in_ori':leadfield_in_ori}
		
		if orientation is not None: #then also save the scalar amplitude
			dipole_info['scalar_amplitude'] = best_amplitude

		dipoles_for_trials.append(dipole_info) #add the information to the list
		extraction_times.append(time.perf_counter() - start_time) #append the extraction (and saving) time

	return dipoles_for_trials, extraction_times

def get_single_trial_dipole(trial_data, trial_index, times, orientation, leadfield_in_ori, leadfield_at_pos, leadfield_at_pos_pinv):
	best_r2 = -np.inf #initialize best r2 value
	best_r2_default = best_r2

	for time_index, time in enumerate(times):

		data_measured = trial_data[:, time_index] #get the data at the current time point

		if orientation is not None:
			#scalar amplitude of the fixed source, from y_measured = amplitude * leadfield_in_ori = leadfield_in_ori * amplitude
			amplitude = np.dot(leadfield_in_ori.T,data_measured) /  np.dot(leadfield_in_ori.T, leadfield_in_ori)
			dipole_moment = orientation*amplitude #Q = aq
			ori = orientation
		else:
			dipole_moment = np.matmul(leadfield_at_pos_pinv,data_measured) #Q = pinv(L(r))y
			amplitude = np.linalg.norm(dipole_moment) #a = ||Q|| (l2-norm)
			ori = dipole_moment/amplitude #ori = Q/||Q||

		data_predicted = np.matmul(leadfield_at_pos, dipole_moment) #y_predicted = LQ

		r2_now = r2_score(data_measured, data_predicted) #coefficient of determination of the dipole fit
		
		if r2_now > best_r2: #then update the best dipole statistics
			best_r2 = r2_now #update the best r2 that must be exceeded to update the best dipole statistic
			best_y_predicted = data_predicted #update the best predicted topography (forward-modeled dipole)
			best_amplitude = amplitude #update the best amplitude
			best_dipole_moment = dipole_moment #update best dipole moment
			best_time = time #update the best time
			best_data_measured = data_measured #update the respectively "measured data"

	if best_r2 == best_r2_default:
		raise ValueError(f"Did not find a sufficient R2 for trial {trial_index}")

	return best_y_predicted, best_amplitude, ori, best_dipole_moment, best_time, best_r2, best_data_measured

def determine_optimal_ori_and_pos(dipoles, forward, evoked):
    #get the dipole amplitudes, orientations and positions
    dipole_amplitudes = np.array([dipole['amplitude'] for dipole in dipoles])
    dipole_orientations = np.array([dipole['orientation'] for dipole in dipoles])
    dipole_positions = np.array([dipole['position'] for dipole in dipoles])
    dipole_times = np.array([dipole['time'] for dipole in dipoles])
    #get weighted dipole position and orientation measures
    weighted_ori = np.average(dipole_orientations, weights=dipole_amplitudes, axis=0)
    weighted_ori /= np.linalg.norm(weighted_ori) #ensure that the orientation is unit-length
    weighted_pos = np.average(dipole_positions, weights=dipole_amplitudes, axis=0) #weighted dipole position
   
    #find the nearest valid position and position index to the weighted position
    pos_index = np.argmin(np.linalg.norm(forward['source_rr'] - weighted_pos, axis=1))
    pos_of_pos_index = forward['source_rr'][pos_index] #position of the nearest valid position index

    #determine the properties of the weighted dipole
    weighted_dipole_stats_fixed = {'dipole_moments':[], 'dipole_amplitudes': [], 'data_predicteds':[], 'data_measureds':[], 'r2_scores': [], 'times': [], 'orientation':weighted_ori, 'position':weighted_pos, 'scalar_dipole_amplitudes': []}
    weighted_dipole_stats_free = {'dipole_moments':[], 'dipole_amplitudes': [], 'data_predicteds':[], 'data_measureds':[], 'r2_scores': [], 'times': [], 'orientations':[], 'position':weighted_pos}
    evoked_cropped = evoked.copy().crop(np.min(dipole_times), np.max(dipole_times))
    evoked_data = evoked_cropped.data
    L = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0)
    position_now = pos_index*3
    leadfield_at_pos = L[:,position_now:position_now+3] #leadfield at position
    leadfield_in_ori = np.matmul(leadfield_at_pos, weighted_ori) #project to orientation
    leadfield_at_pos_pinv = np.linalg.pinv(leadfield_at_pos)

    for time_index in range(evoked_data.shape[1]): #get the dipole stats in the average response
        data_measured = evoked_data[:, time_index]
        amplitude = np.dot(leadfield_in_ori.T,data_measured) / np.dot(leadfield_in_ori.T, leadfield_in_ori)
        dipole_moment = weighted_ori*amplitude #Q = aq
        data_predicted = np.matmul(leadfield_at_pos, dipole_moment) #y_predicted = LQ
        #update dictionary
        weighted_dipole_stats_fixed['r2_scores'].append(r2_score(data_measured, data_predicted)) #coefficient of determination of the dipole fit
        weighted_dipole_stats_fixed['dipole_moments'].append(dipole_moment)
        weighted_dipole_stats_fixed['scalar_dipole_amplitudes'].append(amplitude)
        weighted_dipole_stats_fixed['dipole_amplitudes'].append(np.abs(amplitude))
        weighted_dipole_stats_fixed['data_predicteds'].append(data_predicted)
        weighted_dipole_stats_fixed['data_measureds'].append(data_measured)
        weighted_dipole_stats_fixed['times'].append(evoked_cropped.times[time_index])

    for time_index in range(evoked_data.shape[1]): #get the dipole stats in the average response for free ori
        data_measured = evoked_data[:, time_index]
        dipole_moment = np.matmul(leadfield_at_pos_pinv,data_measured)
        data_predicted = np.matmul(leadfield_at_pos, dipole_moment) #y_predicted = LQ
        amplitude = np.linalg.norm(dipole_moment)
        #update dictionary
        weighted_dipole_stats_free['r2_scores'].append(r2_score(data_measured, data_predicted)) #coefficient of determination of the dipole fit
        weighted_dipole_stats_free['dipole_moments'].append(dipole_moment)
        weighted_dipole_stats_free['dipole_amplitudes'].append(amplitude)
        weighted_dipole_stats_free['orientations'].append(dipole_moment/amplitude)
        weighted_dipole_stats_free['data_predicteds'].append(data_predicted)
        weighted_dipole_stats_free['data_measureds'].append(data_measured)
        weighted_dipole_stats_free['times'].append(evoked_cropped.times[time_index])
   
    return weighted_pos, weighted_ori, pos_index, pos_of_pos_index, weighted_dipole_stats_fixed, weighted_dipole_stats_free

def init_dipole_params():
	return -np.inf, None, None, None

def plot_info_on_evoked(evoked, subject_response_extraction_info, weighted_pos, weighted_ori, subjects_dir, subject_plot, transpath):
    if weighted_ori is None:
        weighted_dipole_stats = subject_response_extraction_info['weighted_dipole_stats_free']
    else:
        weighted_dipole_stats = subject_response_extraction_info['weighted_dipole_stats_fixed']
    print(f"time range to consider is {subject_response_extraction_info['optimal_time_range']}")
    evoked.plot_joint(list(evoked.copy().crop(subject_response_extraction_info['optimal_time_range'][0],subject_response_extraction_info['optimal_time_range'][1]).times))
    print("Best-fitting weighted dipole in the average response. Note that it is in fsaverage.")
    best_r2_score_index_of_weighted = np.argmax(weighted_dipole_stats['r2_scores'])
    best_r2_score_of_weighted = weighted_dipole_stats['r2_scores'][best_r2_score_index_of_weighted]
    amplitude = weighted_dipole_stats['dipole_amplitudes'][best_r2_score_index_of_weighted]
    time = weighted_dipole_stats['times'][best_r2_score_index_of_weighted]
    if weighted_ori is None:
        ori = weighted_dipole_stats['orientations'][best_r2_score_index_of_weighted]
    else:
        ori = weighted_ori
   
    #create a mne.Dipole object for plotting
    dipole = mne.Dipole(times=[time],
                      pos=[weighted_pos], amplitude=[amplitude],
                        ori=[ori], gof=[best_r2_score_of_weighted])
   
    #create a title for the figure
    title = f"AMP {np.round(amplitude*1e9,1)} nAm, R2 {np.round(best_r2_score_of_weighted*100,1)}%"

    #plot
    mne.viz.plot_dipole_locations(dipole, subject=subject_plot, trans=transpath,
                                subjects_dir=subjects_dir, mode='orthoview',title=title)
	
def plot_dipole_stats_over_trials(subject_response_extraction_info):
	fixed_ori_dipoles = subject_response_extraction_info['trial_dipoles_fixed_ori']
	free_ori_dipoles = subject_response_extraction_info['trial_dipoles_free_ori']
	amps_fixed = []
	amps_free = []
	gofs_fixed = []
	gofs_free = []
	amps_fixed_scalar = []
	for free_ori_dipole, fixed_ori_dipole in zip(free_ori_dipoles, fixed_ori_dipoles):
		amps_fixed.append(fixed_ori_dipole['amplitude']*1e9)
		amps_fixed_scalar.append(fixed_ori_dipole['scalar_amplitude']*1e9)
		amps_free.append(free_ori_dipole['amplitude']*1e9)
		gofs_fixed.append(fixed_ori_dipole['r2'])
		gofs_free.append(free_ori_dipole['r2'])
	fig, axs = plt.subplots(3,2,figsize=(10,5))
	axs[0,0].plot(amps_fixed)
	axs[0,0].set_title("fixed orientation (absolute amplitude)")
	axs[1,0].plot(amps_fixed_scalar)
	axs[1,0].set_title("fixed orientation (scalar amplitude)")
	axs[0,0].set_xticks([])
	axs[1,0].set_xticks([])
	axs[2,0].plot(amps_free)
	axs[2,0].set_title("free orientation")
	axs[0,1].plot(gofs_fixed)
	axs[0,1].set_title("r2 fixed orientation")
	axs[1,1].plot(gofs_free)
	axs[1,1].set_title("r2 free orientation")
	axs[0,1].set_xticks([])
	axs[1,1].set_xticks([])
	axs[0,1].set_xticks([])
	axs[1,1].set_xticks([])
	axs[2,1].set_xticks([])
	axs[2,1].set_yticks([])
	fig.supylabel("Dipole amplitude (nAm)")
	fig.supxlabel("Trials")
	print("amplitude correlations between fixed and free ori dipoles (absolute amplitudes)", np.corrcoef(amps_fixed, amps_free)[0,1])
	print("amplitude correlations between fixed ori dipoles (absolute amplitudes vs scalar amplitudes)", np.corrcoef(amps_fixed, amps_fixed_scalar)[0,1])
	print("gof-amplitude correlations in fixed ori dipoles", np.corrcoef(amps_fixed, gofs_fixed)[0,1])
	print("gof-scalar amplitude correlations in fixed ori dipoles", np.corrcoef(amps_fixed_scalar, gofs_fixed)[0,1])
	print("gof-amplitude correlations in free ori dipoles", np.corrcoef(amps_free, gofs_free)[0,1])
	print("gof (R2) correlations between fixed and free ori dipoles", np.corrcoef(gofs_fixed, gofs_free)[0,1])
	#plt.show()
	
def run_dipole_calculation_for_subject(
    subject, 
    subjects_directory_eeg, 
    subjects_directory_dipoles, 
    forward,
    subjects_dir_fsaverage,
    save_results=True
):
    """
    Runs the full dipole fitting and saving pipeline for a single subject.
    """
    print(f"--- Starting processing for subject: {subject} ---")
    
    # --- Define constants from the latest version of the original script ---
    subject_plot = 'fsaverage'
    transpath = os.path.join(subjects_dir_fsaverage, subject_plot, "bem", f"{subject_plot}-trans.fif")
    tmin_init, tmax_init = 0.038, 0.050  # Use updated time range
    min_window_size = 3
    max_window_size = 6
    window_size_exponent = 1.5
    n_calibration_trials = 100

    subject_directory = os.path.join(subjects_directory_eeg, subject)
    subject_directory_dipoles = os.path.join(subjects_directory_dipoles, subject)
    subject_response_extraction_info = {}

    try:
        epochs = mne.read_epochs(os.path.join(subject_directory, f'{subject}_post.fif'), verbose=False)
    except FileNotFoundError:
        print(f"ERROR: Could not find post-stimulus epoch file for subject {subject}. Skipping.")
        return

    # The channel order check from the original script
    if not epochs.info['ch_names'] == forward.ch_names:
        raise ValueError(f"Channel mismatch for subject {subject}. Aborting.")

    evoked = epochs.copy()[:n_calibration_trials].average()

    best_dipole_per_time = dipoles_for_times(evoked, forward, tmin_init, tmax_init)
    subject_response_extraction_info['best_dipole_per_time'] = best_dipole_per_time

    # Updated call to include all windowing parameters and capture all outputs
    optimal_time_range, dipoles_in_time_range, windows_df = determine_optimal_time_range(
        best_dipole_per_time, min_window_size, max_window_size, window_size_exponent
    )
    subject_response_extraction_info['optimal_time_range'] = optimal_time_range
    subject_response_extraction_info['dipoles_in_time_range'] = dipoles_in_time_range
    subject_response_extraction_info['windows_df'] = windows_df # Save the windows dataframe

    best_dipole_index = np.argmax([dipole['r2'] for dipole in dipoles_in_time_range])
    subject_response_extraction_info['best_dipole_to_evoked'] = dipoles_in_time_range[best_dipole_index]



    # Updated call to capture all position-related outputs
    weighted_pos, weighted_ori, position_index, pos_of_pos_index, weighted_dipole_stats_fixed, weighted_dipole_stats_free = determine_optimal_ori_and_pos(
        dipoles_in_time_range, forward, evoked
    )
    subject_response_extraction_info.update({
        'weighted_pos': weighted_pos,
        'weighted_ori': weighted_ori,
        'nearest_to_weighted_pos_pos_index': position_index,
        'pos_of_weighted_pos_index': pos_of_pos_index, # Save the new variable
        'weighted_dipole_stats_fixed': weighted_dipole_stats_fixed,
        'weighted_dipole_stats_free': weighted_dipole_stats_free
    })
    
    print(f"Displaying results for {subject}")
    # The plotting call is now inside the saving block for better organization

    for fixed_orientation in [weighted_ori, None]:
        dipoles_for_trials, extraction_times = fit_dipoles_to_single_trials(
            epochs, forward, position_index, fixed_orientation, optimal_time_range[0], optimal_time_range[1]
        )
        orientation_identifier = 'free_ori' if fixed_orientation is None else 'fixed_ori'
        subject_response_extraction_info[f'trial_dipoles_{orientation_identifier}'] = dipoles_for_trials
        subject_response_extraction_info[f'dipoles_{orientation_identifier}_extraction_times'] = extraction_times
        # Add the informative print statement back
        print(f"Average {orientation_identifier} dipole extraction times {np.mean(extraction_times)*1e3:.2f} ms")

    # Plotting and saving logic
    if save_results:
        os.makedirs(subject_directory_dipoles, exist_ok=True)
        output_path = os.path.join(subject_directory_dipoles, f'{subject}_response_extraction_info.npz')
        # Use np.savez_compressed for more efficient storage if desired
        np.savez(output_path, **subject_response_extraction_info)
        print(f"Results saved to {output_path}")

        try:
            # Plot evoked info
            plot_info_on_evoked(evoked, subject_response_extraction_info, weighted_pos, None, subjects_dir_fsaverage, subject_plot, transpath)
            plt.savefig(os.path.join(subject_directory_dipoles, f"{subject}_evoked_dipole_location.png"))
            plt.close()

            # Plot trial stats
            plot_dipole_stats_over_trials(subject_response_extraction_info)
            plt.savefig(os.path.join(subject_directory_dipoles, f"{subject}_trial_dipole_stats.png"))
            plt.close()
        except Exception as e:
            print(f"Warning: Could not generate plots for {subject}. Error: {e}")

    print(f"--- Finished processing for subject: {subject} ---")



def main():
    """
    Parses command-line arguments and runs the dipole calculation for a single subject.
    """
    parser = argparse.ArgumentParser(description="Run single-trial dipole amplitude calculation for one subject.")
    parser.add_argument("--subject", required=True, type=str, help="The subject identifier (e.g., 'sub-001').")
    args = parser.parse_args()

    # --- Configuration ---
    # Define a base directory for easier path management
    base_dir = "/mnt/lustre/work/macke/mwe626/repos/eegjepa/EDAPT_neurips/EDAPT_TMS/preprocessing"
    
    subjects_directory_eeg = os.path.join(base_dir, "data_processed_final_pre_ica_False_final_v4")
    subjects_directory_dipoles = os.path.join(base_dir, "dipoles_with_calibration_final_v4")
    subjects_dir_fsaverage = os.path.join(base_dir, "extracted_files", "subjects_dir_fsaverage")
    fsaverage_forward_path = os.path.join(subjects_dir_fsaverage, "fsaverage", "fsaverage-fwd.fif")

    # This list defines the desired channel set and order
    common_channels = [
        'AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
        'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2', 
        'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 
        'FC5', 'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 
        'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 
        'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8'
    ]
    
    # Ensure the main output directory exists
    os.makedirs(subjects_directory_dipoles, exist_ok=True)

    # Load the forward model once and select the common channels in a fixed order
    forward = mne.read_forward_solution(fsaverage_forward_path)
    forward = forward.pick_channels(common_channels, ordered=True)

    # Run the processing for the specified subject
    run_dipole_calculation_for_subject(
        subject=args.subject,
        subjects_directory_eeg=subjects_directory_eeg,
        subjects_directory_dipoles=subjects_directory_dipoles,
        forward=forward,  # Corrected argument name
        subjects_dir_fsaverage=subjects_dir_fsaverage, # Added necessary argument
        save_results=True
    )

if __name__ == "__main__":
    main()



