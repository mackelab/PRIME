#%%
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
import mne_icalabel
import numpy as np
#import warnings

def get_ica(epochs_or_raw, n_components, picks, bad_component_thresholds, n_min_comps_to_reject, thresh_min_comps_to_reject):
    #warnings.filterwarnings('ignore') #stop warning about high-pass problems, because the data has been high-pass filtered but it is not marked epochs.info["highpass"] (could not be set directly)
    ica = ICA(n_components=n_components, random_state=97, method='infomax', fit_params=dict(extended=True),verbose=False) #initialize ICA
    ica.fit(epochs_or_raw, picks=picks, verbose=False) #fit ICA to the structure using picks channels (if picks=None, then use all channels)

    bad_component_names = list(bad_component_thresholds.keys()) #bad component names
    ic_labels_dict = mne_icalabel.label_components(epochs_or_raw, ica, method='iclabel') #label the components using icalabel
    ic_labels = ic_labels_dict['labels'] #get the component labels
    ic_probabilities = ic_labels_dict['y_pred_proba'] #get the label probabilities
    exclude_ic_indices = []
    component_indices_excluded = {}
    for bad_component_name in bad_component_names:
        # get the indices to exclude (those that exceed the respective probability threshold)
        exclude_ic_indices_probabilities_comp = [(probability, index) for index, (label, probability) in enumerate(zip(ic_labels, ic_probabilities))
                                             if label==bad_component_name and probability > bad_component_thresholds[bad_component_name]]
        if len(exclude_ic_indices_probabilities_comp) > 0:
            exclude_ic_indices_comp = np.array([pair[1] for pair in exclude_ic_indices_probabilities_comp])
            probabilities = [pair[0] for pair in exclude_ic_indices_probabilities_comp]
            print(f"Found {len(probabilities)} {bad_component_name} components with probabilities: {probabilities} with the threshold {bad_component_thresholds[bad_component_name]}.")
        else:
            print(f"Did not find {bad_component_name} components with the threshold {bad_component_thresholds[bad_component_name]}.")
            exclude_ic_indices_comp = np.array([])
        if n_min_comps_to_reject[bad_component_name] is not False:
            if len(exclude_ic_indices_comp) < n_min_comps_to_reject[bad_component_name]:
                n_comps_left_to_reject = n_min_comps_to_reject[bad_component_name] - len(exclude_ic_indices_comp)
                #we need to check whether an additional component or components should be removed

                #get probabilites and indices of remaining components of the bad_component_name
                print(f"Not enough {bad_component_name} components were rejected (found {len(exclude_ic_indices_comp)} components out of requested {n_min_comps_to_reject[bad_component_name]}).\nChecking all {bad_component_name} component probabilities...")
                probabilities_indices_left = [(probability, index) for index, (label, probability) in enumerate(zip(ic_labels, ic_probabilities))
                                    if label==bad_component_name and probability > thresh_min_comps_to_reject[bad_component_name] and index not in exclude_ic_indices_comp]
                if len(probabilities_indices_left) > 0:
                    probabilities = [pair[0] for pair in probabilities_indices_left]
                    indices = [pair[1] for pair in probabilities_indices_left]
                    print(f"Found {len(probabilities)} {bad_component_name} components with probabilities: {probabilities}, with the threshold {thresh_min_comps_to_reject[bad_component_name]}.")
                    sorted_probabilities, sorted_indices = zip(*sorted(zip(probabilities, indices), reverse=True)) #probabilities and indices in descending order
                    sorted_probabilities, sorted_indices = np.array(sorted_probabilities), np.array(sorted_indices) #transform to numpy arrays
                    inds_over_probability_threshold = np.where(sorted_probabilities > thresh_min_comps_to_reject[bad_component_name])[0] #indices that go over the threshold
                    if len(inds_over_probability_threshold) > n_comps_left_to_reject:
                        inds_over_probability_threshold = inds_over_probability_threshold[0:n_comps_left_to_reject]
                    n_comps_to_additionally_reject = len(inds_over_probability_threshold)
                    if n_comps_to_additionally_reject > 0: #if at least one was over the threshold
                        exclude_ic_indices_comp = np.union1d(exclude_ic_indices_comp, sorted_indices[0:n_comps_to_additionally_reject])
                        print(f"Found additional {n_comps_to_additionally_reject} components with probability {sorted_probabilities[0:n_comps_to_additionally_reject]}")
                    else:
                        print(f"Did not find more suitable components to reject.")
                else:
                    print(f"Did not find more suitable components to reject.")
            component_indices_excluded[bad_component_name] = list(exclude_ic_indices_comp.astype(int))
            exclude_ic_indices = np.union1d(exclude_ic_indices, exclude_ic_indices_comp) #union of bad ic indices
            
    ica.exclude = list(exclude_ic_indices.astype(int)) #mark the indices of bad components exceeding the probability threshold to be rejected
    return ica, component_indices_excluded, ic_labels_dict #return the ICA structure with (most likely) some of the indices marked bad and other info

def apply_ica_from(epochs_from, epochs_to, picks, n_ic_components, bad_component_thresholds, n_min_comps_to_reject, thresh_min_comps_to_reject):
    ica = get_ica(epochs_from, n_ic_components, picks, bad_component_thresholds, n_min_comps_to_reject, thresh_min_comps_to_reject) #get the ic decomposition
    ica.apply(epochs_to) #apply the ic decomposition filters to epochs_to
    return epochs_to, ica

def get_number_of_components(data, pc_threshold):
    #organize the data to a more proper format for PCA
    n_channels = data.shape[1] #number of channels in the data
    data_reshaped = data.transpose(0,2,1).reshape(-1, n_channels) #reshape the data to n_trials * n_samples x n_channels
    #intialize the PCA instance and fit PCA to the data
    pca = PCA()
    pca.fit(data_reshaped)
    cumulative_variances = np.cumsum(pca.explained_variance_ratio_) #get the cumulative sum of explained variances
    n_components = np.argmax(cumulative_variances > pc_threshold) + 1 #get the first index of 'True' and add 1 to account for 0-based indexing
    print(f"Selecting {n_components} components for ICA based on PCA.")
    return n_components

