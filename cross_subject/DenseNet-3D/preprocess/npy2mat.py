import numpy as np
from scipy.io import savemat
from scipy.signal import resample

data = np.load("../SA.npy")
trials_struct = []

for i in range(16):
    raw_data = data[i]
    raw_data = raw_data[:15000,:]
    
    # down sample
    original_srate = 1000
    new_srate = 128


    num_samples = int(raw_data.shape[0] * new_srate / original_srate)


    raw_data_resampled = np.zeros((num_samples, raw_data.shape[1]))


    for ch in range(raw_data.shape[1]):
        raw_data_resampled[:, ch] = resample(raw_data[:, ch], num_samples)


    if i % 2 == 0:
        attend = 'L'
    else:
        attend = 'R'

    trial_struct = {'EEG':raw_data_resampled,'attended_ear':attend}
    trials_struct.append(trial_struct)
        
    trial_struct = {'EEG':raw_data_resampled}
    trials_struct.append(trial_struct) 

savemat('../SA.mat',{'trials': trials_struct})

