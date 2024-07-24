import numpy as np
import h5py
import torch
import config as cfg
from train_valid_and_test import train_valid_model,test_model
from AADdataset import sliding_window


def from_mat_to_tensor(raw_data):
    #transpose, the dimention of mat and numpy is contrary
    Transpose = np.transpose(raw_data)
    Nparray = np.array(Transpose)
    return Nparray

# all the number of sbjects in the experiment
# train one model for every subject

# read the data
eegname = cfg.process_data_dir + '/' +  cfg.dataset_name
eegdata = h5py.File(eegname, 'r')
data = from_mat_to_tensor(eegdata['EEG'])  # eeg data (8, 16, 17408, 7, 7)
label = from_mat_to_tensor(eegdata['ENV'])  # 0 or 1, representing the attended direction (8, 16, 17408)



# random seed
torch.manual_seed(2024)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(2024)

res = torch.zeros(8)


# get the data of all subject
eegdata = np.reshape(data,(-1, 17408, 7, 7))
windowed_data = sliding_window(eegdata, 128, 64)
eeglabel = np.reshape(label,(-1, 17408, 1))
windowed_label = sliding_window(eeglabel, 128, 64)


eegdata = windowed_data.reshape(16*8 * 271, cfg.decision_window, 7, 7)
eeglabel = windowed_label.reshape(16*8 * 271, cfg.decision_window)

# Define the number of segments for each subject
num_segments = 16 * 271

# Create dictionaries to store data and labels for each subject
eeg_data = {
    'SA': eegdata[:num_segments, :, :, :],
    'SB': eegdata[num_segments:2*num_segments, :, :, :],
    'SC': eegdata[2*num_segments:3*num_segments, :, :, :],
    'SD': eegdata[3*num_segments:4*num_segments, :, :, :],
    'SE': eegdata[4*num_segments:5*num_segments, :, :, :],
    'SF': eegdata[5*num_segments:6*num_segments, :, :, :],
    'SG': eegdata[6*num_segments:7*num_segments, :, :, :],
    'SH': eegdata[7*num_segments:8*num_segments, :, :, :],
}

eeg_label = {
    'SA': eeglabel[:num_segments, :],
    'SB': eeglabel[num_segments:2*num_segments, :],
    'SC': eeglabel[2*num_segments:3*num_segments, :],
    'SD': eeglabel[3*num_segments:4*num_segments, :],
    'SE': eeglabel[4*num_segments:5*num_segments, :],
    'SF': eeglabel[5*num_segments:6*num_segments, :],
    'SG': eeglabel[6*num_segments:7*num_segments, :],
    'SH': eeglabel[7*num_segments:8*num_segments, :],
}

# List of all subjects
subjects = ['SA', 'SB', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH']
# Loop through each subject
index = 0
for subject in subjects:
    # Create a list of all other subjects
    other_subjects = [s for s in subjects if s != subject]
    
    # Concatenate eeg_data and eeg_label for all other subjects
    concatenated_eeg_data = np.concatenate([eeg_data[s] for s in other_subjects], axis=0)
    concatenated_eeg_label = np.concatenate([eeg_label[s] for s in other_subjects], axis=0)

    train_valid_model(concatenated_eeg_data,concatenated_eeg_label,index)
    res [index] = test_model(eeg_data[subject],eeg_label[subject],index)
    index = index + 1
    print("good job!")
    

np.savetxt('result.csv', res.numpy(), delimiter=',')


