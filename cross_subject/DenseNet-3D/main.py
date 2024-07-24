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

res = torch.zeros((cfg.kfold_num,8))


from sklearn.model_selection import KFold,train_test_split
kfold = KFold(n_splits=cfg.kfold_num, shuffle=True, random_state=2024)


# get the data of all subject
eegdata = np.reshape(data,(-1, 17408, 7, 7))
windowed_data = sliding_window(eegdata, 128, 64)
eeglabel = np.reshape(label,(-1, 17408, 1))
windowed_label = sliding_window(eeglabel, 128, 64)


eegdata = windowed_data.reshape(16*8 * 271, cfg.decision_window, 7, 7)
eeglabel = windowed_label.reshape(16*8 * 271, cfg.decision_window)

eeg_data_SA = eegdata[:16*271,:,:,:]
eeg_data_SB = eegdata[16*271:2*16*271,:,:,:]
eeg_data_SC = eegdata[2*16*271:3*16*271,:,:,:]
eeg_data_SD = eegdata[3*16*271:4*16*271,:,:,:]
eeg_data_SE = eegdata[4*16*271:5*16*271,:,:,:]
eeg_data_SF = eegdata[5*16*271:6*16*271,:,:,:]
eeg_data_SG = eegdata[6*16*271:7*16*271,:,:,:]
eeg_data_SH = eegdata[7*16*271:8*16*271,:,:,:]

eeg_label_SA = eeglabel[:16*271,:]
eeg_label_SB = eeglabel[16*271:2*16*271,:]
eeg_label_SC = eeglabel[2*16*271:3*16*271,:]
eeg_label_SD = eeglabel[3*16*271:4*16*271,:]
eeg_label_SE = eeglabel[4*16*271:5*16*271,:]
eeg_label_SF = eeglabel[5*16*271:6*16*271,:]
eeg_label_SG = eeglabel[6*16*271:7*16*271,:]
eeg_label_SH = eeglabel[7*16*271:8*16*271,:]


for fold, (train_ids,  test_ids) in enumerate(kfold.split(eeg_data_SA)):
    train_eeg_data = np.concatenate((eeg_data_SA[train_ids] , eeg_data_SB[train_ids] , eeg_data_SC[train_ids] , eeg_data_SD[train_ids] , eeg_data_SE[train_ids] , eeg_data_SF[train_ids] , eeg_data_SG[train_ids] , eeg_data_SH[train_ids]),axis=0) 
    train_eeg_label = np.concatenate((eeg_label_SA[train_ids] , eeg_label_SB[train_ids] , eeg_label_SC[train_ids] , eeg_label_SD[train_ids] , eeg_label_SE[train_ids] , eeg_label_SF[train_ids] , eeg_label_SG[train_ids] , eeg_label_SH[train_ids]),axis=0)
    train_valid_model(train_eeg_data, train_eeg_label, fold)
    res[fold][0] = test_model(eeg_data_SA[test_ids], eeg_label_SA[test_ids],fold)
    res[fold][1] = test_model(eeg_data_SB[test_ids], eeg_label_SB[test_ids],fold)
    res[fold][2] = test_model(eeg_data_SC[test_ids], eeg_label_SC[test_ids],fold)
    res[fold][3] = test_model(eeg_data_SD[test_ids], eeg_label_SD[test_ids],fold)
    res[fold][4] = test_model(eeg_data_SE[test_ids], eeg_label_SE[test_ids],fold)
    res[fold][5] = test_model(eeg_data_SF[test_ids], eeg_label_SF[test_ids],fold)
    res[fold][6] = test_model(eeg_data_SG[test_ids], eeg_label_SG[test_ids],fold)
    res[fold][7] = test_model(eeg_data_SH[test_ids], eeg_label_SH[test_ids],fold)

print("good job!")

np.savetxt('result.csv', res.numpy(), delimiter=',')


