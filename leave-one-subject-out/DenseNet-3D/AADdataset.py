import torch
import config as cfg
from torch.utils.data import Dataset
import numpy as np

class AADdataset_1point(Dataset):
    def __init__(self, eeg, label):
        self.eeg = eeg
        self.label = label

    def __len__(self):
        return self.eeg.shape[0]*self.eeg.shape[1]

    def __getitem__(self, index):
        x = self.eeg[int(index/self.eeg.shape[1])][index%self.eeg.shape[1]]
        y = self.label[int(index/self.eeg.shape[1])][index%self.eeg.shape[1]]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# winlen must be the gcd of the 128*60, in which 128 means 1s window_len
class AADdataset_1second(Dataset):
    def __init__(self, eeg, label):
        self.eeg = eeg
        self.label = label

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, index):
        x = self.eeg[index]
        y = self.label[index][0] # the label is same across the specific length
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# winlen must be the gcd of the 128*60, in which 128 means 1s window_len
class AADdataset_test(Dataset):
    def __init__(self, eeg):
        self.eeg = eeg

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, index):
        x = self.eeg[index]
        x = torch.tensor(x, dtype=torch.float32)
        return x
    

def sliding_window(data, window_size, overlap):
    num_segments, segment_length, *other_dims = data.shape
    step = window_size - overlap
    num_windows = (segment_length - window_size) // step + 1

    windows = []
    for i in range(num_segments):
        for start in range(0, segment_length - window_size + 1, step):
            window = data[i, start:start + window_size]
            windows.append(window)
    windows = np.array(windows)

    # return windows.reshape(-1, window_size, *other_dims)
    return windows


