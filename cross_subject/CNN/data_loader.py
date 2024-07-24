import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = np.load(data_file)
        self.labels = np.load(label_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_sample = self.data[idx]*100000
        label = self.labels[idx]
        return torch.tensor(data_sample, dtype=torch.float), torch.tensor(label, dtype=torch.float)
    

class TestDataset(Dataset):
    def __init__(self, data_file):
        self.data = np.load(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx] * 100000
        return torch.tensor(data_sample, dtype=torch.float)


