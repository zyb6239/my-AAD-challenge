import torch

model_name = 'DenseNet_37-I3D'
process_data_dir = '/disk2/ybzhang/code/second/ISCSLP/ASAD_DenseNet/4_processed_data'

dataset_name = 'ISCSLP_audio_2D.mat'

device_ids = 0
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 10
batch_size = 256
sample_rate = 128
categorie_num = 2
sbnum = 8

lr=1e-3
weight_decay=0.01

# the length of decision window
decision_window = 128 #1s



