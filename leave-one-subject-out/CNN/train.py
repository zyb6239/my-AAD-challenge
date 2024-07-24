from data_loader import CustomDataset
from model_cnn import CNN
import torch
import torch.nn as nn
import logging
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

num_epochs = 200
batch_size = 128

data_path = '/disk2/ybzhang/dataset/ISCSLP_Challenge/Data_for_CS/data_for_CS/audio-only'
label_path = '/disk2/ybzhang/dataset/ISCSLP_Challenge/Data_for_CS/data_for_CS/label'
experiments_folder = os.path.dirname(__file__)
subjects = ['SA','SB','SC','SD','SE','SF','SG','SH']
res = torch.zeros(8)

for i in range(0,8):

    logger = logging.getLogger(subjects[i])
    logger.setLevel(logging.INFO)
    log_file = os.path.join(experiments_folder, 'logs', subjects[i] + ".log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print("leave " + subjects[i] +" out" )
    
    train_data_files = []
    train_label_files = []

    for j in range (0,8):
        if j != i:            
            train_data_files.append(os.path.join(data_path,subjects[j]+'.npy'))
            train_label_files.append(os.path.join(label_path,subjects[j]+'.npy'))
            
    print(train_data_files)
    print(train_label_files)
    train_datasets = [CustomDataset(data_file, label_file) for data_file, label_file in zip(train_data_files, train_label_files)]
    train_dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in train_datasets]
    
    val_data_file = os.path.join(data_path,subjects[i]+'.npy')
    val_label_file = os.path.join(label_path,subjects[i]+'.npy')

    val_dataset = CustomDataset(val_data_file, val_label_file)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    model = CNN()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(experiments_folder,'tensorboard',subjects[i]))

    min_loss = 10.0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for n, train_dataloader in enumerate(train_dataloaders, 0):
            for data in train_dataloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = labels.view(-1).long()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        # TensorBoard
        train_accuracy = 100 * correct / total
        writer.add_scalar('Training accuracy', train_accuracy, epoch)
        writer.add_scalar('Training loss', loss, epoch)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_dataloader:
                eeg, labels = data[0].to(device), data[1].to(device)
                labels = labels.view(-1).long()
                outputs = model(eeg)
                val_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        val_accuracy = 100 * correct / total
        writer.add_scalar('Val accuracy', val_accuracy, epoch)
        writer.add_scalar('Val loss', val_loss, epoch)

        message = 'Epoch %d: Train Accuracy: %.2f %% Train loss: %.2f | Val Accuracy: %.2f %% Val loss: %.2f' % (epoch + 1, train_accuracy,loss, val_accuracy,val_loss)
        print(message)
        logger.info(message)
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(experiments_folder,'models',subjects[i]+"_cnn.pth"))
            print('save')
            logger.info('save')
            res [i] = val_accuracy

    print('Finished Training')
    writer.close()
print(res)
np.savetxt(os.path.join(experiments_folder,'result.csv'), res, delimiter=',', newline='\n')

