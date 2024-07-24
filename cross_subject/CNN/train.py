from data_loader import CustomDataset
from model_cnn import CNN
import torch
import torch.nn as nn
import logging
import os
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold,train_test_split
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

num_epochs = 200
batch_size = 64
seed_value = 1715
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value) 
np.random.seed(seed_value)


data_path = '/disk2/ybzhang/dataset/ISCSLP_Challenge/Data_for_CS/data_for_CS/audio-only'
experiments_folder = os.path.dirname(__file__)
subjects = ['SA','SB','SC','SD','SE','SF','SG','SH'] 



res = np.zeros((8,5))


fold_accuracy = []
fold_models = []
kfold = KFold(n_splits=5, shuffle=True, random_state=2024)

label_path = '/disk2/ybzhang/dataset/ISCSLP_Challenge/Data_for_CS/data_for_CS/label/SA.npy'
index = 0
for fold, (train_index, test_index) in enumerate(kfold.split(range(len(np.load(label_path))))):

    all_train_loaders = []
    all_test_loaders = []
    for subject in subjects:
        eeg_path = os.path.join('/disk2/ybzhang/dataset/ISCSLP_Challenge/Data_for_CS/data_for_CS/audio-only',subject+'.npy')
        label_path = os.path.join('/disk2/ybzhang/dataset/ISCSLP_Challenge/Data_for_CS/data_for_CS/label',subject+'.npy')

        train_dataset = CustomDataset(eeg_path, label_path)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        all_train_loaders.append(train_loader)
    
        test_dataset = CustomDataset(eeg_path, label_path)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_index)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
        all_test_loaders.append(test_loader)
    

    combined_train_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in all_train_loaders])
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
    combined_test_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in all_test_loaders])
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=False)
    test_loader_A = all_test_loaders[0]
    test_loader_B = all_test_loaders[1]
    test_loader_C = all_test_loaders[2]
    test_loader_D = all_test_loaders[3]
    test_loader_E = all_test_loaders[4]
    test_loader_F = all_test_loaders[5]
    test_loader_G = all_test_loaders[6]
    test_loader_H = all_test_loaders[7]



    model = CNN()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    best_accuracy = 0
    min_loss = 10.0
    counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        

        model.train()
        for i, data in enumerate(combined_train_loader, 0):
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

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in combined_test_loader:
                eeg, labels = data[0].to(device), data[1].to(device)
                labels = labels.view(-1).long()
                outputs = model(eeg)
                val_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        val_accuracy = 100 * correct / total

        message = 'k_fold: %d Epoch: %d Train Accuracy: %.2f %% Train loss: %.2f | Val Accuracy: %.2f %% Val loss: %.2f' % (fold+1,epoch + 1, train_accuracy,loss, val_accuracy,val_loss)
        print(message)
        if val_loss < min_loss:
            counter = 0
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(experiments_folder,'models',str(fold+1)+"_cnn.pth"))
            print('save')
            best_accuracy = val_accuracy

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader_A:
                    eeg, labels = data[0].to(device), data[1].to(device)
                    labels = labels.view(-1).long()
                    outputs = model(eeg)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
            val_accuracy = 100 * correct / total 
            #add csv
            res[0,index] = val_accuracy


            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader_B:
                    eeg, labels = data[0].to(device), data[1].to(device)
                    labels = labels.view(-1).long()
                    outputs = model(eeg)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total 
            #add csv
            res[1,index] = val_accuracy

            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader_C:
                    eeg, labels = data[0].to(device), data[1].to(device)
                    labels = labels.view(-1).long()
                    outputs = model(eeg)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total 
            #add csv
            res[2,index] = val_accuracy

            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader_D:
                    eeg, labels = data[0].to(device), data[1].to(device)
                    labels = labels.view(-1).long()
                    outputs = model(eeg)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total 
            #add csv
            res[3,index] = val_accuracy


            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader_E:
                    eeg, labels = data[0].to(device), data[1].to(device)
                    labels = labels.view(-1).long()
                    outputs = model(eeg)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total 
            #add csv
            res[4,index] = val_accuracy


            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader_F:
                    eeg, labels = data[0].to(device), data[1].to(device)
                    labels = labels.view(-1).long()
                    outputs = model(eeg)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total 
            #add csv
            res[5,index] = val_accuracy


            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader_G:
                    eeg, labels = data[0].to(device), data[1].to(device)
                    labels = labels.view(-1).long()
                    outputs = model(eeg)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total 
            #add csv
            res[6,index] = val_accuracy


            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader_H:
                    eeg, labels = data[0].to(device), data[1].to(device)
                    labels = labels.view(-1).long()
                    outputs = model(eeg)
                    val_loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total 
            #add csv
            res[7,index] = val_accuracy



        else:
            counter += 1
            if counter >= 20:
                print("Early stopping!")
                break


    index = index + 1

np.savetxt(os.path.join(experiments_folder,'result.csv'), res, delimiter=',', newline='\n')

