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
seed_value = 3407
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # 如果你在使用 CUDA，还需要设置这个
np.random.seed(seed_value)


data_path = '/disk2/ybzhang/dataset/ISCSLP_Challenge/mixed/SS/audio_only'
experiments_folder = os.path.dirname(__file__)
subjects = ['SA','SB','SC','SD','SE','SF','SG','SH'] 

index = 0 
for subject in subjects:
    res = np.empty((1,5))
    logger = logging.getLogger(subject)
    logger.setLevel(logging.INFO)
    log_file = os.path.join(experiments_folder, 'logs', subject + ".log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)   
    logger.addHandler(file_handler)

    print(subject)

    fold_accuracy = []
    fold_models = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=2024)
    eeg_path = os.path.join(data_path,subject,'eeg.npy')
    label_path = os.path.join(data_path,subject,'label.npy')
    

    for fold, (train_index, test_index) in enumerate(kfold.split(range(len(np.load(label_path))))):
        train_dataset = CustomDataset(eeg_path, label_path)
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        
        test_dataset = CustomDataset(eeg_path, label_path)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_index)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)


        model = CNN()
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

        # TensorBoard
        writer = SummaryWriter(log_dir=os.path.join(experiments_folder,'tensorboard',subject,str(fold)))
        best_accuracy = 0
        min_loss = 10.0
        counter = 0

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            

            model.train()
            for i, data in enumerate(train_loader, 0):
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


            #TensorBoard
            train_accuracy = 100 * correct / total
            writer.add_scalar('Training accuracy', train_accuracy, epoch)
            writer.add_scalar('Training loss', loss, epoch)

            # validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
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

            message = 'k_fold: %d Epoch: %d Train Accuracy: %.2f %% Train loss: %.2f | Val Accuracy: %.2f %% Val loss: %.2f' % (fold+1,epoch + 1, train_accuracy,loss, val_accuracy,val_loss)
            print(message)
            logger.info(message)
            if val_loss < min_loss:
                counter = 0
                min_loss = val_loss
                torch.save(model.state_dict(), os.path.join(experiments_folder,'models',subject+str(fold+1)+"_cnn.pth"))
                print('save')
                best_accuracy = val_accuracy
                logger.info('save')
            else:
                counter += 1
                if counter >= 20:
                    print("Early stopping!")
                    break
        scheduler.step()
        fold_accuracy.append(best_accuracy)
        print('Finished Training')
        writer.close()
    res[index,:] = fold_accuracy
    index = index + 1

np.savetxt(os.path.join(experiments_folder,'result.csv'), res, delimiter=',', newline='\n')

