import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(17,32), padding=(8,0))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(128, 1))
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)  
        self.fc2 = nn.Linear(in_features=5, out_features=2)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        conv_out = self.conv_layer(x)
        relu_out = self.relu(conv_out)
        avg_pool_out = self.avg_pool(relu_out)
        flatten_out = torch.flatten(avg_pool_out, start_dim=1)
        fc1_out = self.fc1(flatten_out)
        sigmoid_out = self.sigmoid(fc1_out)
        dropout_out = self.dropout(sigmoid_out)
        fc2_out = self.fc2(dropout_out)

        return fc2_out