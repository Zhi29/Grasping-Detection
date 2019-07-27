import torch.nn as nn
from dataprocess import *


dropout_rate = 0.2
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels = 3 , out_channels = 32, kernel_size = 5, stride = 1, padding = 2), nn.ReLU(), nn.MaxPool2d(kernel_size = 2)) #320
        self.Conv2 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride = 1), nn.ReLU(), nn.MaxPool2d(kernel_size = 2)) #158
        self.Conv3 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1), nn.ReLU(), nn.MaxPool2d(kernel_size = 2)) #78
        self.Conv4 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1), nn.ReLU(), nn.MaxPool2d(kernel_size = 2)) #38
        self.Conv5 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1), nn.ReLU(), nn.MaxPool2d(kernel_size = 2)) #18
        self.Conv6 = nn.Sequential(nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1), nn.ReLU(), nn.MaxPool2d(kernel_size = 2)) #9

        self.fc1 = nn.Linear(8*8*256, 1024)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, NUM_LABELS*5)

        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.Conv6(x)

        x = x.view(x.size(0), -1)
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        x = self.activate(self.fc3(x))
        output = self.out(x)

        return output
