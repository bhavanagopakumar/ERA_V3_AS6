import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):  
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.07)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.07)
        )

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

        self.conv_1x1_1 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=8,kernel_size=1,stride=1,padding=0,bias=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.07)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.07)
        )

        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2, 2)
        )

        self.conv_1x1_2 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=8,kernel_size=1,stride=1,padding=0,bias=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(8, 8, 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.07)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.07)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 10, 3), 
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv_1x1_1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv_1x1_2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x)