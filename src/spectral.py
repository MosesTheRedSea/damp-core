import torch
from torch import nn

class Spectral(nn.Module):
    def __init__(self, input_channels=16, output_features=128):

        super(Spectral, self).__init__()
        self.skip = nn.Conv2d(96, 128, kernel_size=1)

        self.conv3 = nn.Sequential(
            nn.Conv2d(input_channels,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels,32,5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(input_channels,32,7,padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(96,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128,8,1),
            nn.ReLU(),
            nn.Conv2d(8,128,1),
            nn.Sigmoid()
        )
  
    def forward(self, x):

        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)

        x = torch.cat([x1, x2, x3], dim=1)
        residual = self.skip(x)

        x = self.conv_block(x)
        x = x + residual
        x = x * self.se(x)
        x = torch.relu(x)

        return x 