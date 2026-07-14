import torch
from torch import nn

class Temporal(nn.Module):
    def __init__(self, input_channels=16, output_features=128):
        
        super(Temporal, self).__init__()

        self.conv3 = nn.Sequential(
            nn.Conv1d(input_channels, 32, 3, padding=1), 
            nn.BatchNorm1d(32), 
            nn.ReLU()
        )

        self.conv15 = nn.Sequential(
            nn.Conv1d(input_channels, 32, 15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.conv31 = nn.Sequential(
            nn.Conv1d(input_channels, 32, 31, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.proj = nn.Conv1d(96, 128, kernel_size=1)

        self.resnet1d = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128)
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  
            nn.Conv1d(128, 8, 1),
            nn.ReLU(),
            nn.Conv1d(8, 128, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):

        x1 = self.conv3(x)
        x2 = self.conv15(x)
        x3 = self.conv31(x)
        
        x = torch.cat([x1, x2, x3], dim=1)   
 
        residual = self.proj(x)              
        out = self.resnet1d(x)                
        x = torch.relu(out + residual)       
 
        se = self.se(x)                       
        x = x * se                           
 
        return x 

    
