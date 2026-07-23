import torch
from torch import nn

class Temporal(nn.Module):

    def __init__(self, input_channels=16, output_features=128):
        
         # Tensor shape into model 
        # (128, 16 channels, 1100 samples) — one sample per microphone, cropped around the echo window

        super(Temporal, self).__init__()

        # Multi-Scale Convolutions
        self.conv3 = nn.Sequential(
            # (16, 32, 3=kernel_size, padding=1)
            nn.Conv1d(input_channels, 32, 3, padding=1), 
            nn.BatchNorm1d(32), 
            nn.ReLU()
        )   

        self.conv15 = nn.Sequential(
            # (16, 32, 15=kernel_size, padding=7)
            nn.Conv1d(input_channels, 32, 15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.conv31 = nn.Sequential(
            # (16, 32, 31=kernel_size, padding=15)
            nn.Conv1d(input_channels, 32, 31, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # (32 + 32 + 32) = 96 (96, 129, kernel_size=1)
        self.proj = nn.Conv1d(96, 128, kernel_size=1)

        # 2 3-Kernel convolutions 96 dim input -> 128 dim output - extracts acoustic features
        self.resnet1d = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128)
        )

        # (Batch, 128 channels, 1100 samples)
        # helps us know which channels are most important
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  
            #  128 dim summary passes through two bottleneck layers
            nn.Conv1d(128, 8, 1),
            nn.ReLU(),
            nn.Conv1d(8, 128, 1),
            # scaling weight for each of your 128 channels
            nn.Sigmoid() # between 0.0 & 1.0
        )   

        # (Batch, 128 channels, 1) - global summary of the acoustic energy in each channel
    
    def forward(self, input):

        # Grab Features via Convolutions different kernel sizes
        feat_1 = self.conv3(input)
        feat_2 = self.conv15(input)
        feat_3 = self.conv31(input)
        
        # Concat features together
        total_feat = torch.cat([feat_1, feat_2, feat_3], dim=1)   

        # projects it to 128 dim
        residual = self.proj(total_feat)       

        # extracts deep acoustic features with 2 3 kernel conv
        out = self.resnet1d(total_feat)        

        # concats the output of the resnet1d + proj
        total_feat = torch.relu(out + residual)       

        # Squeeze-and-Excitation
        se = self.se(total_feat)             

        # original features * calculated 0 <-> 1 weights
        total_feat = total_feat * se                           
 
        return total_feat