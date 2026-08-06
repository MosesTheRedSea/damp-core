import torch
from torch import nn

class Spectral(nn.Module):

    # Spectrograms provide a visual snapshot of sound
    # time x frequency - amplitude is color intensity
    # pitch changes, harmonics, transient bursts, and rhythm

    # 2D Convolutions convert 1D audio into images AI recognition

    def __init__(self, input_channels=16, output_features=128, use_se=True):
        
        # (batch_size, 16, H, W)
        super(Spectral, self).__init__()
        self.use_se = use_se
        
        # Conv2d 
        self.skip = nn.Conv2d(96, 128, kernel_size=1)
        
        # Multi-Scale Convolutions different filers
        self.conv3 = nn.Sequential(
            # (16, 32, 3=kernel_size, padding=1)
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            # (16, 32, 5=kernel_size, padding=2)
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            # (16, 32, 7=kernel_size, padding=3)
            nn.Conv2d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )   

        self.conv_block = nn.Sequential(
            # (96, 128, 3=kernel_size, padding=1)
            nn.Conv2d(96,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )   

        if self.use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(128,8, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(8,128, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, input):

        # Grab Features via Convolutions different kernel sizes
        feat_1 = self.conv3(input)
        feat_2 = self.conv5(input)
        feat_3 = self.conv7(input)

        # Concat features together
        total_feat = torch.cat([feat_1, feat_2, feat_3], dim=1)

        # 1 x 1 2D Convolution to upscale
        residual = self.skip(total_feat)

        output = self.conv_block(total_feat)
        output = output + residual
        output = torch.relu(output)

        if self.use_se:
            output = output * self.se(output)

        return output 