import torch
from torch import nn
from src.temporal import Temporal
from src.spectral import Spectral
from src.cross_attention import CrossBranchAttention

class DampNet(nn.Module):

    def __init__(self, num_det_classes, num_mat_classes):

        super().__init__()

        # Impulse Responses
        self.temporal_branch = Temporal()

        # Spectrograms
        self.spectral_branch = Spectral()

        self.cross_attn = CrossBranchAttention(dim=128, num_heads=8)

        self.temporal_pool = nn.Linear(128, 1)
        self.spectral_pool = nn.Linear(128, 1)
 
        # Seperate features for detection & distance regression
        self.semantic_proj = nn.Sequential(  
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.geometric_proj = nn.Sequential( 
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.detection_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Linear(64, num_det_classes)
        )

        self.material_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Linear(64, num_mat_classes)
        )

        self.distance_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Linear(64, 1)
        )

        # Abalation Study - Remove a Layer is the output shit 
        # Remove Cross Attetion see if performance changes

    def forward(self, ir_1d, spec_2d):

        temporal_feat = self.temporal_branch(ir_1d)   
        spectral_feat = self.spectral_branch(spec_2d)  

        # Bi-directional Cross Attention
        temporal_attn, spectral_attn = self.cross_attn(temporal_feat, spectral_feat)

        temporal_weights = torch.softmax(self.temporal_pool(temporal_attn), dim=1)   
        temporal_embed = (temporal_attn * temporal_weights).sum(dim=1)                     

        spectral_weights = torch.softmax(self.spectral_pool(spectral_attn), dim=1)  
        spectral_embed = (spectral_attn * spectral_weights).sum(dim=1)                      
 
        semantic  = self.semantic_proj(spectral_embed)  
        geometric = self.geometric_proj(temporal_embed) 

        self.semantic  = semantic
        self.geometric = geometric
 
        det_out  = self.detection_head(semantic)
        mat_out  = self.material_head(semantic)

        dist_out = self.distance_head(geometric)
 
        return det_out, dist_out, mat_out

    @property
    def orthogonality_loss(self):
        if not hasattr(self, 'semantic'):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        s = nn.functional.normalize(self.semantic,  dim=1)
        g = nn.functional.normalize(self.geometric, dim=1)
        cos_sim = (s * g).sum(dim=1)
        return cos_sim.pow(2).mean()
 
