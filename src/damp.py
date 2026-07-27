import torch
from torch import nn
from src.temporal import Temporal
from src.spectral import Spectral
from src.cross_attention import CrossBranchAttention

class DAMP(nn.Module):

    def __init__(self, num_det_classes, num_mat_classes, 
                 task='all', use_cross_attn=True, use_ortho_loss=True, temporal_only=False):

        super().__init__()

        self.task = task
        self.use_cross_attn = use_cross_attn
        self.temporal_only = temporal_only
        self.use_ortho_loss = use_ortho_loss

        self.temporal_branch = Temporal()

        if not temporal_only:
            self.spectral_branch = Spectral()

        if use_cross_attn:
            self.cross_attn = CrossBranchAttention(dim=128, num_heads=8)

        self.temporal_pool = nn.Linear(128, 1)

        if not temporal_only:
            self.spectral_pool = nn.Linear(128,1)
 
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

        self.detection_fuse = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        self.detection_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_det_classes)
        )

        self.material_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_mat_classes)
        )

        self.distance_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, ir_1d, spec_2d):

        t_feat = self.temporal_branch(ir_1d)   

        if self.temporal_only:
            t_attn = t_feat.transpose(1,2)
            t_weights = torch.softmax(self.temporal_pool(t_attn), dim=1)
            geometric = (t_attn * t_weights).sum(dim=1)
            semantic = geometric
            self.orthogonality_loss = torch.tensor(0.0, device=geometric.device)

        else:
            s_feat = self.spectral_branch(spec_2d)  

            if self.use_cross_attn:
                t_attn, s_attn = self.cross_attn(t_feat, s_feat)
            else:
                t_attn = t_feat.transpose(1,2)
                B,C,H,W = s_feat.shape
                s_attn = ( s_feat.view(B,C,H*W).transpose(1,2))

            t_weights = torch.softmax(self.temporal_pool(t_attn), dim=1)   
            t_embed = (t_attn * t_weights).sum(dim=1)                     

            s_weights = torch.softmax(self.spectral_pool(s_attn), dim=1)  
            s_embed = (s_attn * s_weights).sum(dim=1)                      
    
            semantic  = self.semantic_proj(s_embed)  
            geometric = self.geometric_proj(t_embed) 

            if self.use_ortho_loss:
                semantic_norm = torch.nn.functional.normalize(semantic, dim=1)
                geometric_norm = torch.nn.functional.normalize(geometric, dim=1)
                self.orthogonality_loss = (
                    semantic_norm * geometric_norm
                ).sum(dim=1).abs().mean()
            else:
                self.orthogonality_loss = torch.tensor(0.0, device=geometric.device)

            fused    = self.detection_fuse(torch.cat([semantic, geometric], dim=1))
            det_out  = self.detection_head(fused)
            mat_out  = self.material_head(semantic)
            dist_out = self.distance_head(geometric)

        if self.task == 'det':
            return det_out, None, None
        
        elif self.task == 'dist':
            return None, dist_out, None
        
        elif self.task == 'mat':
            return None, None, mat_out
        else:
            return det_out, dist_out, mat_out

 
