import torch
from torch import nn

class CrossBranchAttention(nn.Module):

    def __init__(self, dim=128, num_heads=8, dropout=0.1):

        super().__init__()

        self.t_to_s = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Spectral attends to Temporal
        self.s_to_t = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
 
        self.temporal_norm = nn.LayerNorm(dim)
        self.spectral_norm = nn.LayerNorm(dim)

        self.temporal_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        self.spectral_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        self.temporal_norm2 = nn.LayerNorm(dim)
        self.spectral_norm2 = nn.LayerNorm(dim)


    def forward(self, temporal_feat, spec_feat):

        # temporal_feat: (B, 128, L)   -> (B, L, 128)
        # spec_feat:     (B, 128, H, W) -> (B, HW, 128)

         # # Temporal: (B,128,L) -> (B,L,128)
        # temporal_feat = temporal_feat.transpose(1, 2)
                      
        # # Spectral: (B,128,H,W) -> (B,HW,128)
        # B, C, H, W = spec_feat.shape
        # spec_feat = (spec_feat.view(B, C, H * W).transpose(1, 2))

        # # Temporal attends to Spectral
        # t_q = self.temporal_query(temporal_feat)
        # s_k = self.spectogram_key(spec_feat)
        # s_v = self.spectogram_value(spec_feat)

        # # https://nlp.seas.harvard.edu/annotated-transformer/
        # # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
 
        t = temporal_feat.transpose(1, 2)                         
 
        B, C, H, W = spec_feat.shape
        s = spec_feat.view(B, C, H * W).transpose(1, 2)          
 
        # Temporal attends to Spectral (Q=t, K=s, V=s) 
        t_attn, _ = self.t_to_s(query=t, key=s, value=s)
        t = self.temporal_norm(t + t_attn)                         
        t = self.temporal_norm2(t + self.temporal_ffn(t))          
 
        # Spectral attends to Temporal (Q=s, K=t, V=t) 
        s_attn, _ = self.s_to_t(query=s, key=t, value=t)
        s = self.spectral_norm(s + s_attn)                        
        s = self.spectral_norm2(s + self.spectral_ffn(s))         
 
        return t, s   # (B, L, 128), (B, HW, 128)


    
     