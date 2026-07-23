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

        # (Batch, SequenceLength, Features)
        # temporal_feat = # (Batch, 128 channels, 1100 samples)
        t = temporal_feat.transpose(1, 2) 

        # t = (Batch, 1100, 128)
        # spec_feat = (Batch, 128 channels, Height, Width)
        B, C, H, W = spec_feat.shape

        s = spec_feat.view(B, C, H * W).transpose(1, 2)

        # (Batch, Height * Width, 128)
        # both attentions computed from the ORIGINAL t/s, not a chain
        t_attn, _ = self.t_to_s(query=t, key=s, value=s)

        # for this specific microphone sample at time step X which frequency spot in the spectrogram (key) matter

        s_attn, _ = self.s_to_t(query=s, key=t, value=t)

        t = self.temporal_norm(t + t_attn)
        t = self.temporal_norm2(t + self.temporal_ffn(t))

        s = self.spectral_norm(s + s_attn)
        s = self.spectral_norm2(s + self.spectral_ffn(s))

        return t, s