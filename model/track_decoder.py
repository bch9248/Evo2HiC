import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import RelativeTransformerEncoderLayer

class Track_Decoder(nn.Module):
    def __init__(self, dim, num_tracks, nhead=8, dim_feedforward=2048, dropout=0.05, num_layers=11, max_relative_position=64, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = nn.ModuleList([
            RelativeTransformerEncoderLayer(
                d_model=dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_relative_position=max_relative_position
            ) for _ in range(num_layers)
        ])

        self.mlp = nn.Linear(dim, num_tracks)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.mlp(x)