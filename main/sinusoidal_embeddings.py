"""
Sinusoidal position embeddings for encoding timestep information in the diffusion process.
Converts scalar timesteps into high-dimensional vectors that the model can process.
"""

import torch
import torch.nn as nn
import math


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, embed_dim: int, max_T: int = 1000):
        super().__init__()
        position = torch.arange(max_T).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(max_T, embed_dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('embeddings', embeddings)

    def forward(self, t):
        return self.embeddings[t][:, :, None, None]
