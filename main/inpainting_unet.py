"""
Mask-conditioned U-Net for image inpainting using diffusion models.
Processes both the masked image and mask to predict and fill missing regions.
"""

import torch
import torch.nn as nn
from typing import List
from .sinusoidal_embeddings import SinusoidalEmbeddings
from .unet_layer import UnetLayer


class InpaintingUNET(nn.Module):
    def __init__(self,
            Channels: List = [64, 128, 256, 512, 512, 384],
            Attentions: List = [False, True, False, False, False, True],
            Upscales: List = [False, False, False, True, True, True],
            num_groups: int = 32,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            input_channels: int = 3,
            mask_channels: int = 1,
            output_channels: int = 3,
            time_steps: int = 1000):
        super().__init__()
        
        self.num_layers = len(Channels)
        
        self.image_conv = nn.Conv2d(input_channels, Channels[0]//2, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(mask_channels, Channels[0]//2, kernel_size=3, padding=1)
        self.combine_conv = nn.Conv2d(Channels[0], Channels[0], kernel_size=3, padding=1)
        
        out_channels = (Channels[-1]//2) + Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(Channels))
        
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t, mask):
        image_features = self.image_conv(x)
        mask_features = self.mask_conv(mask)
        
        combined = torch.cat([image_features, mask_features], dim=1)
        x = self.relu(self.combine_conv(combined))
        
        residuals = []
        
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        
        return self.output_conv(self.relu(self.late_conv(x)))
