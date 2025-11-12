"""
Noise scheduler for the DDPM diffusion process.
Manages beta and alpha schedules for adding and removing noise during training and generation.
"""

import torch
import torch.nn as nn


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int = 1000):
        super().__init__()
        
        beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0).requires_grad_(False)
        
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha_cumprod)

    def forward(self, t):
        return self.beta[t], self.alpha[t]
