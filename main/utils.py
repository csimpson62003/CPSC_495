"""
Utility functions for device setup, random seeding, and visualization.
Provides helper functions for training and inference workflows.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from einops import rearrange
from typing import List


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def setup_cuda_device(preferred_gpu: int = 0):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    if preferred_gpu < torch.cuda.device_count():
        selected_gpu = preferred_gpu
    else:
        selected_gpu = 0
    
    torch.cuda.set_device(selected_gpu)
    device = torch.device(f"cuda:{selected_gpu}")
    
    return device


def display_reverse(images: List):
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        x = (x + 1) / 2
        x = np.clip(x, 0, 1)
        ax.imshow(x)
        ax.axis('off')
    plt.show()
