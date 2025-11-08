"""
Utility Functions for the Diffusion Model
==========================================
Helper functions for setup, training, and inference.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from einops import rearrange
from typing import List


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def setup_cuda_device(preferred_gpu: int = 0):
    """
    Set up CUDA and select the best available GPU device.
    
    Args:
        preferred_gpu: Which GPU to prefer (0 for first GPU, 1 for second, etc.)
    Returns:
        torch.device: The selected device (cuda:X or cpu)
    """
    print("=" * 50)
    print("CUDA SETUP")
    print("=" * 50)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available! Using CPU instead.")
        return torch.device("cpu")
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    
    # List all available GPUs
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    # Select GPU
    if preferred_gpu < torch.cuda.device_count():
        selected_gpu = preferred_gpu
    else:
        selected_gpu = 0
        print(f"⚠️  Preferred GPU {preferred_gpu} not available, using GPU {selected_gpu}")
    
    torch.cuda.set_device(selected_gpu)
    device = torch.device(f"cuda:{selected_gpu}")
    
    print(f"✅ Selected GPU {selected_gpu}: {torch.cuda.get_device_name(selected_gpu)}")
    print("=" * 50)
    return device


def display_reverse(images: List):
    """
    Visualize the reverse diffusion process.
    Shows progression from noise to generated image.
    """
    fig, axes = plt.subplots(1, 10, figsize=(15, 3))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        x = (x + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        x = np.clip(x, 0, 1)
        ax.imshow(x)
        ax.axis('off')
    plt.show()
