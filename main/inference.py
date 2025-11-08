"""
Image Generation Through Reverse Diffusion
==========================================
Generates new images by denoising pure random noise.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from timm.utils import ModelEmaV3

from .unet import UNET
from .ddpm_scheduler import DDPM_Scheduler
from .utils import setup_cuda_device, display_reverse


def inference(checkpoint_path: str=None,
            num_time_steps: int=1000,
            ema_decay: float=0.9999):
    """Generate images using trained model."""
    
    device = setup_cuda_device(preferred_gpu=0)
    
    # Load trained model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNET().to(device)
    model.load_state_dict(checkpoint['weights'])
    
    # Use EMA weights
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps).to(device)
    
    # Timesteps for visualization
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []

    with torch.no_grad():
        model = ema.module.eval()
        
        # Generate 10 sample images
        for i in range(10):
            # Start with random noise
            z = torch.randn(1, 3, 64, 64).to(device)
            
            # Reverse diffusion process
            for t in reversed(range(1, num_time_steps)):
                t_tensor = torch.tensor([t], device=device)
                
                beta_t = scheduler.beta[t_tensor]
                alpha_t = scheduler.alpha[t_tensor]
                
                # Denoising step
                temp = (beta_t / ((torch.sqrt(1-alpha_t)) * (torch.sqrt(1-beta_t))))
                z = (1/(torch.sqrt(1-beta_t)))*z - (temp*model(z, t_tensor))
                
                # Save intermediate images
                if t in times:
                    images.append(z.cpu())
                
                # Add noise for next step
                e = torch.randn(1, 3, 64, 64, device=device)
                z = z + (e*torch.sqrt(beta_t))
            
            # Final denoising step
            t0_tensor = torch.tensor([0], device=device)
            beta_0 = scheduler.beta[t0_tensor]
            alpha_0 = scheduler.alpha[t0_tensor]
            temp = beta_0 / ((torch.sqrt(1-alpha_0)) * (torch.sqrt(1-beta_0)))
            x = (1/(torch.sqrt(1-beta_0)))*z - (temp*model(z, t0_tensor))

            images.append(x.cpu())
            
            # Display final image
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach().cpu()
            x = x.numpy()
            x = (x + 1) / 2
            x = np.clip(x, 0, 1)
            
            plt.figure(figsize=(4, 4))
            plt.imshow(x)
            plt.axis('off')
            plt.title(f"Generated Face {i+1}")
            plt.show()
            
            # Show denoising process
            display_reverse(images)
            images = []
