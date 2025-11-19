"""
Inference function for filling in masked regions using the trained diffusion model.
Performs iterative denoising to generate realistic content in missing areas.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from einops import rearrange
from timm.utils import ModelEmaV3

from .inpainting_unet import InpaintingUNET
from .ddpm_scheduler import DDPM_Scheduler
from .utils import setup_cuda_device


def load_and_preprocess_image(image_path: str, size: int = 16):
    image = Image.open(image_path).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transform(image).unsqueeze(0)


def load_mask(mask_path: str, size: int = 16):
    mask_img = Image.open(mask_path).convert('L')
    mask_img = mask_img.resize((size, size), Image.NEAREST)
    mask_array = np.array(mask_img) / 255.0
    mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(0)
    return mask_tensor


def inpaint_image(image_path: str,
                  mask_path: str,
                  checkpoint_path: str = 'checkpoints/inpainting_checkpoint',
                  num_denoising_steps: int = 50,
                  save_result: str = None):
    
    device = setup_cuda_device(preferred_gpu=0)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = InpaintingUNET().to(device)
    
    try:
        model.load_state_dict(checkpoint['weights'])
    except RuntimeError as e:
        model.load_state_dict(checkpoint['weights'], strict=False)
    
    if 'ema' in checkpoint:
        ema = ModelEmaV3(model, decay=0.9999)
        try:
            ema.load_state_dict(checkpoint['ema'])
            model = ema.module.eval()
        except RuntimeError as e:
            ema.load_state_dict(checkpoint['ema'], strict=False)
            model = ema.module.eval()
    else:
        model = model.eval()
    
    scheduler = DDPM_Scheduler(num_time_steps=1000).to(device)
    
    image = load_and_preprocess_image(image_path, size=16).to(device)
    mask = load_mask(mask_path, size=16).to(device)
    
    masked_image = image * mask
    
    with torch.no_grad():
        noise = torch.randn_like(image)
        x = image * mask + noise * (1 - mask)
        
        t_start = 200
        alpha_start = scheduler.alpha[torch.tensor([t_start], device=device)].view(1, 1, 1, 1)
        x = torch.sqrt(alpha_start) * x + torch.sqrt(1 - alpha_start) * torch.randn_like(x)
        
        step_size = max(1, 1000 // num_denoising_steps)
        timesteps = list(range(999, 0, -step_size))
        
        for i, t_val in enumerate(timesteps):
            t = torch.tensor([t_val], device=device)
            
            predicted_noise = model(x * mask, t, mask)
            
            alpha_t = scheduler.alpha[t].view(1, 1, 1, 1)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            pred_x0 = image * mask + pred_x0 * (1 - mask)
            
            if i < len(timesteps) - 1:
                t_next = torch.tensor([timesteps[i + 1]], device=device)
                alpha_t_next = scheduler.alpha[t_next].view(1, 1, 1, 1)
                
                x = torch.sqrt(alpha_t_next) * pred_x0 + torch.sqrt(1 - alpha_t_next) * predicted_noise
                x = image * mask + x * (1 - mask)
            else:
                x = pred_x0
        
        result = x
    
    display_results(image, masked_image, mask, result, save_result)
    
    return result


def display_results(original, masked, mask, result, save_path=None):
    def tensor_to_image(tensor):
        img = tensor.squeeze(0).detach().cpu()
        img = rearrange(img, 'c h w -> h w c')
        img = (img + 1) / 2
        img = np.clip(img.numpy() if torch.is_tensor(img) else img, 0, 1)
        return img
    
    original_display = tensor_to_image(original)
    masked_display = tensor_to_image(masked)
    mask_display = mask.squeeze().detach().cpu().numpy()
    result_display = tensor_to_image(result)
    
    if save_path:
        result_array = np.array(result_display * 255, dtype=np.uint8)
        result_image = Image.fromarray(result_array)
        result_image.save(save_path, format='PNG', quality=100, optimize=False)
        
        comparison_path = save_path.replace('.png', '_comparison.png')
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(original_display)
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(masked_display)
        axes[0, 1].set_title('Masked (Input)', fontsize=14)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(mask_display, cmap='gray', interpolation='nearest')
        axes[1, 0].set_title('Mask (white=keep, black=fill)', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(result_display)
        axes[1, 1].set_title('Inpainted Result', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', format='png')
        plt.close()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_display)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(masked_display)
    axes[1].set_title('Masked', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(mask_display, cmap='gray', interpolation='nearest')
    axes[2].set_title('Mask', fontsize=14)
    axes[2].axis('off')
    
    axes[3].imshow(result_display)
    axes[3].set_title('Inpainted', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
