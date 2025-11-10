"""
Image Inpainting Inference
===========================
Fill in missing/masked regions of images using trained diffusion model.
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


def load_and_preprocess_image(image_path: str, size: int = 128):
    """Load and preprocess image."""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(image).unsqueeze(0)


def load_mask(mask_path: str, size: int = 128):
    """
    Load binary mask.
    White (255) = keep region
    Black (0) = inpaint region
    """
    mask_img = Image.open(mask_path).convert('L')  # Grayscale
    mask_img = mask_img.resize((size, size), Image.NEAREST)
    mask_array = np.array(mask_img) / 255.0  # Normalize to 0-1
    mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0).unsqueeze(0)
    return mask_tensor


def inpaint_image(image_path: str,
                  mask_path: str,
                  checkpoint_path: str = 'checkpoints/inpainting_checkpoint',
                  num_denoising_steps: int = 50,
                  save_result: str = None):
    """
    Fill in masked regions of an image.
    
    Args:
        image_path: Path to image with regions to fill
        mask_path: Path to mask (white=keep, black=fill)
        checkpoint_path: Trained model
        num_denoising_steps: Quality (higher = better)
        save_result: Output path
    """
    
    print("🖌️ Starting Image Inpainting...")
    
    # Load model
    device = setup_cuda_device(preferred_gpu=0)
    
    print("📥 Loading trained model...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except:
        # Try with weights_only=True for older checkpoints
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = InpaintingUNET().to(device)
    
    # Try loading state dict with strict=False to handle mismatches
    try:
        model.load_state_dict(checkpoint['weights'])
    except RuntimeError as e:
        print(f"⚠️  Warning: Some weights didn't load: {e}")
        print("   Attempting partial load...")
        model.load_state_dict(checkpoint['weights'], strict=False)
    
    # Use EMA weights if available
    if 'ema' in checkpoint:
        ema = ModelEmaV3(model, decay=0.9999)
        ema.load_state_dict(checkpoint['ema'])
        model = ema.module.eval()
    else:
        model = model.eval()
    
    scheduler = DDPM_Scheduler(num_time_steps=1000).to(device)
    
    # Load image and mask
    print("🖼️ Loading image and mask...")
    image = load_and_preprocess_image(image_path, size=128).to(device)
    mask = load_mask(mask_path, size=128).to(device)
    
    print(f"   Image shape: {image.shape}")
    print(f"   Mask shape: {mask.shape}")
    
    # Apply mask to image (set masked regions to 0)
    masked_image = image * mask
    
    # Inpainting using diffusion
    print("🔄 Filling in masked regions...")
    
    with torch.no_grad():
        # Start from pure noise in masked regions, keep known regions
        noise = torch.randn_like(image)
        
        # Mix: keep known regions, random noise in masked regions
        x = image * mask + noise * (1 - mask)
        
        # Add some noise to known regions too for blending
        t_start = 500
        alpha_start = scheduler.alpha[torch.tensor([t_start], device=device)].view(1, 1, 1, 1)
        x = torch.sqrt(alpha_start) * x + torch.sqrt(1 - alpha_start) * torch.randn_like(x)
        
        # Denoise step by step
        step_size = max(1, 1000 // num_denoising_steps)
        timesteps = list(range(999, 0, -step_size))  # 999 not 1000 (0-indexed)
        
        for i, t_val in enumerate(timesteps):
            t = torch.tensor([t_val], device=device)
            
            # Predict noise CONDITIONED on mask
            # Debug shapes
            if i == 0:
                print(f"   Input to model - x*mask shape: {(x * mask).shape}, mask shape: {mask.shape}")
            
            predicted_noise = model(x * mask, t, mask)
            
            # Get alpha
            alpha_t = scheduler.alpha[t].view(1, 1, 1, 1)
            
            # Predict clean image
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Keep known regions, update masked regions
            pred_x0 = image * mask + pred_x0 * (1 - mask)
            
            # DDIM step
            if i < len(timesteps) - 1:
                t_next = torch.tensor([timesteps[i + 1]], device=device)
                alpha_t_next = scheduler.alpha[t_next].view(1, 1, 1, 1)
                
                x = torch.sqrt(alpha_t_next) * pred_x0 + torch.sqrt(1 - alpha_t_next) * predicted_noise
            else:
                x = pred_x0
        
        result = x
    
    # Upscale to 512x512
    print("⬆️ Upscaling to 512x512...")
    result = torch.nn.functional.interpolate(
        result,
        size=(512, 512),
        mode='bicubic',
        align_corners=False,
        antialias=True
    )
    
    # Load high-res for display
    image_hr = load_and_preprocess_image(image_path, size=512).to(device)
    mask_hr = torch.nn.functional.interpolate(mask, size=(512, 512), mode='nearest')
    masked_image_hr = image_hr * mask_hr
    
    # Display results
    print("✅ Inpainting complete!")
    display_results(image_hr, masked_image_hr, mask_hr, result, save_result)
    
    return result


def display_results(original, masked, mask, result, save_path=None):
    """Display and save results."""
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
        # Save high-quality result
        result_array = np.array(result_display * 255, dtype=np.uint8)
        result_image = Image.fromarray(result_array)
        result_image.save(save_path, format='PNG', quality=100, optimize=False)
        print(f"💾 Result saved: {save_path}")
        
        # Save comparison
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
        print(f"📊 Comparison saved: {comparison_path}")
        plt.close()
    
    # Display
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
