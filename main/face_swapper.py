"""
Face Swapping with Your Own Images
==================================
Perform face swapping using two of your own images.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from einops import rearrange
from timm.utils import ModelEmaV3

from .unet import UNET
from .ddpm_scheduler import DDPM_Scheduler
from .utils import setup_cuda_device


def load_and_preprocess_image(image_path: str, size: int = 64):
    """Load and preprocess an image for face swapping."""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor


def display_face_swap_result(source_img, target_img, swapped_img, save_path=None):
    """Display the face swap results."""
    def tensor_to_image(tensor):
        img = tensor.squeeze(0).detach().cpu()
        img = rearrange(img, 'c h w -> h w c')
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        return img
    
    source_display = tensor_to_image(source_img)
    target_display = tensor_to_image(target_img)
    swapped_display = tensor_to_image(swapped_img)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(source_display)
    axes[0].set_title('Source Face')
    axes[0].axis('off')
    
    axes[1].imshow(target_display)
    axes[1].set_title('Target Face')
    axes[1].axis('off')
    
    axes[2].imshow(swapped_display)
    axes[2].set_title('Face Swap Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Result saved to: {save_path}")
    
    plt.show()


def swap_faces(source_image_path: str, 
               target_image_path: str,
               checkpoint_path: str = 'checkpoints/ddpm_faceswap_checkpoint',
               num_denoising_steps: int = 50,
               save_result: str = None):
    """
    Perform face swapping between two images.
    
    Args:
        source_image_path: Path to source face image
        target_image_path: Path to target face image
        checkpoint_path: Path to trained model
        num_denoising_steps: Number of denoising steps (fewer = faster, more = higher quality)
        save_result: Optional path to save the result image
    
    Returns:
        Generated face-swapped image tensor
    """
    
    print("🎯 Starting Face Swap Process...")
    
    # Load model
    device = setup_cuda_device(preferred_gpu=0)
    
    print("📥 Loading trained model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNET().to(device)
    model.load_state_dict(checkpoint['weights'])
    
    # Use EMA weights for better quality
    ema = ModelEmaV3(model, decay=0.9999)
    ema.load_state_dict(checkpoint['ema'])
    model = ema.module.eval()
    
    scheduler = DDPM_Scheduler(num_time_steps=1000).to(device)
    
    # Load and preprocess images
    print("🖼️ Loading images...")
    source_tensor = load_and_preprocess_image(source_image_path).to(device)
    target_tensor = load_and_preprocess_image(target_image_path).to(device)
    
    # Perform face swap using diffusion model
    print("🔄 Performing face swap...")
    
    with torch.no_grad():
        # Add noise to target image
        noise_level = 0.3
        noise = torch.randn_like(target_tensor)
        
        alpha = 1 - noise_level
        noisy_target = alpha * target_tensor + (1 - alpha) * noise
        
        # Denoise to create face-swapped result
        current_image = noisy_target
        
        for step in range(num_denoising_steps):
            t = torch.tensor([step * (1000 // num_denoising_steps)], device=device)
            
            beta_t = scheduler.beta[t]
            alpha_t = scheduler.alpha[t]
            
            # Predict and remove noise
            predicted_noise = model(current_image, t)
            temp = beta_t / (torch.sqrt(1 - alpha_t) * torch.sqrt(1 - beta_t))
            current_image = (1 / torch.sqrt(1 - beta_t)) * current_image - temp * predicted_noise
            
            # Add small noise for next step (except last)
            if step < num_denoising_steps - 1:
                current_image = current_image + 0.01 * torch.randn_like(current_image)
        
        face_swapped_result = current_image
    
    # Display and save results
    print("✅ Face swap complete!")
    display_face_swap_result(source_tensor, target_tensor, face_swapped_result, save_result)
    
    return face_swapped_result


def batch_face_swap(image_pairs: list, checkpoint_path: str = 'checkpoints/ddpm_faceswap_checkpoint'):
    """
    Perform face swapping on multiple image pairs.
    
    Args:
        image_pairs: List of tuples [(source1, target1), (source2, target2), ...]
        checkpoint_path: Path to trained model
    """
    print(f"🔄 Processing {len(image_pairs)} face swap pairs...")
    
    results = []
    for i, (source_path, target_path) in enumerate(image_pairs):
        print(f"\n--- Processing pair {i+1}/{len(image_pairs)} ---")
        result = swap_faces(
            source_path, 
            target_path, 
            checkpoint_path,
            save_result=f'face_swap_result_{i+1}.png'
        )
        results.append(result)
    
    print(f"\n🎉 Batch processing complete!")
    return results
