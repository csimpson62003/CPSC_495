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


def load_and_preprocess_image(image_path: str, size: int = 512):
    """Load and preprocess an image for face swapping."""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
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
        img = np.clip(img.numpy() if torch.is_tensor(img) else img, 0, 1)
        return img
    
    source_display = tensor_to_image(source_img)
    target_display = tensor_to_image(target_img)
    swapped_display = tensor_to_image(swapped_img)
    
    # Save high-quality individual result image
    if save_path:
        # Save the swapped result as a high-quality PNG
        swapped_array = np.array(swapped_display * 255, dtype=np.uint8)
        result_image = Image.fromarray(swapped_array)
        result_image.save(save_path, format='PNG', quality=100, optimize=False)
        print(f"High-quality result saved to: {save_path}")
        
        # Also save a comparison image
        comparison_path = save_path.replace('.png', '_comparison.png')
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(source_display)
        axes[0].set_title('Source Face', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(target_display)
        axes[1].set_title('Target Face', fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(swapped_display)
        axes[2].set_title('Face Swap Result', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight', format='png')
        print(f"Comparison image saved to: {comparison_path}")
        plt.close()
    
    # Display the result
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(source_display)
    axes[0].set_title('Source Face', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(target_display)
    axes[1].set_title('Target Face', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(swapped_display)
    axes[2].set_title('Face Swap Result', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
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
    # Load images at high resolution for final output
    source_tensor_hr = load_and_preprocess_image(source_image_path, size=512).to(device)
    target_tensor_hr = load_and_preprocess_image(target_image_path, size=512).to(device)
    
    # Also load at model resolution (64x64) for processing
    source_tensor = load_and_preprocess_image(source_image_path, size=64).to(device)
    target_tensor = load_and_preprocess_image(target_image_path, size=64).to(device)
    
    # Perform face swap using diffusion model
    print("🔄 Performing face swap...")
    
    with torch.no_grad():
        # Simple approach: Start with target image and denoise it
        # This works because model was trained on altered/swapped faces
        
        # Create a blended starting point 
        # Mix more of the source to preserve facial features
        blend_weight = 0.7  # More weight on source face
        blended = blend_weight * source_tensor + (1 - blend_weight) * target_tensor
        
        # Add moderate noise to the blended image
        noise = torch.randn_like(blended)
        
        # Use less noise for sharper results
        t_start = torch.tensor([200], device=device)  # Reduced from 300
        alpha_t_start = scheduler.alpha[t_start].view(1, 1, 1, 1)
        
        # Create noisy version
        x = torch.sqrt(alpha_t_start) * blended + torch.sqrt(1 - alpha_t_start) * noise
        
        # Denoise step by step (DDIM sampling for better quality)
        step_size = 1000 // num_denoising_steps
        timesteps = list(range(200, 0, -step_size))
        
        for i, t_val in enumerate(timesteps):
            t = torch.tensor([t_val], device=device)
            
            # Predict noise
            predicted_noise = model(x, t)
            
            # Get alpha values
            alpha_t = scheduler.alpha[t].view(1, 1, 1, 1)
            
            # Predict clean image (x0)
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Guide with source features
            if i < len(timesteps) // 3:
                # Early steps only: lighter guidance to preserve sharpness
                pred_x0 = 0.8 * pred_x0 + 0.2 * source_tensor
            
            if i < len(timesteps) - 1:
                # Get next timestep's alpha
                t_next = torch.tensor([timesteps[i + 1]], device=device)
                alpha_t_next = scheduler.alpha[t_next].view(1, 1, 1, 1)
                
                # DDIM step (deterministic)
                sigma = 0.0
                noise_component = predicted_noise * torch.sqrt(torch.tensor(1 - sigma**2, device=device))
                x = torch.sqrt(alpha_t_next) * pred_x0 + torch.sqrt(1 - alpha_t_next) * noise_component
            else:
                x = pred_x0
        
        face_swapped_result = x
        
        # Lighter final blending to reduce blur
        face_swapped_result = 0.9 * face_swapped_result + 0.1 * source_tensor
        face_swapped_result = torch.clamp(face_swapped_result, -1, 1)
        
        # Upscale result to 512x512 using high-quality interpolation
        print("⬆️ Upscaling to 512x512...")
        face_swapped_result = torch.nn.functional.interpolate(
            face_swapped_result,
            size=(512, 512),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )
        
        # Sharpen the result slightly
        # Apply unsharp masking to enhance edges
        blurred = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(face_swapped_result, (1, 1, 1, 1), mode='replicate'),
            kernel_size=3,
            stride=1
        )
        sharpened = face_swapped_result + 0.3 * (face_swapped_result - blurred)
        face_swapped_result = torch.clamp(sharpened, -1, 1)
    
    # Display and save results
    print("✅ Face swap complete!")
    display_face_swap_result(source_tensor_hr, target_tensor_hr, face_swapped_result, save_result)
    
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
