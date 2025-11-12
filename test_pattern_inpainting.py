#!/usr/bin/env python3
"""
Pattern Inpainting Inference
============================
Test the trained model on specific pixel patterns.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'main'))

import torch
import matplotlib.pyplot as plt
import numpy as np
from main.pixel_pattern_dataset import PixelPatternDataset
from main.inpainting_unet import InpaintingUNET
from main.ddpm_scheduler import DDPM_Scheduler
import torchvision.transforms as transforms

def create_custom_pattern(pattern_type="purple_square", size=128):
    """Create a specific pattern for testing."""
    image = torch.zeros(3, size, size)
    
    if pattern_type == "purple_square":
        # Purple square
        purple = torch.tensor([0.5, 0.2, 0.8])  # Purple color
        image = purple.view(3, 1, 1).expand(3, size, size)
        
    elif pattern_type == "rainbow_horizontal":
        # Horizontal rainbow
        for i in range(size):
            hue = i / size
            rgb = hsv_to_rgb(hue, 1.0, 1.0)
            image[:, :, i] = rgb.view(3, 1)
            
    elif pattern_type == "checkerboard":
        # Black and white checkerboard
        square_size = 16
        for y in range(size):
            for x in range(size):
                if ((y // square_size) + (x // square_size)) % 2 == 0:
                    image[:, y, x] = 1.0  # White
                else:
                    image[:, y, x] = 0.0  # Black
                    
    elif pattern_type == "red_blue_stripes":
        # Red and blue vertical stripes
        stripe_width = 20
        for x in range(size):
            if (x // stripe_width) % 2 == 0:
                image[0, :, x] = 1.0  # Red
                image[1, :, x] = 0.0
                image[2, :, x] = 0.0
            else:
                image[0, :, x] = 0.0  # Blue
                image[1, :, x] = 0.0
                image[2, :, x] = 1.0
    
    # Normalize to [-1, 1] range
    return image * 2.0 - 1.0

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB."""
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    
    if h < 1/6:
        rgb = torch.tensor([c, x, 0])
    elif h < 2/6:
        rgb = torch.tensor([x, c, 0])
    elif h < 3/6:
        rgb = torch.tensor([0, c, x])
    elif h < 4/6:
        rgb = torch.tensor([0, x, c])
    elif h < 5/6:
        rgb = torch.tensor([x, 0, c])
    else:
        rgb = torch.tensor([c, 0, x])
    
    return rgb + m

def create_test_mask(size=128, mask_type="center_square"):
    """Create a test mask."""
    mask = torch.ones(1, size, size)
    
    if mask_type == "center_square":
        # Square hole in center
        hole_size = size // 3
        start = (size - hole_size) // 2
        end = start + hole_size
        mask[:, start:end, start:end] = 0
        
    elif mask_type == "random_holes":
        # Random circular holes
        num_holes = 3
        for _ in range(num_holes):
            center_y = torch.randint(size//4, 3*size//4, (1,))
            center_x = torch.randint(size//4, 3*size//4, (1,))
            radius = torch.randint(size//8, size//4, (1,))
            
            y_coords, x_coords = torch.meshgrid(
                torch.arange(size), torch.arange(size), indexing='ij'
            )
            distances = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            mask[0, distances < radius] = 0
            
    elif mask_type == "left_half":
        # Remove left half
        mask[:, :, :size//2] = 0
    
    return mask

def inpaint_pattern(model, scheduler, image, mask, device, num_inference_steps=50):
    """Run inpainting inference."""
    model.eval()
    
    with torch.no_grad():
        batch_size = 1
        
        # Apply mask to image
        masked_image = image * mask
        
        # Start with random noise
        sample = torch.randn_like(image)
        
        # Reverse diffusion process
        for t in reversed(range(0, num_inference_steps)):
            t_tensor = torch.tensor([t], device=device)
            
            # Predict noise
            predicted_noise = model(
                sample,  # Current sample 
                t_tensor, 
                mask     # Just the mask
            )
            
            # Denoise step
            alpha = scheduler.alpha[t]
            alpha_prev = scheduler.alpha[t-1] if t > 0 else torch.tensor(1.0)
            
            beta = 1 - alpha
            
            # DDPM sampling step
            sample = (1 / torch.sqrt(alpha)) * (
                sample - (beta / torch.sqrt(1 - alpha)) * predicted_noise
            )
            
            if t > 0:
                noise = torch.randn_like(sample)
                sample = sample + torch.sqrt(1 - alpha_prev) * noise
            
            # Apply mask constraint (keep known pixels)
            sample = sample * (1 - mask) + image * mask
    
    return sample

def test_pattern_inpainting():
    """Test pattern inpainting with custom examples."""
    print("🧪 Testing Pattern Inpainting...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = "checkpoints/pattern_inpainting_checkpoint"
    if not os.path.exists(checkpoint_path):
        print("❌ No checkpoint found! Please train the model first.")
        print("Run: python train_patterns.py")
        return
    
    print(f"📥 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize model
    model = InpaintingUNET().to(device)
    scheduler = DDPM_Scheduler(num_time_steps=1000).to(device)
    
    model.load_state_dict(checkpoint['weights'])
    model.eval()
    
    # Test patterns
    test_cases = [
        ("purple_square", "center_square"),
        ("rainbow_horizontal", "left_half"), 
        ("checkerboard", "random_holes"),
        ("red_blue_stripes", "center_square")
    ]
    
    fig, axes = plt.subplots(len(test_cases), 4, figsize=(16, 4*len(test_cases)))
    
    for i, (pattern_type, mask_type) in enumerate(test_cases):
        print(f"\n🎨 Testing {pattern_type} with {mask_type} mask...")
        
        # Create test pattern and mask
        image = create_custom_pattern(pattern_type, size=128).unsqueeze(0).to(device)
        mask = create_test_mask(size=128, mask_type=mask_type).to(device)
        
        # Run inpainting
        result = inpaint_pattern(model, scheduler, image, mask, device)
        
        # Convert for visualization
        image_vis = torch.clamp((image.squeeze(0) + 1) / 2, 0, 1).cpu().permute(1, 2, 0)
        mask_vis = mask.squeeze(0).squeeze(0).cpu()
        masked_vis = torch.clamp(((image * mask).squeeze(0).cpu() + 1) / 2, 0, 1).permute(1, 2, 0)
        result_vis = torch.clamp((result.squeeze(0) + 1) / 2, 0, 1).cpu().permute(1, 2, 0)
        
        # Plot
        axes[i, 0].imshow(image_vis)
        axes[i, 0].set_title(f"Original\n{pattern_type}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_vis, cmap='gray')
        axes[i, 1].set_title(f"Mask\n{mask_type}")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(masked_vis)
        axes[i, 2].set_title("Masked Input")
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(result_vis)
        axes[i, 3].set_title("Inpainted Result")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('pattern_inpainting_results.png', dpi=150, bbox_inches='tight')
    print("\n💾 Saved results to: pattern_inpainting_results.png")
    plt.show()
    
    print("✅ Pattern inpainting test completed!")

if __name__ == "__main__":
    test_pattern_inpainting()