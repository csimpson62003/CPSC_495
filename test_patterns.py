#!/usr/bin/env python3
"""
Test Pixel Pattern Dataset
==========================
Test script to verify the new pixel pattern dataset is working correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'main'))

import torch
import matplotlib.pyplot as plt
from main.pixel_pattern_dataset import PixelPatternDataset

def test_patterns():
    """Test and visualize pixel patterns."""
    print("🎨 Testing Pixel Pattern Dataset...")
    
    # Create dataset
    dataset = PixelPatternDataset(
        size=100,
        image_size=128,
        pattern_types=[
            'solid_color',
            'horizontal_rainbow', 
            'vertical_rainbow',
            'radial_rainbow',
            'checkerboard',
            'horizontal_stripes',
            'vertical_stripes',
            'diagonal_stripes',
            'concentric_circles',
            'gradient'
        ]
    )
    
    print(f"✅ Dataset created with {len(dataset)} patterns")
    
    # Test a few samples
    print("\n🔍 Testing samples...")
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}: {sample['pattern_type']}")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Image range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
    
    # Visualize patterns
    print("\n🖼️ Creating visualization...")
    visualize_patterns_and_masks(dataset, num_samples=9)
    
    print("✅ Pattern dataset test completed!")

def visualize_patterns_and_masks(dataset, num_samples=9):
    """Visualize patterns with their masks and masked versions."""
    fig, axes = plt.subplots(3, num_samples, figsize=(18, 6))
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # Convert from [-1,1] to [0,1] for visualization  
        image = torch.clamp((sample['image'] + 1) / 2, 0, 1)
        masked_image = torch.clamp((sample['masked_image'] + 1) / 2, 0, 1)
        mask = sample['mask'].squeeze(0)  # Remove channel dimension
        
        # Convert to numpy for matplotlib
        image_np = image.permute(1, 2, 0).numpy()
        masked_image_np = masked_image.permute(1, 2, 0).numpy() 
        mask_np = mask.numpy()
        
        # Original pattern
        axes[0, i].imshow(image_np)
        axes[0, i].set_title(f"{sample['pattern_type']}", fontsize=8)
        axes[0, i].axis('off')
        
        # Mask (white=keep, black=inpaint)
        axes[1, i].imshow(mask_np, cmap='gray')
        axes[1, i].set_title("Mask", fontsize=8)
        axes[1, i].axis('off')
        
        # Masked pattern (what the model sees)
        axes[2, i].imshow(masked_image_np)
        axes[2, i].set_title("Masked Input", fontsize=8)
        axes[2, i].axis('off')
    
    # Row labels
    fig.text(0.02, 0.83, 'Original\nPattern', fontsize=10, ha='left', va='center')
    fig.text(0.02, 0.50, 'Mask\n(White=Keep)', fontsize=10, ha='left', va='center')
    fig.text(0.02, 0.17, 'Masked\nInput', fontsize=10, ha='left', va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    
    # Save the visualization
    plt.savefig('pattern_samples.png', dpi=150, bbox_inches='tight')
    print("💾 Saved visualization to: pattern_samples.png")
    plt.show()

if __name__ == "__main__":
    test_patterns()