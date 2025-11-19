"""
Preview Generated Patterns
===========================
Display a grid of sample patterns to verify the dataset.
"""

import matplotlib.pyplot as plt
from PIL import Image
import glob
import random

def preview_patterns(num_samples=16):
    """Show a grid of random pattern samples."""
    pattern_files = glob.glob('data/pattern_*.png')
    
    if len(pattern_files) == 0:
        print("❌ No patterns found in data/")
        print("Run: python generate_patterns.py")
        return
    
    # Sample random patterns
    samples = random.sample(pattern_files, min(num_samples, len(pattern_files)))
    
    # Create grid
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for idx, img_path in enumerate(samples):
        img = Image.open(img_path)
        axes[idx].imshow(img, cmap='gray')
        # Extract pattern type from filename
        pattern_name = img_path.split('_', 2)[-1].replace('.png', '').replace('_', ' ')
        axes[idx].set_title(pattern_name, fontsize=10)
        axes[idx].axis('off')
    
    # Hide extra subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('pattern_preview.png', dpi=150, bbox_inches='tight')
    print(f"✅ Preview saved: pattern_preview.png")
    print(f"📊 Showing {len(samples)} of {len(pattern_files)} patterns")
    plt.show()


if __name__ == "__main__":
    preview_patterns(16)
