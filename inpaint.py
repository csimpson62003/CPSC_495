"""
Image Inpainting - Fill Missing Regions
========================================
Use YOUR trained diffusion model to fill in masked regions of images.
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error messages

from main.inpainting_inference import inpaint_image
from PIL import Image, ImageDraw
import numpy as np


def create_example_mask(image_path, mask_path, mask_type='center'):
    """
    Create an example mask for testing.
    
    Args:
        image_path: Path to image
        mask_path: Where to save mask
        mask_type: 'center', 'random', or 'strokes'
    """
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # Create white mask (white = keep, black = fill)
    mask = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(mask)
    
    if mask_type == 'center':
        # Rectangle in center
        w = width // 3
        h = height // 3
        x = (width - w) // 2
        y = (height - h) // 2
        draw.rectangle([x, y, x+w, y+h], fill=0)
        
    elif mask_type == 'random':
        # Random circle
        cx = width // 2
        cy = height // 2
        r = min(width, height) // 4
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=0)
        
    elif mask_type == 'strokes':
        # Random brush strokes
        for _ in range(5):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            x2 = np.random.randint(0, width)
            y2 = np.random.randint(0, height)
            draw.line([x1, y1, x2, y2], fill=0, width=20)
    
    mask.save(mask_path)
    print(f"✅ Created mask: {mask_path}")


def main():
    print("🖌️ Image Inpainting with Diffusion")
    print("=" * 50)
    
    # Create folder for images
    if not os.path.exists("my_images"):
        os.makedirs("my_images")
        print("📁 Created 'my_images' folder")
        print("\n⚠️  Please add:")
        print("   - my_images/image.jpg - Image to inpaint")
        print("   - my_images/mask.png - Mask (white=keep, black=fill)")
        print("\n💡 Or run with auto_create_mask=True")
        return
    
    # Your image paths
    image_path = "my_images/image.jpg"
    mask_path = "my_images/mask.png"
    output_file = "inpainted_result.png"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("💡 Add your image to 'my_images/image.jpg'")
        return
    
    # Create mask if it doesn't exist
    if not os.path.exists(mask_path):
        print("⚠️  Mask not found. Creating example mask...")
        create_example_mask(image_path, mask_path, mask_type='center')
    
    print(f"\n📸 Image: {image_path}")
    print(f"🎭 Mask: {mask_path}")
    print(f"💾 Output: {output_file}")
    
    print("\n🚀 Starting inpainting...")
    
    try:
        result = inpaint_image(
            image_path=image_path,
            mask_path=mask_path,
            checkpoint_path='checkpoints/inpainting_checkpoint',
            num_denoising_steps=50,
            save_result=output_file
        )
        
        print(f"\n✅ SUCCESS!")
        print(f"📁 Inpainted image: {output_file}")
        print(f"📁 Comparison: {output_file.replace('.png', '_comparison.png')}")
        
        print("\n💡 Tips:")
        print("   - Increase num_denoising_steps (50 → 100) for better quality")
        print("   - White regions in mask = keep original")
        print("   - Black regions in mask = fill in (inpaint)")
        
    except FileNotFoundError as e:
        print(f"❌ Model not found: {e}")
        print("💡 You need to train the model first!")
        print("   Run: python train_inpainting_colab.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
