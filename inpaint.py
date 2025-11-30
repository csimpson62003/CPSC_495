"""
Image Inpainting - Fill Missing Regions
========================================
Provide an image and mask, and the model will inpaint the masked regions.

Usage:
    python inpaint.py

Make sure to:
1. Place your image at: my_images/image.png
2. Place your mask at: my_images/mask.png (white=keep, black=fill)
3. Have a trained model at: checkpoints/inpainting_checkpoint
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from main.inpainting_inference import inpaint_image


def main():
    print("🖌️ Image Inpainting with Diffusion")
    print("=" * 50)
    
    # Paths
    image_path = "my_images/image.png"
    mask_path = "my_images/mask.png"
    checkpoint_path = "checkpoints/inpainting_checkpoint"
    output_file = "inpainted_result.png"
    
    # Validate inputs
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("💡 Place your image at 'my_images/image.png'")
        return
    
    if not os.path.exists(mask_path):
        print(f"❌ Mask not found: {mask_path}")
        print("💡 Place your mask at 'my_images/mask.png'")
        print("   (White pixels = keep, Black pixels = inpaint)")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model checkpoint not found: {checkpoint_path}")
        print("💡 Train the model first: python train_colab.py")
        return
    
    print(f"\n📸 Image: {image_path}")
    print(f"🎭 Mask: {mask_path}")
    print(f"🤖 Model: {checkpoint_path}")
    print(f"💾 Output: {output_file}")
    
    print("\n🚀 Starting inpainting...")
    
    try:
        result = inpaint_image(
            image_path=image_path,
            mask_path=mask_path,
            checkpoint_path=checkpoint_path,
            num_denoising_steps=100,  # Increased for better quality
            save_result=output_file
        )
        
        print(f"\n✅ SUCCESS!")
        print(f"📁 Result saved: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

