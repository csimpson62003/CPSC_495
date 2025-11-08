"""
Simple Face Swap with Your Own Images
=====================================
Put your images in 'my_photos/' and update the paths below.
"""

from main.face_swapper import swap_faces
import os


def simple_face_swap():
    # Update these paths to your images
    source_face = "my_photos/person1.jpg"
    target_person = "my_photos/person2.jpg"
    output_file = "my_face_swap_result.png"
    
    print(f"📸 Taking face from: {source_face}")
    print(f"📸 Putting it on: {target_person}")
    print(f"💾 Saving result to: {output_file}")
    
    # Check if images exist
    if not os.path.exists(source_face):
        print(f"❌ Source image not found: {source_face}")
        return
        
    if not os.path.exists(target_person):
        print(f"❌ Target image not found: {target_person}")
        return
    
    print("\n🚀 Starting face swap...")
    
    try:
        result = swap_faces(
            source_image_path=source_face,
            target_image_path=target_person,
            checkpoint_path='checkpoints/ddpm_faceswap_checkpoint',
            num_denoising_steps=50,
            save_result=output_file
        )
        
        print(f"✅ SUCCESS! Check your result: {output_file}")
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("💡 You need to train the model first!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Create my_photos folder if it doesn't exist
    if not os.path.exists("my_photos"):
        os.makedirs("my_photos")
        print("📁 Created 'my_photos' folder")
    
    simple_face_swap()
