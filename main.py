"""
Face-Swapping Diffusion Model - Main Script
============================================
Choose mode: "face_swap", "train", or "generate"
"""

from main import train, inference, swap_faces


def main():
    checkpoint_path = 'checkpoints/ddpm_faceswap_checkpoint'
    
    # Change mode to: "face_swap", "train", or "generate"
    mode = "face_swap"
    
    if mode == "face_swap":
        # Face swap with your own images
        source_image = "my_photos/trump.png"
        target_image = "my_photos/image.png"
        
        result = swap_faces(
            source_image_path=source_image,
            target_image_path=target_image,
            checkpoint_path=checkpoint_path,
            num_denoising_steps=50,
            save_result="my_face_swap_result.png"
        )
        print("✅ Face swap completed!")
    
    elif mode == "train":
        # Train the model (use train_colab.py instead for Google Colab)
        train(
            checkpoint_path=checkpoint_path,
            lr=1e-4,
            num_epochs=200,
            batch_size=8,
            max_dataset_size=None  # Set to number to limit dataset size
        )
        print("✅ Training complete!")
    
    elif mode == "generate":
        # Generate random face images
        inference(checkpoint_path)


if __name__ == '__main__':
    main()
