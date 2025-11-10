"""
Check what's in the checkpoint file
"""
import torch

checkpoint_path = 'checkpoints/inpainting_checkpoint'

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\nCheckpoint keys:", checkpoint.keys())

if 'weights' in checkpoint:
    print("\nModel state dict keys (first 10):")
    keys = list(checkpoint['weights'].keys())
    for i, key in enumerate(keys[:10]):
        shape = checkpoint['weights'][key].shape
        print(f"  {key}: {shape}")
    
    print(f"\n... and {len(keys) - 10} more keys")
    
    # Check if it's inpainting or old model
    if 'image_conv.weight' in keys:
        print("\n✅ This is an INPAINTING model checkpoint")
    elif 'conv_in.weight' in keys:
        print("\n❌ This is an OLD face-swap model checkpoint")
        print("   You need to train a new inpainting model!")
    else:
        print("\n⚠️  Unknown model type")

if 'epoch' in checkpoint:
    print(f"\nTrained for {checkpoint['epoch']} epochs")
if 'loss' in checkpoint:
    print(f"Final loss: {checkpoint['loss']:.5f}")
