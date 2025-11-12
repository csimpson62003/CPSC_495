# Image Inpainting Usage Guide

## Overview
This project fills in masked regions of images using a diffusion-based inpainting model.

## Quick Start

### 1. Prepare Your Inputs

Place two files in the `my_images/` folder:
- `image.png` - Your image to inpaint
- `mask.png` - Binary mask (white = keep, black = fill in)

### 2. Run Inpainting
```bash
python inpaint.py
```

The result will be saved as `inpainted_result.png`.

## Creating Masks

Your mask should be a grayscale or binary image where:
- **White pixels (255)** = Keep original
- **Black pixels (0)** = Fill in (inpaint)

You can create masks using any image editor (Photoshop, GIMP, etc.) or programmatically.

## Programmatic Usage

```python
from main.inpainting_inference import inpaint_image

result = inpaint_image(
    image_path="my_images/image.png",
    mask_path="my_images/mask.png",
    checkpoint_path="checkpoints/inpainting_checkpoint",
    num_denoising_steps=50,  # Increase for better quality
    output_size=512
)
```

## Training

To train on your own dataset:

```bash
python train_colab.py
```

This downloads CelebA dataset and trains the model. Checkpoints are saved every 100 epochs to `checkpoints/`.

## Tips

- Increase `num_denoising_steps` (50 → 100) for better quality but slower inference
- Smaller masked regions produce better results
- GPU highly recommended for training and inference
- Model works best on image types similar to training data