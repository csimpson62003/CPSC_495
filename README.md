# Image Inpainting with Diffusion Models 🖌️

A PyTorch implementation of an image inpainting model using denoising diffusion probabilistic models (DDPM). The model fills in masked regions of images by learning from diverse training data.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_colab.py
```

This will:
- Download the CelebA dataset from Kaggle (~200k face images)
- Train the mask-conditioned diffusion model
- Save checkpoints to `checkpoints/` every 100 epochs
- Train at 128x128 resolution (upscaled to 512x512 during inference)

Training typically takes several hours on GPU.

### 3. Run Inpainting
```bash
python inpaint.py
```

Provide your own image and mask (white = keep, black = fill in) to inpaint missing regions.

## How It Works

The model uses a **mask-conditioned U-Net** that takes:
- **Image**: Your input image
- **Mask**: Binary mask (white = keep, black = fill in)
- **Timestep**: Current diffusion timestep

The model iteratively denoises the masked region while preserving the known pixels. Results are upscaled from 128x128 to 512x512 for high-quality output.

## Usage

### Basic Inpainting
```python
from main.inpainting_inference import inpaint_image

# Provide image and mask paths
result = inpaint_image(
    image_path="your_image.jpg",
    mask_path="your_mask.png",  # White=keep, Black=fill
    checkpoint_path="checkpoints/inpainting_checkpoint",
    output_size=512
)
```

The mask should be a binary image where:
- **White (255)** = preserve original pixels
- **Black (0)** = inpaint this region

## Project Structure

```
CPSC_495/
├── train_colab.py              # Training script
├── inpaint.py                  # Inference script
├── requirements.txt            # Dependencies
├── checkpoints/                # Saved models
│   └── inpainting_checkpoint
├── data/                       # Training data directory
└── main/                       # Core components
    ├── inpainting_dataset.py   # Dataset with mask generation
    ├── inpainting_unet.py      # Mask-conditioned U-Net
    ├── train_inpainting.py     # Training logic
    ├── inpainting_inference.py # Inference logic
    ├── unet.py                 # Base U-Net architecture
    ├── ddpm_scheduler.py       # Diffusion scheduler
    ├── attention.py            # Attention mechanisms
    ├── res_block.py            # Residual blocks
    ├── unet_layer.py           # U-Net layers
    ├── sinusoidal_embeddings.py# Time embeddings
    └── utils.py                # Utilities
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- einops
- timm
- tqdm
- matplotlib
- numpy
- kagglehub
- pillow

## Technical Details

**Architecture**: U-Net with attention mechanisms and residual blocks, conditioned on masked image and binary mask.

**Training**: Adam optimizer with 1e-4 learning rate, MSE loss computed only on masked regions, EMA weights for stability.

**Inference**: 1000-step diffusion process that iteratively denoises masked regions while preserving known pixels.
