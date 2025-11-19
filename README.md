# Image Inpainting with Diffusion Models 🖌️

A PyTorch implementation of a **pattern inpainting model** using denoising diffusion probabilistic models (DDPM). The model fills in masked regions of **grayscale geometric patterns** by learning structural rules.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Pattern Dataset
```bash
python generate_patterns.py
```

This creates 5000 synthetic geometric patterns including:
- Checkerboards (various sizes)
- Stripes (horizontal, vertical, diagonal)
- Dots and grids
- Waves and zigzags
- Concentric shapes
- Random geometric shapes
- Mixed patterns

### 3. Train the Model
```bash
python train_colab.py
```

This will:
- Auto-generate patterns if none exist
- Train the mask-conditioned diffusion model on grayscale patterns
- Save checkpoints to `checkpoints/` every 100 epochs
- Train at 128x128 resolution

Training typically takes several hours on GPU.

### 4. Run Inpainting
```bash
python inpaint.py
```

Provide your own pattern image and mask (white = keep, black = fill in) to inpaint missing regions.

## How It Works

The model uses a **mask-conditioned U-Net** optimized for grayscale pattern inpainting:
- **Input channels**: 1 (grayscale)
- **Output channels**: 1 (grayscale)
- **Image**: Your grayscale pattern image
- **Mask**: Binary mask (white = keep, black = fill in)
- **Timestep**: Current diffusion timestep

The model learns geometric and structural patterns, enabling it to intelligently complete missing regions with contextually appropriate patterns. By using grayscale, the model focuses purely on pattern structure without color complexity.

## Pattern Dataset

The model trains on synthetically generated geometric patterns:

**Pattern Types:**
- Checkerboards (4x4 to 32x32)
- Stripes (horizontal, vertical, diagonal, varying widths)
- Dots/Polka dots (various sizes and spacing)
- Grids and lattices
- Sine waves (horizontal, vertical)
- Zigzags
- Concentric circles and squares
- Random geometric shapes
- Mixed pattern combinations

**Advantages:**
- ✅ Perfect for pattern learning
- ✅ No color matching complexity
- ✅ Unlimited variations
- ✅ Fast generation (no downloads)
- ✅ Clean, noise-free patterns

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
