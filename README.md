# Face-Swapping Diffusion Model

A PyTorch implementation of a face-swapping model using diffusion models.

## Quick Start for Google Colab

### 1. Clone the Repository
```bash
!git clone https://github.com/csimpson62003/CPSC_495.git
%cd CPSC_495
```

### 2. Install Dependencies
```bash
!pip install -r requirements.txt
```

### 3. Configure Git (IMPORTANT - Protects Your Progress!)
```bash
!python setup_colab.py
```

This will prompt you for:
- GitHub username
- GitHub email
- GitHub Personal Access Token (get one at https://github.com/settings/tokens)

**Why?** Training checkpoints will be automatically pushed to GitHub every 10 epochs. If Colab disconnects, your progress is saved!

### 4. Train the Model
```bash
!python train_colab.py
```

This will:
- Download the face-swap dataset from Kaggle (~7000 image pairs)
- Train the diffusion model
- Save checkpoints to `checkpoints/` every 10 epochs
- **Automatically push checkpoints to GitHub** (if configured in step 3)

Training typically takes several hours depending on your hardware.

## Usage After Training

### Face Swap with Your Own Images

1. Upload your images to the `my_photos/` folder
2. Edit `use_my_images.py` to point to your image paths
3. Run:
```bash
!python use_my_images.py
```

Or use `main.py`:
```bash
!python main.py
```

## Training Configuration

You can adjust training parameters in `train_colab.py`:
- `batch_size`: Reduce if you get out-of-memory errors (default: 8)
- `num_epochs`: Number of training epochs (default: 200)
- `lr`: Learning rate (default: 1e-4)
- `max_dataset_size`: Limit dataset size for testing (default: None = all data)
- `save_every_n_epochs`: Save checkpoint every N epochs (default: 10)
- `push_to_github`: Auto-push checkpoints to GitHub (default: True)

### Resume Training After Disconnect

If Colab disconnects, simply:
1. Clone the repo again (or reconnect)
2. Run `!python train_colab.py`

The training will automatically resume from the last saved checkpoint!

## Project Structure

```
CPSC_495/
├── main.py                    # Main script with different modes
├── train_colab.py            # Simple training script for Colab
├── use_my_images.py          # Simple face swap script
├── requirements.txt          # Python dependencies
├── my_photos/                # Put your images here
├── checkpoints/              # Trained models saved here
└── main/                     # Core model components
    ├── train.py              # Training logic
    ├── inference.py          # Image generation
    ├── face_swapper.py       # Face swapping functions
    ├── unet.py               # U-Net architecture
    ├── ddpm_scheduler.py     # Diffusion scheduler
    └── ...                   # Other components
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

## Notes

- The model works with 64x64 resolution images
- Images are automatically resized during preprocessing
- Training uses the face-swap dataset from Kaggle
- GPU recommended for training (CPU will be very slow)

## Modes

The model supports three modes (set in `main.py`):

1. **face_swap**: Swap faces between two images
2. **train**: Train the model from scratch
3. **generate**: Generate random face images

For Google Colab training, use `train_colab.py` instead.
