# DanceMosaic Reproduction

This project reproduces the "Walk Before You Dance" (DanceMosaic) model for motion in-betweening using the AIST++ dataset.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: You may need to install PyTorch separately depending on your CUDA version.*

2. **Prepare Data**
   - Place your AIST++ motion data (pickle files) in a directory (e.g., `data/aist_plusplus`).
   - The data loader expects `.pkl` files containing motion parameters (e.g., SMPL poses).

## Usage

### Training

1. **Train VQ-VAE**
   ```bash
   python train.py --mode vqvae --data_dir path/to/data --save_dir checkpoints
   ```

2. **Train Masked Transformer**
   ```bash
   python train.py --mode transformer --data_dir path/to/data --save_dir checkpoints --vqvae_path checkpoints/vqvae_epoch_50.pth
   ```

### Inference (In-betweening)

Generate motion between a start and end sequence:

```bash
python inference.py \
    --vqvae_path checkpoints/vqvae_epoch_50.pth \
    --transformer_path checkpoints/transformer_epoch_50.pth \
    --start_motion path/to/start.npy \
    --end_motion path/to/end.npy \
    --target_len 30 \
    --output_path result
```

## Verification

To verify the pipeline (requires dependencies installed):
```bash
python verify_pipeline.py
```
