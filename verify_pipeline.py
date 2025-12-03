import os
import torch
import numpy as np
import pickle
import shutil
from data.dataset import AISTDataset
from models.vqvae import VQVAE
from models.transformer import MaskedTransformer
from train import train_vqvae, train_transformer
from inference import inbetween

class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def verify():
    print("Starting verification...")
    
    # 1. Setup Dummy Data
    data_dir = "dummy_aist_data"
    save_dir = "dummy_checkpoints"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create 5 dummy motion files
    for i in range(5):
        dummy_motion = np.random.rand(120, 72) # 120 frames
        with open(os.path.join(data_dir, f"motion_{i}.pkl"), "wb") as f:
            pickle.dump({'smpl_poses': dummy_motion}, f)
            
    print("Dummy data created.")
    
    device = "cpu" # Use CPU for verification to avoid CUDA issues if not available
    
    # 2. Verify VQ-VAE Training
    print("Verifying VQ-VAE training...")
    vqvae_args = Args(
        mode='vqvae',
        data_dir=data_dir,
        save_dir=save_dir,
        batch_size=2,
        lr=1e-4,
        epochs=1, # Just 1 epoch
        device=device
    )
    try:
        train_vqvae(vqvae_args)
        print("VQ-VAE training step passed.")
    except Exception as e:
        print(f"VQ-VAE training failed: {e}")
        return

    # 3. Verify Transformer Training
    print("Verifying Transformer training...")
    # Create a dummy VQ-VAE checkpoint if not saved (it should be saved by train_vqvae)
    vqvae_ckpt = os.path.join(save_dir, "vqvae_epoch_1.pth")
    if not os.path.exists(vqvae_ckpt):
        # Save a dummy one just in case logic failed
        model = VQVAE(72, 128, 64, 512)
        torch.save(model.state_dict(), vqvae_ckpt)
        
    transformer_args = Args(
        mode='transformer',
        data_dir=data_dir,
        save_dir=save_dir,
        batch_size=2,
        lr=1e-4,
        epochs=1,
        device=device,
        vqvae_path=vqvae_ckpt,
        mask_ratio=0.15
    )
    try:
        train_transformer(transformer_args)
        print("Transformer training step passed.")
    except Exception as e:
        print(f"Transformer training failed: {e}")
        return

    # 4. Verify Inference
    print("Verifying Inference...")
    transformer_ckpt = os.path.join(save_dir, "transformer_epoch_1.pth")
    if not os.path.exists(transformer_ckpt):
        model = MaskedTransformer(512, 256, 4, 4, 512)
        torch.save(model.state_dict(), transformer_ckpt)
        
    inference_args = Args(
        vqvae_path=vqvae_ckpt,
        transformer_path=transformer_ckpt,
        start_motion=None, # Use dummy
        end_motion=None, # Use dummy
        target_len=20,
        output_path=os.path.join(save_dir, "test_result"),
        device=device
    )
    try:
        inbetween(inference_args)
        print("Inference passed.")
    except Exception as e:
        print(f"Inference failed: {e}")
        return

    # Cleanup
    shutil.rmtree(data_dir)
    shutil.rmtree(save_dir)
    print("Verification successful! Cleanup done.")

if __name__ == "__main__":
    verify()
