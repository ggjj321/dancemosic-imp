import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import AISTDataset
from models.vqvae import VQVAE
from models.transformer import MaskedTransformer
import os

def train_vqvae(args):
    dataset = AISTDataset(args.data_dir, split='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = VQVAE(input_dim=72, hidden_dim=128, embedding_dim=64, num_embeddings=512).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(args.device) # [B, T, C]
            batch = batch.permute(0, 2, 1) # [B, C, T]
            
            optimizer.zero_grad()
            loss, recon, perplexity = model(batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"vqvae_epoch_{epoch+1}.pth"))

def train_transformer(args):
    # Load VQ-VAE
    vqvae = VQVAE(input_dim=72, hidden_dim=128, embedding_dim=64, num_embeddings=512).to(args.device)
    vqvae.load_state_dict(torch.load(args.vqvae_path))
    vqvae.eval()
    
    dataset = AISTDataset(args.data_dir, split='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = MaskedTransformer(num_embeddings=512, d_model=256, nhead=4, num_layers=4, dim_feedforward=512).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(args.device) # [B, T, C]
            batch = batch.permute(0, 2, 1) # [B, C, T]
            
            with torch.no_grad():
                indices = vqvae.encode(batch) # [B*T, 1]
                indices = indices.view(batch.size(0), -1) # [B, T]
            
            # Masking
            B, T = indices.shape
            mask = torch.rand(B, T).to(args.device) < args.mask_ratio
            masked_indices = indices.clone()
            masked_indices[mask] = 512 # Mask token ID
            
            optimizer.zero_grad()
            # Transformer expects [T, B]
            output = model(masked_indices.permute(1, 0)) # [T, B, NumEmbeddings]
            
            # Calculate loss only on masked tokens
            output = output.permute(1, 0, 2) # [B, T, NumEmbeddings]
            loss = criterion(output[mask], indices[mask])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
        
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"transformer_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['vqvae', 'transformer'])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vqvae_path", type=str, help="Path to pretrained VQ-VAE for transformer training")
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.mode == 'vqvae':
        train_vqvae(args)
    elif args.mode == 'transformer':
        train_transformer(args)
