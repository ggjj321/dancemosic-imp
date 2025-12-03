import argparse
import torch
import numpy as np
from models.vqvae import VQVAE
from models.transformer import MaskedTransformer
from utils.visualization import plot_motion
import math

def iterative_decoding(transformer, indices, mask_token_id, steps=10):
    """
    Iterative decoding (MaskGIT style).
    indices: [1, T] with masked tokens.
    """
    transformer.eval()
    unknown_number_in_the_beginning = torch.sum(indices == mask_token_id)
    
    for i in range(steps):
        # Predict
        with torch.no_grad():
            logits = transformer(indices.permute(1, 0)) # [T, 1, V]
            probs = torch.softmax(logits, dim=-1) # [T, 1, V]
            predicted_ids = torch.argmax(probs, dim=-1).permute(1, 0) # [1, T]
            confidence = torch.max(probs, dim=-1)[0].permute(1, 0) # [1, T]
        
        # Determine how many to keep
        ratio = 1.0 * (i + 1) / steps
        num_to_keep = int(unknown_number_in_the_beginning * ratio)
        
        # Identify currently masked tokens
        mask_indices = (indices == mask_token_id)
        
        if not mask_indices.any():
            break
            
        # Update indices with predictions where currently masked
        indices[mask_indices] = predicted_ids[mask_indices]
        
        # If not last step, re-mask low confidence ones among the newly filled
        if i < steps - 1:
            # We only want to keep the highest confidence ones among the ones that were masked
            # But standard MaskGIT re-masks based on global confidence or just keeps top k
            # Simplified: Keep top 'num_to_keep' confident predictions among the originally masked positions
            
            # For simplicity in this demo: just fill all and don't re-mask (naive iterative)
            # Or better: random re-masking or confidence based.
            # Let's implement a simple confidence-based selection
            
            # Get confidence of current predictions for originally masked positions
            current_confidences = confidence[mask_indices]
            
            # Sort and find threshold
            if len(current_confidences) > 0:
                # We want to keep 'num_to_keep' tokens, so we mask 'total_masked - num_to_keep'
                num_to_mask = unknown_number_in_the_beginning - num_to_keep
                if num_to_mask > 0:
                    threshold = torch.kthvalue(current_confidences, int(num_to_mask)).values
                    
                    # Re-mask those below threshold
                    # Note: this is a bit tricky with indexing, simplified approach:
                    # Just mask the lowest confidence ones
                    
                    # Create a temporary mask for positions that were originally masked
                    temp_mask = torch.zeros_like(indices, dtype=torch.bool)
                    temp_mask[mask_indices] = True
                    
                    # Filter confidence to only those positions
                    masked_conf = torch.where(temp_mask, confidence, torch.tensor(float('inf')).to(confidence.device))
                    
                    # Find bottom k
                    _, bottom_k_indices = torch.topk(masked_conf.view(-1), int(num_to_mask), largest=False)
                    
                    # Mask them
                    indices.view(-1)[bottom_k_indices] = mask_token_id

    return indices

def inbetween(args):
    device = args.device
    
    # Load Models
    vqvae = VQVAE(input_dim=args.input_dim, hidden_dim=128, embedding_dim=64, num_embeddings=512).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=device))
    vqvae.eval()
    
    transformer = MaskedTransformer(num_embeddings=512, d_model=256, nhead=4, num_layers=4, dim_feedforward=512).to(device)
    transformer.load_state_dict(torch.load(args.transformer_path, map_location=device))
    transformer.eval()
    
    # Load Data
    # Assuming args.start_motion and args.end_motion are .npy or .pkl files
    # For demo, generate random start/end if not provided
    if args.start_motion and args.end_motion:
        # Load logic here
        pass
    else:
        print("Using dummy start/end motion.")
        start_motion = torch.randn(1, args.input_dim, 20).to(device) # [B, C, T]
        end_motion = torch.randn(1, args.input_dim, 20).to(device)
    
    # Encode
    with torch.no_grad():
        start_indices = vqvae.encode(start_motion).view(1, -1) # [1, T_start]
        end_indices = vqvae.encode(end_motion).view(1, -1) # [1, T_end]
    
    # Create Middle Mask
    target_len = args.target_len
    mask_token_id = 512
    middle_indices = torch.full((1, target_len), mask_token_id, device=device, dtype=torch.long)
    
    # Concatenate
    full_indices = torch.cat([start_indices, middle_indices, end_indices], dim=1)
    
    # In-betweening
    # We only want to update the middle part, but the transformer sees the whole context
    # We can fix the start/end in the iterative decoding or just let it predict (but we want to constrain start/end)
    # The simple iterative_decoding function above updates everything that is masked.
    # Since start/end are NOT masked, they will be preserved.
    
    print("Generating in-between motion...")
    generated_indices = iterative_decoding(transformer, full_indices, mask_token_id)
    
    # Decode
    # generated_indices: [1, T_total]
    # We need to map indices back to quantized vectors
    # VQVAE decode expects quantized vectors usually, or we add a helper in VQVAE
    # Let's manually map for now
    with torch.no_grad():
        # [1, T] -> [1, T, C] (one-hot) -> [1, T, E] (embeddings)
        # Or just use embedding layer from quantizer
        indices_flat = generated_indices.view(-1)
        encodings = torch.zeros(indices_flat.shape[0], 512, device=device)
        encodings.scatter_(1, indices_flat.unsqueeze(1), 1)
        quantized = torch.matmul(encodings, vqvae.quantizer.embedding.weight) # [T_total, E]
        quantized = quantized.view(1, -1, 64).permute(0, 2, 1) # [1, E, T_total]
        
        generated_motion = vqvae.decoder(quantized)
    
    # Save
    generated_motion_np = generated_motion.squeeze().permute(1, 0).cpu().numpy() # [T, C]
    plot_motion(generated_motion_np, args.output_path + ".png")
    np.save(args.output_path + ".npy", generated_motion_np)
    print(f"Saved result to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_path", type=str, required=True)
    parser.add_argument("--transformer_path", type=str, required=True)
    parser.add_argument("--start_motion", type=str, default=None)
    parser.add_argument("--end_motion", type=str, default=None)
    parser.add_argument("--target_len", type=int, default=30)
    parser.add_argument("--output_path", type=str, default="result")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input_dim", type=int, default=72, help="Input dimension of motion data (e.g. 72 for SMPL, 51 for keypoints)")
    
    args = parser.parse_args()
    inbetween(args)
