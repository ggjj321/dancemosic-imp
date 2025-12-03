import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(Quantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs: [B, C, T] -> [B, T, C]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # [B, T, C] -> [B, C, T]
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encoding_indices

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 4, 2, 1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 4, 2, 1)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 3, 1, 1)
        self.conv2 = nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, 2, 1)
        self.conv3 = nn.ConvTranspose1d(hidden_dim, output_dim, 4, 2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_embeddings):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, embedding_dim)
        self.quantizer = Quantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon, perplexity

    def encode(self, x):
        z = self.encoder(x)
        _, _, _, indices = self.quantizer(z)
        return indices
    
    def decode(self, indices):
        # indices: [B*T, 1] -> [B, T] (need to reshape outside or handle here)
        # For simplicity, assume we get quantized vectors or handle indices mapping
        pass # Need to implement indices -> quantized -> decoder if used separately
