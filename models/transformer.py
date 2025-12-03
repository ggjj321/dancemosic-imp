import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MaskedTransformer(nn.Module):
    def __init__(self, num_embeddings, d_model, nhead, num_layers, dim_feedforward, max_len=500):
        super(MaskedTransformer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings + 1, d_model) # +1 for MASK token
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward),
            num_layers
        )
        self.decoder = nn.Linear(d_model, num_embeddings)
        self.mask_token_id = num_embeddings

    def forward(self, src):
        # src: [T, B] (indices)
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
