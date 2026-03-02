"""
src/models/transformer_encoder.py
Custom Transformer Encoder (2-4 layers) for text classification.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, 
                 output_dim, dropout, pad_idx):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch size, src len]
        
        # Padding mask for Transformer (True/1 for positions to be ignored)
        src_key_padding_mask = (src == self.embedding.padding_idx)
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        # src: [batch size, src len, d_model]
        
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        # output: [batch size, src len, d_model]
        
        # Mean pooling (ignoring padding)
        # Use simple mean for now
        mask = (~src_key_padding_mask).float().unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1)
        
        return self.fc(self.dropout(pooled))
