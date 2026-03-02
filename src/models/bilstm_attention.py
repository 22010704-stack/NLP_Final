"""
src/models/bilstm_attention.py
BiLSTM with Self-Attention for text classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: [batch size, sent len, hidden_dim]
        energy = self.projection(encoder_outputs) # [batch size, sent len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1) # [batch size, sent len]
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1) # [batch size, hidden_dim]
        return outputs, weights


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=True, 
                           dropout=dropout if n_layers > 1 else 0,
                           batch_first=True)
        
        self.attention = SelfAttention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch size, sent len]
        
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch size, sent len, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch size, sent len, hid dim * 2]
        
        # Self-attention pooling
        attn_output, weights = self.attention(outputs)
        # attn_output: [batch size, hid dim * 2]
        
        return self.fc(self.dropout(attn_output))
