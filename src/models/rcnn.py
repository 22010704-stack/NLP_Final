"""
src/models/rcnn.py
Recurrent Convolutional Neural Network (RCNN) for text classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RCNN(nn.Module):
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
        
        # Concat original word embedding with left/right context from BiLSTM
        self.W2 = nn.Linear(2 * hidden_dim + embedding_dim, hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text: [batch size, sent len]
        
        embedded = self.embedding(text)
        # embedded: [batch size, sent len, emb dim]
        
        outputs, _ = self.rnn(embedded)
        # outputs: [batch size, sent len, hid dim * 2]
        
        # Concatenate: [left context, word embedding, right context]
        # BiLSTM output already contains left+right context at each time step
        combined = torch.cat((outputs, embedded), dim=2)
        # combined: [batch size, sent len, 2*hidden_dim + embedding_dim]
        
        # Apply projection followed by tanh (paper specification)
        latent = torch.tanh(self.W2(combined))
        # latent: [batch size, sent len, 2*hidden_dim]
        
        # Global max pooling over the sequence length
        latent = latent.permute(0, 2, 1)
        pooled = F.max_pool1d(latent, latent.shape[2]).squeeze(2)
        # pooled: [batch size, 2*hidden_dim]
        
        return self.fc(self.dropout(pooled))
