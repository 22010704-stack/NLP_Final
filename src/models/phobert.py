"""
src/models/phobert.py
PhoBERT with a custom Attention Pooling head.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.w = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        # x: [batch size, seq len, hidden dim]
        # mask: [batch size, seq len]
        
        # Compute projection
        u = torch.tanh(self.w(x)) # [batch size, seq len, hidden dim]
        
        # Compute scores
        # v: [hidden dim], u: [batch size, seq len, hidden dim]
        score = torch.matmul(u, self.v) # [batch size, seq len]
        
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
            
        weights = torch.softmax(score, dim=1) # [batch size, seq len]
        
        # Weighted sum
        out = torch.sum(x * weights.unsqueeze(-1), dim=1) # [batch size, hidden dim]
        return out


class PhoBERTClassifier(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", output_dim=3, dropout=0.1):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        hidden_dim = self.phobert.config.hidden_size
        
        self.attention_pool = AttentivePooling(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_ids, attention_mask):
        # input_ids: [batch size, seq len]
        
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: [batch size, seq len, hidden dim]
        last_hidden_state = outputs.last_hidden_state
        
        # Custom head: Attentive Pooling instead of [CLS] token pooling
        pooled = self.attention_pool(last_hidden_state, mask=attention_mask)
        # pooled: [batch size, hidden dim]
        
        return self.fc(self.dropout(pooled))
