"""
src/dataset.py
PyTorch Dataset classes for student feedback.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import List
from .preprocessing import simple_tokenize


class FeedbackDataset(Dataset):
    """Dataset for custom deep learning models (KimCNN, BiLSTM, etc.)"""
    def __init__(self, df: pd.DataFrame, vocab, max_len: int = 50):
        self.texts = df["text"].values
        self.labels = df["label"].values
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = simple_tokenize(text)
        encoded = self.vocab.encode(tokens, max_len=self.max_len)
        
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BERTFeedbackDataset(Dataset):
    """Dataset for PhoBERT or other Transformer-based models using HuggingFace tokenizers."""
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 64):
        self.texts = df["text"].values
        self.labels = df["label"].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }
