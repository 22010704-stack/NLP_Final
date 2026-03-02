"""
src/vocabulary.py
Vocabulary builder for non-BERT models.
"""

from collections import Counter
from typing import List, Iterable
import pickle


class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.itos = {0: "<pad>", 1: "<unk>"}
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.freqs = Counter()

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, token_iterator: Iterable[List[str]]):
        """Build vocab from an iterator of token lists."""
        for tokens in token_iterator:
            self.freqs.update(tokens)

        # Add words that meet min_freq
        idx = len(self.itos)
        for word, freq in self.freqs.items():
            if freq >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def encode(self, tokens: List[str], max_len: int = None) -> List[int]:
        """Convert tokens to indices with optional padding/truncation."""
        encoded = [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]
        
        if max_len is not None:
            if len(encoded) > max_len:
                encoded = encoded[:max_len]
            else:
                encoded += [self.stoi["<pad>"]] * (max_len - len(encoded))
        
        return encoded

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices back to tokens."""
        return [self.itos.get(idx, "<unk>") for idx in indices]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
