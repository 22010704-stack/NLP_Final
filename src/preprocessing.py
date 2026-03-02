"""
src/preprocessing.py
Text preprocessing utilities for Vietnamese student feedback classification.
"""

import re
import string


# Vietnamese stopwords (common, non-informative words)
VIETNAMESE_STOPWORDS = {
    "và", "của", "là", "có", "trong", "được", "cho", "với", "các", "một",
    "này", "đó", "về", "từ", "thì", "đã", "rất", "mà", "họ", "khi",
    "để", "bị", "hay", "như", "nên", "ra", "vì", "nhưng", "thế", "lại",
    "đây", "vào", "tôi", "ta", "chúng", "cũng", "không", "chỉ", "cần",
    "thầy", "cô", "giáo", "viên", "sinh", "môn", "học", "bài", "lớp",
    "những", "theo", "sau", "trước", "hơn", "đều", "còn", "qua", "nhiều",
    "phải", "ai", "gì", "nào", "thêm", "đến", "bao", "do", "tuy",
}


def clean_text(text: str) -> str:
    """
    Basic Vietnamese text cleaning:
    - Lowercase
    - Remove URLs, emails
    - Remove excessive punctuation
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove email
    text = re.sub(r"\S+@\S+", "", text)

    # Replace multiple punctuation with single
    text = re.sub(r"[!?]{2,}", "!", text)
    text = re.sub(r"\.{2,}", ".", text)

    # Remove special characters but keep Vietnamese diacritics and basic punctuation
    text = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?;:]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def simple_tokenize(text: str) -> list:
    """
    Simple word-level tokenization by splitting on whitespace.
    For Vietnamese, word segmentation is non-trivial; we use simple split
    as a reasonable baseline for character-sharing words.
    """
    text = clean_text(text)
    # Remove punctuation for token purposes
    text = re.sub(r"[.,!?;:]", " ", text)
    tokens = text.split()
    return tokens


def tokenize_with_stopword_removal(text: str) -> list:
    """Tokenize and remove stopwords."""
    tokens = simple_tokenize(text)
    return [t for t in tokens if t not in VIETNAMESE_STOPWORDS]


def add_noise_typo(text: str, noise_rate: float = 0.05, seed: int = 42) -> str:
    """
    Add noise to text for robustness testing:
    - Random character deletion
    - Random character insertion
    Used for robustness evaluation.
    """
    import random
    rng = random.Random(seed)
    chars = list(text)
    n_noisy = max(1, int(len(chars) * noise_rate))

    for _ in range(n_noisy):
        if not chars:
            break
        op = rng.choice(["delete", "duplicate"])
        idx = rng.randint(0, len(chars) - 1)
        if op == "delete":
            chars.pop(idx)
        elif op == "duplicate" and idx < len(chars):
            chars.insert(idx + 1, chars[idx])

    return "".join(chars)


def remove_diacritics(text: str) -> str:
    """
    Remove Vietnamese diacritics for no-accent robustness test.
    Replaces accented chars with base Latin character.
    """
    import unicodedata
    # Normalize to NFD then strip combining characters
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")
