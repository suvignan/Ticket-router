"""What the API runs. Tests call THESE functions, never a test-only copy."""
from functools import lru_cache
from pathlib import Path

import numpy as np
from tokenizers import BertWordPieceTokenizer

from src.constants import ROOT
from src.text import clean_text, first_350, format_for_bert

BERT_VOCAB = ROOT / "models" / "bert_vocab.txt"
MAX_LEN = 512


@lru_cache(maxsize=None)
def bert_tokenizer(vocab_path: Path = BERT_VOCAB) -> BertWordPieceTokenizer:
    """Loaded once per process. Same WordPiece as KerasHub (100% token-ID parity, Day 13), without TensorFlow."""
    tok = BertWordPieceTokenizer(str(vocab_path), lowercase=True)     # adds [CLS] ... [SEP]
    tok.enable_truncation(MAX_LEN)
    return tok


def encode_for_bert(raw_text, vocab_path: Path = BERT_VOCAB) -> tuple[np.ndarray, np.ndarray]:
    """raw complaint -> (token_ids, padding_mask), shape (1, n), real length (no pad-to-512: same argmax, faster)."""
    ids = np.array([bert_tokenizer(vocab_path).encode(format_for_bert(first_350(clean_text(raw_text)))).ids], dtype=np.int32)
    return ids, (ids != 0).astype(np.int32)
