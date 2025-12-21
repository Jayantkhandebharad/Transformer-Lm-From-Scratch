# cs336_basics/tokenizer/io.py
from __future__ import annotations

import json
from typing import Dict, List, Tuple


def save_vocab_and_merges(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], vocab_path: str, merges_path: str) -> None:
    """
    Save vocab and merges in a simple JSON format.

    Why:
      The assignment asks you to serialize these to disk for inspection/experiments. :contentReference[oaicite:6]{index=6}
    """
    vocab_json = {str(k): v.decode("latin1") for k, v in vocab.items()}  # latin1 is 1-1 bytes mapping
    merges_json = [[a.decode("latin1"), b.decode("latin1")] for a, b in merges]

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f)

    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_json, f)


def load_vocab_and_merges(vocab_path: str, merges_path: str) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Load vocab and merges saved by save_vocab_and_merges.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)

    with open(merges_path, "r", encoding="utf-8") as f:
        merges_json = json.load(f)

    vocab = {int(k): v.encode("latin1") for k, v in vocab_json.items()}
    merges = [(a.encode("latin1"), b.encode("latin1")) for a, b in merges_json]
    return vocab, merges
