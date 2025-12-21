# cs336_basics/tokenizer/bpe_train.py
from __future__ import annotations

from typing import Dict, List, Tuple
from collections import Counter, defaultdict

from .pretokenizer import iter_pretokens


BytesToken = bytes
Merge = Tuple[BytesToken, BytesToken]


def _word_to_symbols(
    pretoken: bytes,
    special_tokens_bytes: set[bytes],
) -> Tuple[BytesToken, ...]:
    """
    Convert a pretoken into initial BPE symbols.

    Why:
      - Normal pretokens start as individual bytes.
      - Special tokens must remain atomic and never be split or merged.
    """
    if pretoken in special_tokens_bytes:
        return (pretoken,)
    return tuple(bytes([b]) for b in pretoken)


def _init_pair_counts(
    words: Dict[Tuple[BytesToken, ...], int]
) -> Counter[Merge]:
    """
    Count adjacent symbol pairs across all words.

    Why:
      - BPE chooses the most frequent adjacent pair to merge.
      - We operate on 'words' (already aggregated) instead of raw text
        for efficiency.
    """
    pair_counts: Counter[Merge] = Counter()
    for sym_seq, freq in words.items():
        if len(sym_seq) < 2:
            continue
        for i in range(len(sym_seq) - 1):
            pair_counts[(sym_seq[i], sym_seq[i + 1])] += freq
    return pair_counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Merge]]:
    """
    Train a byte-level BPE tokenizer.

    Returns:
      vocab: dict[int, bytes] mapping token_id -> token_bytes
      merges: list[(bytes, bytes)] in the order they were created

    Why this function exists:
      - This is the *training* phase of a tokenizer.
      - The output is later used by encode/decode.
      - This mirrors how GPT-style tokenizers are built.
    """

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")

    if special_tokens is None:
        special_tokens = []

    special_tokens_bytes = {tok.encode("utf-8") for tok in special_tokens}

    # ------------------------------------------------------------------
    # 1) Initialize vocabulary
    # ------------------------------------------------------------------
    # Byte vocabulary: IDs 0..255 map to single-byte tokens
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Add special tokens after byte tokens (deterministic order)
    next_id = 256
    for tok in special_tokens:
        b = tok.encode("utf-8")
        if b not in vocab.values():
            vocab[next_id] = b
            next_id += 1

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} < initial vocab size {len(vocab)}"
        )

    # ------------------------------------------------------------------
    # 2) Pretokenize corpus and build initial word counts
    # ------------------------------------------------------------------
    words: Dict[Tuple[BytesToken, ...], int] = defaultdict(int)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    for pretoken_bytes in iter_pretokens(text, special_tokens):
        sym_seq = _word_to_symbols(pretoken_bytes, special_tokens_bytes)
        words[sym_seq] += 1

        # ------------------------------------------------------------------
    # 3) BPE merge loop (production-style early stopping)
    # ------------------------------------------------------------------
    merges: List[Merge] = []
    pair_counts = _init_pair_counts(words)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        max_freq = max(pair_counts.values())

        # 🔑 CRITICAL SPEED OPTIMIZATION
        # Stop if no pair occurs more than once
        if max_freq < 2:
            break

        candidates = [p for p, c in pair_counts.items() if c == max_freq]
        best_pair = max(candidates)

        a, b = best_pair
        merged = a + b

        vocab[next_id] = merged
        next_id += 1
        merges.append(best_pair)

        new_words: Dict[Tuple[BytesToken, ...], int] = defaultdict(int)
        new_pair_counts: Counter[Merge] = Counter()

        for sym_seq, freq in words.items():
            out: List[BytesToken] = []
            i = 0
            while i < len(sym_seq):
                if i < len(sym_seq) - 1 and sym_seq[i] == a and sym_seq[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(sym_seq[i])
                    i += 1

            out = tuple(out)
            new_words[out] += freq

            for j in range(len(out) - 1):
                new_pair_counts[(out[j], out[j + 1])] += freq

        words = new_words
        pair_counts = new_pair_counts

        

    return vocab, merges
