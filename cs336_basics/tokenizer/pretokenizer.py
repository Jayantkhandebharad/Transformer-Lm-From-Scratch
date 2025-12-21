from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple
import regex as re

# GPT-2 / tiktoken-style pre-tokenizer pattern from the assignment PDF.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _split_on_special_tokens(text: str, special_tokens: List[str]) -> List[Tuple[str, bool]]:
    """
    Split text into segments that are either:
      - normal text (is_special=False)
      - exact special token strings (is_special=True)

    Why:
      The assignment requires that no merging happen across special-token boundaries.
      So we isolate specials before regex pre-tokenization.
    """
    if not special_tokens:
        return [(text, False)]

    # Build a regex that matches any special token exactly.
    # re.escape is crucial because special tokens include characters like |, <, >, etc.
    pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
    parts = re.split(pattern, text)

    out: List[Tuple[str, bool]] = []
    for p in parts:
        if p == "":
            continue
        if p in special_tokens:
            out.append((p, True))
        else:
            out.append((p, False))
    return out


def iter_pretokens(text: str, special_tokens: List[str]) -> Iterator[bytes]:
    """
    Yield pre-tokens as UTF-8 bytes, with special tokens yielded as their UTF-8 bytes too.

    Why:
      - Training counts pairs inside each pre-token independently.
      - Using an iterator avoids storing all pretokens for huge corpora.
    """
    for segment, is_special in _split_on_special_tokens(text, special_tokens):
        if is_special:
            yield segment.encode("utf-8")
            continue

        # Use finditer to avoid materializing a giant list.
        for m in re.finditer(PAT, segment):
            s = m.group(0)
            if s:
                yield s.encode("utf-8")
