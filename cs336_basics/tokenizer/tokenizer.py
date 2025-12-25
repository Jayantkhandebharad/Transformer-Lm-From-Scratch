from __future__ import annotations

from typing import Dict, List, Iterable
from pathlib import Path

from .pretokenizer import iter_pretokens
from.io import load_vocab_and_merges

class Tokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer.

    Why this class:
    - converts raw text into token IDs using trained BPE merges
    - converts token IDs back into text
    - This is the bridge between text and the transformer
    """

    def __init__(
            self,
            vocab:Dict[int, bytes],
            merges:List[tuple[bytes, bytes]],
            special_tokens:List[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        
        # build reverse vocab: bytes -> token_id
        self.bytes_to_id = {v: k for k,v in vocab.items()}
        
        # Convert merges to fast-lookup dict: (a,b) -> merged_bytes
        self.merge_map = {(a,b):a+b for (a,b) in merges}

        if special_tokens is None:
            special_tokens = []
        
        self.special_tokens = special_tokens
        self.special_tokens_bytes = {tok.encode("utf-8") for tok in special_tokens}

    
    # ------------------------------------------------
    # Construction Helper
    # ------------------------------------------------

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        """
        Load a Tokenizer from vocab and merges files.

        Why:
          - Separate construction logic for clarity.
          - Loading from files is common in practice.
        """
        vocab, merges = load_vocab_and_merges(str(vocab_path), str(merges_path))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    # ------------------------------------------------
    # Encoding 
    # ------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs.

        Why:
            - This is what the mode consumes
        """
        token_ids: List[int] = []

        for pretoken_bytes in iter_pretokens(text, self.special_tokens):
            symbols = self._apply_bpe(pretoken_bytes)
            for sym in symbols:
                token_ids.append(self.bytes_to_id[sym])
        return token_ids
    
    def encode_iterable(self, texts:Iterable[str]) -> Iterable[List[int]]:
        """
        Encode an iterable of texts into lists of token IDs.

        Why:
            - Useful for batch processing large corpora.
        """
        for text in texts:
            for token_ids in self.encode(text):
                yield token_ids
    
    # ------------------------------------------------
    # Decoding
    # ------------------------------------------------
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Why:
            - Needed for generation and debugging.
        """
        byte_chunks = [self.vocab[token_id] for token_id in token_ids]
        all_bytes = b"".join(byte_chunks)
        return all_bytes.decode("utf-8", errors="replace")
    
    # ------------------------------------------------
    # Internal helpers
    # ------------------------------------------------

    def _apply_bpe(self, pretoken: bytes) -> List[bytes]:
        """
        Apply BPE merges to a single pretoken (as bytes).

        why:
        - Replays merge rules learned during training
        - no frequencies, no learning, just deterministic merging.
        """

        # special token stay atomic
        if pretoken in self.special_tokens_bytes:
            return [pretoken]
        
        # start from byte symbols
        symbols = [bytes([b]) for b in pretoken]

        for a,b in self.merges:
            i = 0
            new_symbols = []
            while i< len(symbols):
                if(
                    i<len(symbols)-1
                    and symbols[i] == a
                    and symbols[i+1] == b
                ):
                    new_symbols.append(a+b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols