from __future__ import annotations

import argparse
from pathlib import Path

from cs336_basics.tokenizer.bpe_train import train_bpe
from cs336_basics.tokenizer.io import save_vocab_and_merges, load_vocab_and_merges


def main():
    parser = argparse.ArgumentParser(description="Train and save BPE tokenizer")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to training text file",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Max vocab size (including bytes + special tokens)",
    )
    parser.add_argument(
        "--special_token",
        type=str,
        default="<|endoftext|>",
        help="Special token to add and keep atomic",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for tokenizer files",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Prefix name for output files",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / f"{args.name}.vocab.json"
    merges_path = output_dir / f"{args.name}.merges.json"

    print("▶ Training BPE tokenizer...")
    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=args.vocab_size,
        special_tokens=[args.special_token],
    )

    print("▶ Saving tokenizer files...")
    save_vocab_and_merges(
        vocab=vocab,
        merges=merges,
        vocab_path=str(vocab_path),
        merges_path=str(merges_path),
    )

    print("▶ Verifying save/load roundtrip...")
    vocab2, merges2 = load_vocab_and_merges(
        vocab_path=str(vocab_path),
        merges_path=str(merges_path),
    )

    assert vocab == vocab2, "Vocab mismatch after save/load"
    assert merges == merges2, "Merges mismatch after save/load"

    print("✅ Tokenizer saved successfully")
    print(f"   Vocab file : {vocab_path}")
    print(f"   Merges file: {merges_path}")
    print(f"   Vocab size : {len(vocab)}")
    print(f"   # merges   : {len(merges)}")


if __name__ == "__main__":
    main()
