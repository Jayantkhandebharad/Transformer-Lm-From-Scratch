from __future__ import annotations

import numpy as np
import torch
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample language modeling batches from a 1D numpy token array.

    Returns:
        x, y: torch.LongTensor of shape (batch_size, context_length)
    """
    if dataset.ndim != 1:
        raise ValueError(f"Expected 1D dataset array, got shape {dataset.shape}")

    n = int(dataset.shape[0])
    if n <= context_length:
        raise ValueError(
            f"Dataset too small (len={n}) for context_length={context_length}"
        )

    # Number of valid starting indices
    num_starts = n - context_length

    # Convert dataset to torch tensor (CPU)
    data = torch.from_numpy(dataset).to(dtype=torch.long)

    # Sample random start indices
    starts = torch.randint(
        low=0, high=num_starts, size=(batch_size,), device="cpu"
    )

    # Build index matrix
    offsets = torch.arange(context_length, device="cpu")
    idx = starts[:, None] + offsets[None, :]

    x = data[idx]
    y = data[idx + 1]

    return x.to(device), y.to(device)
