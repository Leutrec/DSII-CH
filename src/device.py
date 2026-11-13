"""Detección automática del dispositivo de cómputo."""

import torch


def get_device() -> torch.device:
    """Elige CUDA, MPS o CPU según el hardware disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
