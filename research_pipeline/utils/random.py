from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    seed: int


def set_global_seed(seed: int) -> SeedState:
    """Seed Python, NumPy, and torch when torch is installed."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return SeedState(seed=seed)


def stable_rng(seed: int, salt: str = "") -> np.random.Generator:
    value = seed
    for char in salt:
        value = (value * 131 + ord(char)) % (2**32)
    return np.random.default_rng(value)

