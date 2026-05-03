"""Shared spin-observable helper functions."""

from __future__ import annotations

import numpy as np


def compute_two_s_from_s2(s2: float) -> float:
    """Recover 2S from <S^2> via S(S+1) = <S^2>."""
    return float(np.sqrt(1.0 + 4.0 * float(s2)) - 1.0)
