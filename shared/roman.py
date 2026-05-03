"""Shared roman-numeral formatting helpers."""

from __future__ import annotations


def to_roman(n: int) -> str:
    """Convert a small positive integer to a Roman numeral string."""
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    result = ""
    for value, symbol in vals:
        while n >= value:
            result += symbol
            n -= value
    return result
