"""Minimal FFT and DRCT (DCT-II style) implementation."""

from __future__ import annotations

import cmath
import math
from typing import Iterable, List


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def fft(values: Iterable[complex]) -> List[complex]:
    """Compute the 1D Cooley-Tukey FFT for a power-of-two input length."""
    data = [complex(v) for v in values]
    size = len(data)

    if size == 0:
        return []
    if not _is_power_of_two(size):
        raise ValueError("fft input length must be a power of two")
    if size == 1:
        return data

    even = fft(data[0::2])
    odd = fft(data[1::2])

    result = [0j] * size
    half = size // 2
    for k in range(half):
        twiddle = cmath.exp(-2j * math.pi * k / size) * odd[k]
        result[k] = even[k] + twiddle
        result[k + half] = even[k] - twiddle

    return result


def ifft(values: Iterable[complex]) -> List[complex]:
    """Compute the inverse FFT for a power-of-two input length."""
    data = [complex(v) for v in values]
    size = len(data)

    if size == 0:
        return []

    transformed = fft(value.conjugate() for value in data)
    return [value.conjugate() / size for value in transformed]


def drct(values: Iterable[float]) -> List[float]:
    """Compute a DRCT using a DCT-II formulation accelerated by FFT."""
    data = [float(v) for v in values]
    size = len(data)

    if size == 0:
        return []

    mirrored = data + list(reversed(data))
    spectrum = fft(mirrored)

    return [
        0.5 * (spectrum[k] * cmath.exp(-1j * math.pi * k / (2 * size))).real
        for k in range(size)
    ]


def idrct(coefficients: Iterable[float]) -> List[float]:
    """Inverse transform for ``drct`` using the paired DCT-III formulation."""
    coeffs = [float(v) for v in coefficients]
    size = len(coeffs)

    if size == 0:
        return []

    scale = 2.0 / size
    return [
        scale
        * (
            0.5 * coeffs[0]
            + sum(
                coeffs[k] * math.cos(math.pi * k * (n + 0.5) / size)
                for k in range(1, size)
            )
        )
        for n in range(size)
    ]
