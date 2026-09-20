"""Adaptive noise floor estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.models import ComplexSignal
from app.detection.fft import PSDResult


@dataclass
class NoiseFloorResult:
    """Noise floor estimation result."""
    noise_floor_db: float
    noise_floor_linear: float
    method: str
    percentile_used: Optional[float] = None
    num_samples_used: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "noise_floor_db": self.noise_floor_db,
            "noise_floor_linear": self.noise_floor_linear,
            "method": self.method,
            "percentile_used": self.percentile_used,
            "num_samples_used": self.num_samples_used,
        }


def estimate_noise_floor_percentile(
    psd: PSDResult,
    percentile: float = 10.0,
) -> NoiseFloorResult:
    """Estimate noise floor using lower percentile of PSD.

    Assumes signal occupies a minority of the band. The lower percentile
    captures the noise-only bins.

    Args:
        psd: PSD result
        percentile: Percentile to use (lower = more aggressive)

    Returns:
        NoiseFloorResult
    """
    noise_linear = np.percentile(psd.psd_linear, percentile)
    noise_db = 10.0 * np.log10(noise_linear + 1e-20)

    return NoiseFloorResult(
        noise_floor_db=float(noise_db),
        noise_floor_linear=float(noise_linear),
        method="percentile",
        percentile_used=percentile,
        num_samples_used=len(psd.psd_linear),
    )


def estimate_noise_floor_median(
    psd: PSDResult,
) -> NoiseFloorResult:
    """Estimate noise floor using median of PSD.

    Good default when signal bandwidth is unknown.

    Returns:
        NoiseFloorResult
    """
    noise_linear = np.median(psd.psd_linear)
    noise_db = 10.0 * np.log10(noise_linear + 1e-20)

    return NoiseFloorResult(
        noise_floor_db=float(noise_db),
        noise_floor_linear=float(noise_linear),
        method="median",
        num_samples_used=len(psd.psd_linear),
    )


def estimate_noise_floor_mad(
    psd: PSDResult,
    iterations: int = 3,
) -> NoiseFloorResult:
    """Estimate noise floor using median absolute deviation (MAD).

    Iteratively refines by removing outliers above the noise.

    Args:
        psd: PSD result
        iterations: Number of refinement iterations

    Returns:
        NoiseFloorResult
    """
    power = psd.psd_linear.copy()

    for _ in range(iterations):
        med = np.median(power)
        mad = np.median(np.abs(power - med))
        threshold = med + 3.0 * mad
        power = power[power <= threshold]

    noise_linear = np.median(power)
    noise_db = 10.0 * np.log10(noise_linear + 1e-20)

    return NoiseFloorResult(
        noise_floor_db=float(noise_db),
        noise_floor_linear=float(noise_linear),
        method="mad",
        num_samples_used=len(power),
    )


def estimate_noise_floor(
    psd: PSDResult,
    method: str = "mad",
    percentile: float = 10.0,
) -> NoiseFloorResult:
    """Estimate noise floor using specified method.

    Args:
        psd: PSD result
        method: "percentile", "median", or "mad"
        percentile: Percentile value (only for percentile method)

    Returns:
        NoiseFloorResult
    """
    if method == "percentile":
        return estimate_noise_floor_percentile(psd, percentile)
    elif method == "median":
        return estimate_noise_floor_median(psd)
    elif method == "mad":
        return estimate_noise_floor_mad(psd)
    else:
        raise ValueError(f"Unknown method: {method}")
