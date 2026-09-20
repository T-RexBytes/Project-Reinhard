"""Bandwidth estimation methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.detection.fft import PSDResult
from app.detection.noise_floor import NoiseFloorResult


@dataclass
class BandwidthResult:
    """Bandwidth estimation result."""
    bandwidth_hz: float
    lower_3db_hz: float
    upper_3db_hz: float
    center_freq_hz: float
    method: str
    occupancy_percent: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "bandwidth_hz": self.bandwidth_hz,
            "lower_3db_hz": self.lower_3db_hz,
            "upper_3db_hz": self.upper_3db_hz,
            "center_freq_hz": self.center_freq_hz,
            "method": self.method,
            "occupancy_percent": self.occupancy_percent,
        }


def estimate_bandwidth_3db(
    psd: PSDResult,
    freq_start_hz: Optional[float] = None,
    freq_end_hz: Optional[float] = None,
) -> BandwidthResult:
    """Estimate bandwidth using -3dB points from peak.

    Args:
        psd: PSD result
        freq_start_hz: Search region start (full band if None)
        freq_end_hz: Search region end (full band if None)

    Returns:
        BandwidthResult
    """
    freqs = psd.frequencies
    psd_db = psd.psd_db

    if freq_start_hz is not None:
        mask = freqs >= freq_start_hz
    else:
        mask = np.ones(len(freqs), dtype=bool)
    if freq_end_hz is not None:
        mask &= freqs <= freq_end_hz

    search_indices = np.where(mask)[0]
    peak_idx = search_indices[np.argmax(psd_db[search_indices])]
    peak_power = psd_db[peak_idx]

    threshold_3db = peak_power - 3.0

    left_indices = np.where(psd_db[:peak_idx] <= threshold_3db)[0]
    right_indices = np.where(psd_db[peak_idx:] <= threshold_3db)[0]

    if len(left_indices) > 0:
        lower_freq = float(freqs[left_indices[-1]])
    else:
        lower_freq = float(freqs[0])

    if len(right_indices) > 0:
        upper_freq = float(freqs[peak_idx + right_indices[0]])
    else:
        upper_freq = float(freqs[-1])

    bw = abs(upper_freq - lower_freq)
    center = (upper_freq + lower_freq) / 2.0

    total_bw = float(freqs[-1] - freqs[0])
    occupancy = (bw / total_bw * 100.0) if total_bw > 0 else 0.0

    return BandwidthResult(
        bandwidth_hz=bw,
        lower_3db_hz=lower_freq,
        upper_3db_hz=upper_freq,
        center_freq_hz=center,
        method="3db",
        occupancy_percent=occupancy,
    )


def estimate_bandwidth_fractional(
    psd: PSDResult,
    fraction: float = 0.99,
    freq_start_hz: Optional[float] = None,
    freq_end_hz: Optional[float] = None,
) -> BandwidthResult:
    """Estimate bandwidth containing a fraction of total power.

    Args:
        psd: PSD result
        fraction: Fraction of total power (0-1)
        freq_start_hz: Search region start
        freq_end_hz: Search region end

    Returns:
        BandwidthResult
    """
    freqs = psd.frequencies
    psd_linear = psd.psd_linear

    if freq_start_hz is not None:
        mask = freqs >= freq_start_hz
    else:
        mask = np.ones(len(freqs), dtype=bool)
    if freq_end_hz is not None:
        mask &= freqs <= freq_end_hz

    search_indices = np.where(mask)[0]
    total_power = np.sum(psd_linear[search_indices])

    sorted_idx = search_indices[np.argsort(psd_linear[search_indices])[::-1]]
    cumulative = np.cumsum(psd_linear[sorted_idx])
    target_power = fraction * total_power

    keep = np.where(cumulative <= target_power)[0]
    if len(keep) == 0:
        keep = np.array([0])

    freqs_in_band = freqs[sorted_idx[keep]]
    lower = float(np.min(freqs_in_band))
    upper = float(np.max(freqs_in_band))

    bw = abs(upper - lower)
    center = (upper + lower) / 2.0

    total_bw = float(freqs[-1] - freqs[0])
    occupancy = (bw / total_bw * 100.0) if total_bw > 0 else 0.0

    return BandwidthResult(
        bandwidth_hz=bw,
        lower_3db_hz=lower,
        upper_3db_hz=upper,
        center_freq_hz=center,
        method=f"fractional_{fraction}",
        occupancy_percent=occupancy,
    )


def estimate_bandwidth(
    psd: PSDResult,
    method: str = "3db",
    fraction: float = 0.99,
    freq_start_hz: Optional[float] = None,
    freq_end_hz: Optional[float] = None,
) -> BandwidthResult:
    """Estimate bandwidth using specified method.

    Args:
        psd: PSD result
        method: "3db" or "fractional"
        fraction: Power fraction (for fractional method)
        freq_start_hz: Search region start
        freq_end_hz: Search region end

    Returns:
        BandwidthResult
    """
    if method == "3db":
        return estimate_bandwidth_3db(psd, freq_start_hz, freq_end_hz)
    elif method == "fractional":
        return estimate_bandwidth_fractional(psd, fraction, freq_start_hz, freq_end_hz)
    else:
        raise ValueError(f"Unknown method: {method}")
