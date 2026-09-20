"""Spectrogram and waterfall computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from app.models import ComplexSignal


@dataclass
class SpectrogramResult:
    """Result of spectrogram computation."""
    spectrogram_db: np.ndarray    # 2D array: (time_bins, freq_bins)
    frequencies: np.ndarray       # Hz
    time_bins: np.ndarray         # seconds
    sample_rate: float
    fft_size: int
    overlap: float
    window: str

    @property
    def num_time_bins(self) -> int:
        return self.spectrogram_db.shape[1]

    @property
    def num_freq_bins(self) -> int:
        return self.spectrogram_db.shape[0]

    @property
    def duration(self) -> float:
        return float(self.time_bins[-1] - self.time_bins[0]) if len(self.time_bins) > 1 else 0.0

    def to_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "fft_size": self.fft_size,
            "overlap": self.overlap,
            "window": self.window,
            "num_time_bins": self.num_time_bins,
            "num_freq_bins": self.num_freq_bins,
            "duration_seconds": self.duration,
            "freq_min_hz": float(self.frequencies[0]),
            "freq_max_hz": float(self.frequencies[-1]),
            "db_min": float(np.min(self.spectrogram_db)),
            "db_max": float(np.max(self.spectrogram_db)),
        }


def compute_spectrogram(
    signal: ComplexSignal,
    fft_size: int = 1024,
    overlap: float = 0.75,
    window: str = "hann",
    max_time_bins: Optional[int] = None,
) -> SpectrogramResult:
    """Compute spectrogram (waterfall data).

    Args:
        signal: Input ComplexSignal
        fft_size: FFT size per frame
        overlap: Overlap ratio (0-1)
        window: Window function name
        max_time_bins: Limit time resolution for large files

    Returns:
        SpectrogramResult with 2D spectrogram data
    """
    samples = signal.data
    sample_rate = signal.metadata.sample_rate
    if sample_rate is None:
        raise ValueError("sample_rate required for spectrogram")

    actual_fft = min(fft_size, len(samples)) if len(samples) > 0 else fft_size
    noverlap = int(actual_fft * overlap)
    if noverlap >= actual_fft:
        noverlap = max(0, actual_fft - 1)

    freqs, times, Sxx = scipy_signal.spectrogram(
        samples,
        fs=sample_rate,
        nperseg=actual_fft,
        noverlap=noverlap,
        window=window,
        return_onesided=False,
        scaling="spectrum",
    )

    freqs = np.fft.fftshift(freqs)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    Sxx_db = 10.0 * np.log10(Sxx + 1e-20)

    if max_time_bins and Sxx_db.shape[1] > max_time_bins:
        step = Sxx_db.shape[1] // max_time_bins
        Sxx_db = Sxx_db[:, ::step]
        times = times[::step]

    return SpectrogramResult(
        spectrogram_db=Sxx_db,
        frequencies=freqs,
        time_bins=times,
        sample_rate=sample_rate,
        fft_size=actual_fft,
        overlap=overlap,
        window=window,
    )
