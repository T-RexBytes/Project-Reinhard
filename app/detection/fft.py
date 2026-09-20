"""FFT and Power Spectral Density computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from app.models import ComplexSignal


@dataclass
class PSDResult:
    """Result of PSD computation."""
    frequencies: np.ndarray      # Hz, baseband centered at 0
    psd_db: np.ndarray           # dB/Hz
    psd_linear: np.ndarray       # linear scale
    sample_rate: float           # Hz
    fft_size: int
    window: str
    num_averages: int
    center_frequency: Optional[float] = None  # Native RF center frequency in Hz

    @property
    def freq_resolution(self) -> float:
        return self.sample_rate / self.fft_size

    @property
    def rf_frequencies(self) -> Optional[np.ndarray]:
        """Absolute RF frequencies (center_frequency + baseband frequencies) if fc known."""
        if self.center_frequency is not None:
            return self.frequencies + self.center_frequency
        return None

    def to_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "fft_size": self.fft_size,
            "window": self.window,
            "num_averages": self.num_averages,
            "freq_resolution_hz": self.freq_resolution,
            "freq_min_hz": float(self.frequencies[0]),
            "freq_max_hz": float(self.frequencies[-1]),
            "center_frequency_hz": self.center_frequency,
            "rf_freq_min_hz": float(self.frequencies[0] + self.center_frequency) if self.center_frequency is not None else None,
            "rf_freq_max_hz": float(self.frequencies[-1] + self.center_frequency) if self.center_frequency is not None else None,
            "psd_min_db": float(np.min(self.psd_db)),
            "psd_max_db": float(np.max(self.psd_db)),
        }


def compute_psd(
    signal: ComplexSignal,
    fft_size: int = 1024,
    window: str = "hann",
    overlap: float = 0.5,
    average: bool = True,
) -> PSDResult:
    """Compute Power Spectral Density using Welch's method.

    Args:
        signal: Input ComplexSignal
        fft_size: FFT size (number of points)
        window: Window function name
        overlap: Overlap ratio (0-1)
        average: Whether to average across segments

    Returns:
        PSDResult with frequency and power arrays
    """
    samples = signal.data
    sample_rate = signal.metadata.sample_rate
    if sample_rate is None:
        raise ValueError("sample_rate required for PSD computation")

    actual_fft = min(fft_size, len(samples)) if len(samples) > 0 else fft_size
    noverlap = int(actual_fft * overlap)
    if noverlap >= actual_fft:
        noverlap = max(0, actual_fft - 1)

    freqs, psd_linear = scipy_signal.welch(
        samples,
        fs=sample_rate,
        nperseg=actual_fft,
        noverlap=noverlap,
        window=window,
        return_onesided=False,
    )

    freqs = np.fft.fftshift(freqs)
    psd_linear = np.fft.fftshift(psd_linear)

    psd_linear = np.abs(psd_linear)
    psd_db = 10.0 * np.log10(psd_linear + 1e-20)

    n_segments = max(1, (len(samples) - actual_fft) // max(1, (actual_fft - noverlap)) + 1)

    return PSDResult(
        frequencies=freqs,
        psd_db=psd_db,
        psd_linear=psd_linear,
        sample_rate=sample_rate,
        fft_size=actual_fft,
        window=window,
        num_averages=n_segments if average else 1,
        center_frequency=signal.metadata.center_frequency,
    )


def compute_fft(
    signal: ComplexSignal,
    fft_size: Optional[int] = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute single FFT frame.

    Returns:
        (frequencies_hz, complex_fft)
    """
    samples = signal.data
    sample_rate = signal.metadata.sample_rate
    if sample_rate is None:
        raise ValueError("sample_rate required for FFT")

    if fft_size is None:
        fft_size = min(len(samples), 4096)

    if len(samples) < fft_size:
        padded = np.zeros(fft_size, dtype=np.complex64)
        padded[: len(samples)] = samples
        samples = padded

    win = scipy_signal.get_window(window, fft_size)
    windowed = samples[:fft_size] * win

    fft_result = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))

    return freqs, fft_result
