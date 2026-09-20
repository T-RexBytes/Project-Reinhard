"""
app/characterization/freq_offset.py
--------------------------------------
Carrier frequency offset (CFO) estimation.

Methods:
  1. PSD peak       : argmax(PSD) — works for AM/CW/narrowband
  2. Squared-signal : FFT(IQ^2) → peak/2 — works for BPSK, removes data
  3. Fourth-power   : FFT(IQ^4) → peak/4 — works for QPSK/8PSK

These are coarse estimates (within a fraction of the symbol rate).
Fine carrier recovery (PLL/Costas) is done in the demodulation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FreqOffsetResult:
    """Carrier frequency offset estimate."""
    offset_hz: float
    confidence: float       # 0–1 heuristic
    method: str
    peak_power_db: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "offset_hz":      self.offset_hz,
            "confidence":     self.confidence,
            "method":         self.method,
            "peak_power_db":  self.peak_power_db,
            "notes":          self.notes,
        }


def estimate_freq_offset_psd_peak(
    iq: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096,
) -> FreqOffsetResult:
    """Coarse offset: frequency of the PSD peak.

    Works best for: AM-DSB, AM-SSB, CW, narrowband signals.
    For QPSK/BPSK the carrier is suppressed, so this gives carrier ≈ 0
    unless there's a frequency offset.
    """
    x = np.asarray(iq, dtype=np.complex128)
    n = len(x)
    fft_n = min(fft_size, n)

    # Windowed FFT
    win = np.hanning(fft_n)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(x[:fft_n] * win, n=fft_n))) ** 2
    freqs    = np.fft.fftshift(np.fft.fftfreq(fft_n, d=1.0 / sample_rate))

    peak_idx = int(np.argmax(spectrum))
    peak_hz  = float(freqs[peak_idx])
    peak_db  = float(10.0 * np.log10(spectrum[peak_idx] + 1e-20))
    noise_db = float(np.percentile(10.0 * np.log10(spectrum + 1e-20), 20))
    snr_peak = peak_db - noise_db

    confidence = float(np.clip(snr_peak / 30.0, 0.1, 0.9))

    return FreqOffsetResult(
        offset_hz=peak_hz,
        confidence=confidence,
        method="psd_peak",
        peak_power_db=peak_db,
        notes=f"PSD peak SNR={snr_peak:.1f} dB",
    )


def estimate_freq_offset_squared(
    iq: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096,
) -> FreqOffsetResult:
    """Coarse offset via IQ^2 (Viterbi & Viterbi / squaring method).

    For BPSK: x^2 eliminates BPSK data modulation → spectral line at 2*fc.
    Offset = peak_freq / 2.
    Works for BPSK. Limited for QPSK (use fourth-power for QPSK).
    """
    x  = np.asarray(iq, dtype=np.complex128)
    x2 = x ** 2
    n  = len(x2)
    fft_n = min(fft_size, n)

    win      = np.hanning(fft_n)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(x2[:fft_n] * win, n=fft_n))) ** 2
    freqs    = np.fft.fftshift(np.fft.fftfreq(fft_n, d=1.0 / sample_rate))

    peak_idx = int(np.argmax(spectrum))
    peak_hz  = float(freqs[peak_idx])
    peak_db  = float(10.0 * np.log10(spectrum[peak_idx] + 1e-20))
    noise_db = float(np.percentile(10.0 * np.log10(spectrum + 1e-20), 20))
    snr_peak = peak_db - noise_db

    offset_hz  = peak_hz / 2.0
    confidence = float(np.clip(snr_peak / 25.0, 0.05, 0.85))

    return FreqOffsetResult(
        offset_hz=offset_hz,
        confidence=confidence,
        method="squared",
        peak_power_db=peak_db,
        notes=f"BPSK squaring: f_peak={peak_hz/1e3:.1f} kHz -> offset={offset_hz/1e3:.1f} kHz",
    )


def estimate_freq_offset_fourth_power(
    iq: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096,
) -> FreqOffsetResult:
    """Coarse offset via IQ^4 (Viterbi & Viterbi QPSK method).

    For QPSK: x^4 eliminates all 4 phases → spectral line at 4*fc.
    Offset = peak_freq / 4.
    Also works for 8PSK: use IQ^8 (not implemented here — overkill for MVP).
    """
    x  = np.asarray(iq, dtype=np.complex128)
    x4 = x ** 4
    n  = len(x4)
    fft_n = min(fft_size, n)

    win      = np.hanning(fft_n)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(x4[:fft_n] * win, n=fft_n))) ** 2
    freqs    = np.fft.fftshift(np.fft.fftfreq(fft_n, d=1.0 / sample_rate))

    peak_idx = int(np.argmax(spectrum))
    peak_hz  = float(freqs[peak_idx])
    peak_db  = float(10.0 * np.log10(spectrum[peak_idx] + 1e-20))
    noise_db = float(np.percentile(10.0 * np.log10(spectrum + 1e-20), 20))
    snr_peak = peak_db - noise_db

    offset_hz  = peak_hz / 4.0
    confidence = float(np.clip(snr_peak / 25.0, 0.05, 0.85))

    return FreqOffsetResult(
        offset_hz=offset_hz,
        confidence=confidence,
        method="fourth_power",
        peak_power_db=peak_db,
        notes=f"QPSK 4th-power: f_peak={peak_hz/1e3:.1f} kHz -> offset={offset_hz/1e3:.1f} kHz",
    )


def estimate_freq_offset(
    iq: np.ndarray,
    sample_rate: float,
    mod_hint: Optional[str] = None,
    fft_size: int = 4096,
) -> FreqOffsetResult:
    """Best-effort carrier frequency offset estimation.

    Picks the best method based on optional modulation hint.
    If no hint, runs all three and returns the most confident result.

    Args:
        iq          : Complex IQ (1D)
        sample_rate : Sampling rate in Hz
        mod_hint    : Optional modulation hint ('BPSK','QPSK','FSK','QAM16',...)
        fft_size    : FFT size

    Returns:
        FreqOffsetResult
    """
    hint = (mod_hint or "").upper()

    # Order matters: "AM" is a substring of "QAM", so QAM/QPSK/PSK must be
    # matched before the generic AM/FM/CW branch or 16QAM gets misrouted to
    # psd_peak (which is unstable on spectrally-flat wideband QAM).
    if "FSK" in hint:
        return estimate_freq_offset_psd_peak(iq, sample_rate, fft_size)

    if "BPSK" in hint:
        return estimate_freq_offset_squared(iq, sample_rate, fft_size)

    if "QAM" in hint or "QPSK" in hint or "PSK" in hint:
        return estimate_freq_offset_fourth_power(iq, sample_rate, fft_size)

    # Constant-envelope / analog: psd_peak is reliable.
    if "AM" in hint or "FM" in hint or "CW" in hint:
        return estimate_freq_offset_psd_peak(iq, sample_rate, fft_size)

    # No hint — run all three, pick most confident
    r_psd  = estimate_freq_offset_psd_peak(iq, sample_rate, fft_size)
    r_sq   = estimate_freq_offset_squared(iq, sample_rate, fft_size)
    r_4th  = estimate_freq_offset_fourth_power(iq, sample_rate, fft_size)

    best = max([r_psd, r_sq, r_4th], key=lambda r: r.confidence)
    best.notes = f"Auto-selected from [psd,sq,4th]; {best.notes}"
    return best
