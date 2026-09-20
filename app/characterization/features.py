"""
app/characterization/features.py
----------------------------------
Extracts modulation-discriminating features from raw complex IQ.

Features computed:
  - phase_variance          : var(angle(IQ))           — high for FM/FSK, low for PSK
  - amplitude_variance_norm : std(|IQ|) / mean(|IQ|)   — high for QAM, low for PSK/BPSK
  - instantaneous_freq_std  : std(d/dt angle(IQ))       — FSK has large swings
  - instantaneous_freq_mean : mean(|inst_freq|)
  - kurtosis_amplitude      : excess kurtosis of |IQ|   — discriminates modulation order
  - kurtosis_phase          : excess kurtosis of angle(IQ)
  - C20                     : 2nd order moment (signal power)
  - C21                     : |E[x^2]| — phase symmetry indicator
  - C40                     : 4th-order cumulant (key for BPSK vs QPSK vs 16QAM)
  - C42                     : C(4,2) cumulant (modulation classifier standard feature)
  - C41                     : C(4,1) cumulant
  - zero_crossing_rate      : IQ zero crossings (FSK indicator)
  - spectral_flatness       : Wiener entropy (noise-like vs tonal)
  - papr_db                 : Peak-to-Average Power Ratio in dB

Reference theoretical values (high SNR):
  |C42| / C20^2
  BPSK  : 2.0
  QPSK  : 1.0
  16QAM : 0.68
  64QAM : 0.619
  FSK   : ~0  (C42 ≈ 0 for constant-envelope)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Optional

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis


@dataclass
class FeatureVector:
    """All modulation-discriminating features derived from one IQ segment."""

    # ── Phase features ──────────────────────────────────────────────────────
    phase_variance: float = 0.0
    """Phase variance (detrended to remove carrier offset slope) — low for PSK, high for FM/FSK"""

    phase_std: float = 0.0
    """std(angle(IQ)) — bounded in [0, pi], low for PSK, high for FM/FSK"""

    kurtosis_phase: float = 0.0
    """Excess kurtosis of instantaneous phase"""

    # ── Amplitude features ───────────────────────────────────────────────────
    amplitude_variance_norm: float = 0.0
    """std(|IQ|) / (mean(|IQ|) + eps) — low for constant-envelope (PSK/FSK), high for QAM"""

    kurtosis_amplitude: float = 0.0
    """Excess kurtosis of |IQ| — discriminates BPSK vs QPSK vs QAM"""

    papr_db: float = 0.0
    """Peak-to-Average Power Ratio in dB — QAM has high PAPR"""

    # ── Instantaneous frequency features ────────────────────────────────────
    instantaneous_freq_mean: float = 0.0
    """mean(|d/dt angle(IQ)|) * fs / (2*pi) in Hz — carrier offset + FM deviation"""

    instantaneous_freq_std: float = 0.0
    """std(d/dt angle(IQ)) * fs / (2*pi) in Hz — FSK has high std"""

    instantaneous_freq_mean_norm: float = 0.0
    """mean(|d/dt angle(IQ)|) / (2*pi) in cycles/sample (dimensionless, scale-invariant)"""

    instantaneous_freq_std_norm: float = 0.0
    """std(d/dt angle(IQ)) / (2*pi) in cycles/sample (dimensionless, scale-invariant)"""

    # ── Physical signal parameters ──────────────────────────────────────────
    raw_power: float = 0.0
    """Mean power E[|x|^2] of the input segment"""

    dc_offset_real: float = 0.0
    """Real part of native DC mean E[x.real]"""

    dc_offset_imag: float = 0.0
    """Imaginary part of native DC mean E[x.imag]"""

    # ── Higher-order cumulants (HOC) ─────────────────────────────────────────
    C20: float = 0.0
    """2nd-order moment: E[|x|^2] (signal power)"""

    C21_abs: float = 0.0
    """|E[x^2]| — measures phase symmetry (0 for QPSK/QAM, non-zero for BPSK/AM)"""

    C21_sym: float = 0.0
    """|C21| / C20 — normalized phase symmetry ratio (BPSK≈1.0, QPSK/QAM/FSK≈0)"""

    C40_abs: float = 0.0
    """|C(4,0)| = |E[x^4] - 3*E[x^2]^2|"""

    C41_abs: float = 0.0
    """|C(4,1)| = |E[x^3*conj(x)] - 3*E[x^2]*E[|x|^2]|"""

    C42_abs: float = 0.0
    """|C(4,2)| = |E[|x|^4] - |E[x^2]|^2 - 2*E[|x|^2]^2|"""

    # Normalized cumulant ratio (the main discriminator)
    C42_norm: float = 0.0
    """|C42| / C20^2 — BPSK≈2.0, QPSK≈1.0, 16QAM≈0.68, FSK≈0"""

    # ── Spectral features ────────────────────────────────────────────────────
    spectral_flatness: float = 0.0
    """Wiener entropy: geometric_mean(PSD) / arithmetic_mean(PSD) — 0=tonal, 1=flat/noise"""

    zero_crossing_rate: float = 0.0
    """Fraction of samples where I or Q changes sign (0-1)"""

    # ── Metadata ─────────────────────────────────────────────────────────────
    num_samples: int = 0
    sample_rate: Optional[float] = None
    snr_db_estimate: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        """Return numeric features as a flat float32 array for ML use."""
        return np.array([
            self.phase_variance,
            self.kurtosis_phase,
            self.amplitude_variance_norm,
            self.kurtosis_amplitude,
            self.papr_db,
            self.instantaneous_freq_mean,
            self.instantaneous_freq_std,
            self.instantaneous_freq_mean_norm,
            self.instantaneous_freq_std_norm,
            self.raw_power,
            self.dc_offset_real,
            self.dc_offset_imag,
            self.C20,
            self.C21_abs,
            self.C21_sym,
            self.C40_abs,
            self.C41_abs,
            self.C42_abs,
            self.C42_norm,
            self.spectral_flatness,
            self.zero_crossing_rate,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "phase_variance", "kurtosis_phase",
            "amplitude_variance_norm", "kurtosis_amplitude", "papr_db",
            "instantaneous_freq_mean", "instantaneous_freq_std",
            "instantaneous_freq_mean_norm", "instantaneous_freq_std_norm",
            "raw_power", "dc_offset_real", "dc_offset_imag",
            "C20", "C21_abs", "C21_sym", "C40_abs", "C41_abs", "C42_abs", "C42_norm",
            "spectral_flatness", "zero_crossing_rate",
        ]


def extract_features(
    iq: np.ndarray,
    sample_rate: float = 1.0,
    snr_db: Optional[float] = None,
    fft_size: int = 512,
) -> FeatureVector:
    """Extract all modulation-discriminating features from a complex IQ array.

    Args:
        iq          : Complex IQ data (1D array, complex64 or complex128)
        sample_rate : Sample rate in Hz (needed for inst_freq in Hz)
        snr_db      : Optional SNR estimate to attach as metadata
        fft_size    : FFT size for spectral flatness computation

    Returns:
        FeatureVector with all features filled in
    """
    x = np.asarray(iq, dtype=np.complex128)
    n = len(x)
    if n < 8:
        return FeatureVector(num_samples=n, sample_rate=sample_rate)

    fv = FeatureVector(num_samples=n, sample_rate=sample_rate, snr_db_estimate=snr_db)

    # ── Amplitude ────────────────────────────────────────────────────────────
    amp = np.abs(x)
    amp_mean = float(np.mean(amp))
    amp_std  = float(np.std(amp))
    amp_eps  = max(amp_mean, 1e-12)

    fv.amplitude_variance_norm = amp_std / amp_eps
    if amp_std < 1e-9:
        fv.kurtosis_amplitude = -1.2  # Degenerate constant envelope
    else:
        k_amp = float(scipy_kurtosis(amp, fisher=True))
        fv.kurtosis_amplitude = -1.2 if (math.isnan(k_amp) or math.isinf(k_amp)) else k_amp

    power = amp ** 2
    mean_power = float(np.mean(power))
    peak_power = float(np.max(power))
    fv.papr_db = float(10.0 * np.log10(peak_power / max(mean_power, 1e-20)))

    # ── Phase ────────────────────────────────────────────────────────────────
    raw_phase = np.angle(x)
    fv.phase_std = float(np.std(raw_phase))
    phase_unwrapped = np.unwrap(raw_phase)
    # Detrend to remove carrier frequency ramp before computing phase variance
    if len(phase_unwrapped) > 1:
        t = np.linspace(-0.5, 0.5, len(phase_unwrapped))
        slope = np.sum(t * phase_unwrapped) / (np.sum(t ** 2) + 1e-12)
        phase_detrended = phase_unwrapped - slope * t
    else:
        phase_detrended = phase_unwrapped
    fv.phase_variance = float(np.var(phase_detrended))
    if np.std(phase_detrended) < 1e-9:
        fv.kurtosis_phase = -1.2
    else:
        k_ph = float(scipy_kurtosis(phase_detrended, fisher=True))
        fv.kurtosis_phase = -1.2 if (math.isnan(k_ph) or math.isinf(k_ph)) else k_ph

    # ── Instantaneous frequency ──────────────────────────────────────────────
    inst_freq_raw = np.diff(phase_unwrapped)   # radians / sample
    inst_freq_hz  = inst_freq_raw * sample_rate / (2.0 * np.pi)
    fv.instantaneous_freq_mean = float(np.mean(np.abs(inst_freq_hz)))
    fv.instantaneous_freq_std  = float(np.std(inst_freq_hz))

    inst_freq_cycles = inst_freq_raw / (2.0 * np.pi)  # cycles / sample (dimensionless)
    fv.instantaneous_freq_mean_norm = float(np.mean(np.abs(inst_freq_cycles)))
    fv.instantaneous_freq_std_norm  = float(np.std(inst_freq_cycles))

    # ── Physical signal parameters ──────────────────────────────────────────
    fv.raw_power = float(np.mean(np.abs(x) ** 2))
    fv.dc_offset_real = float(np.mean(x.real))
    fv.dc_offset_imag = float(np.mean(x.imag))

    # ── Higher-order cumulants (HOC) ─────────────────────────────────────────
    # Moments
    M20  = float(np.mean(np.abs(x) ** 2))               # E[|x|^2]
    M21  = complex(np.mean(x ** 2))                     # E[x^2]
    M40  = complex(np.mean(x ** 4))                     # E[x^4]
    M41  = complex(np.mean((x ** 3) * np.conj(x)))      # E[x^3 * x*]
    M42  = float(np.mean(np.abs(x) ** 4))               # E[|x|^4]

    fv.C20 = M20

    # C(2,1) = E[x^2]
    fv.C21_abs = float(abs(M21))

    # Normalized phase symmetry: |C21| / C20 (BPSK~1, QPSK/QAM~0)
    fv.C21_sym = fv.C21_abs / max(M20, 1e-12)

    # C(4,0) = E[x^4] - 3*(E[x^2])^2
    C40 = M40 - 3.0 * (M21 ** 2)
    fv.C40_abs = float(abs(C40))

    # C(4,1) = E[x^3*x*] - 3*E[x^2]*E[|x|^2]
    C41 = M41 - 3.0 * M21 * M20
    fv.C41_abs = float(abs(C41))

    # C(4,2) = E[|x|^4] - |E[x^2]|^2 - 2*(E[|x|^2])^2
    C42 = M42 - abs(M21) ** 2 - 2.0 * (M20 ** 2)
    fv.C42_abs = float(abs(C42))

    # Normalized: |C42| / C20^2  (BPSK~2, QPSK~1, 16QAM~0.68, FSK~0)
    fv.C42_norm = fv.C42_abs / max(M20 ** 2, 1e-20)

    # ── Spectral flatness (Wiener entropy) ───────────────────────────────────
    n_fft = min(fft_size, n)
    spectrum = np.abs(np.fft.fft(x[:n_fft])) ** 2 + 1e-20
    geom_mean = float(np.exp(np.mean(np.log(spectrum))))
    arith_mean = float(np.mean(spectrum))
    fv.spectral_flatness = geom_mean / max(arith_mean, 1e-20)

    # ── Zero crossing rate ────────────────────────────────────────────────────
    i_sign = np.sign(x.real)
    q_sign = np.sign(x.imag)
    i_crossings = float(np.sum(i_sign[:-1] != i_sign[1:]))
    q_crossings = float(np.sum(q_sign[:-1] != q_sign[1:]))
    fv.zero_crossing_rate = (i_crossings + q_crossings) / (2.0 * max(n - 1, 1))

    return fv


def extract_features_from_iq_array(
    iq_real: np.ndarray,
    iq_imag: np.ndarray,
    sample_rate: float = 1.0,
    snr_db: Optional[float] = None,
) -> FeatureVector:
    """Convenience wrapper for separate I/Q arrays (RadioML format: shape (2, N))."""
    x = (iq_real + 1j * iq_imag).astype(np.complex128)
    return extract_features(x, sample_rate=sample_rate, snr_db=snr_db)
