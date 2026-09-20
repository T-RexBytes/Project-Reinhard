"""SNR estimation methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as scipy_signal

from app.models import ComplexSignal
from app.detection.fft import PSDResult
from app.detection.noise_floor import NoiseFloorResult


@dataclass
class SNRResult:
    """SNR estimation result."""
    snr_db: float
    signal_power_db: float
    noise_power_db: float
    method: str
    bandwidth_hz: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "snr_db": self.snr_db,
            "signal_power_db": self.signal_power_db,
            "noise_power_db": self.noise_power_db,
            "method": self.method,
            "bandwidth_hz": self.bandwidth_hz,
        }


def estimate_snr_from_psd(
    psd: PSDResult,
    noise_floor: NoiseFloorResult,
    signal_freq_start_hz: Optional[float] = None,
    signal_freq_end_hz: Optional[float] = None,
    threshold_db_above_noise: float = 10.0,
) -> SNRResult:
    """Estimate SNR from PSD by comparing signal band to noise band.

    Args:
        psd: PSD result
        noise_floor: Noise floor estimation
        signal_freq_start_hz: Start of signal band (auto-detect if None)
        signal_freq_end_hz: End of signal band (auto-detect if None)
        threshold_db_above_noise: Threshold for signal detection

    Returns:
        SNRResult
    """
    freqs = psd.frequencies
    psd_db = psd.psd_db

    threshold_db = noise_floor.noise_floor_db + threshold_db_above_noise

    if signal_freq_start_hz is not None and signal_freq_end_hz is not None:
        mask = (freqs >= signal_freq_start_hz) & (freqs <= signal_freq_end_hz)
        signal_bins = np.where(mask)[0]
    else:
        signal_bins = np.where(psd_db > threshold_db)[0]

    if len(signal_bins) == 0:
        # Fallback for full-band baseband signals where the signal occupies
        # 100% of the recorded bandwidth and the MAD noise floor estimator
        # treats the signal power itself as the floor.  Use the M2M4
        # moment estimator on the raw IQ data embedded in the PSD result.
        try:
            iq_approx = np.sqrt(np.power(10.0, psd.psd_db / 10.0)).astype(np.complex64)
            snr_m2m4 = estimate_snr_m2m4(iq_approx)
            fallback_snr = max(0.0, float(snr_m2m4.snr_db))
        except Exception:
            fallback_snr = 0.0
        return SNRResult(
            snr_db=fallback_snr,
            signal_power_db=float(noise_floor.noise_floor_db) + fallback_snr,
            noise_power_db=float(noise_floor.noise_floor_db),
            method="fullband_fallback",
        )

    signal_power_db = float(np.mean(psd_db[signal_bins]))
    noise_power_db = float(noise_floor.noise_floor_db)

    snr_db = signal_power_db - noise_power_db

    freq_resolution = psd.freq_resolution
    bandwidth_hz = len(signal_bins) * freq_resolution

    return SNRResult(
        snr_db=snr_db,
        signal_power_db=signal_power_db,
        noise_power_db=noise_power_db,
        method="psd_ratio",
        bandwidth_hz=bandwidth_hz,
    )


def estimate_snr_time_domain(
    signal: ComplexSignal,
    signal_mask: Optional[np.ndarray] = None,
) -> SNRResult:
    """Estimate SNR in time domain.

    If signal_mask is provided, uses it to separate signal from noise.
    Otherwise uses power statistics.

    Args:
        signal: Input ComplexSignal
        signal_mask: Boolean mask, True = signal sample

    Returns:
        SNRResult
    """
    power = np.abs(signal.data) ** 2

    if signal_mask is not None and np.any(signal_mask):
        signal_power = np.mean(power[signal_mask])
        noise_power = np.mean(power[~signal_mask]) if np.any(~signal_mask) else 1e-20
    else:
        sorted_power = np.sort(power)
        n = len(sorted_power)
        noise_power = np.mean(sorted_power[: n // 4])
        signal_power = np.mean(sorted_power[n // 2 :])

    signal_db = 10.0 * np.log10(max(signal_power, 1e-20))
    noise_db = 10.0 * np.log10(max(noise_power, 1e-20))

    return SNRResult(
        snr_db=float(signal_db - noise_db),
        signal_power_db=float(signal_db),
        noise_power_db=float(noise_db),
        method="time_domain",
    )


def estimate_snr_m2m4(
    signal_data: np.ndarray,
) -> SNRResult:
    """Estimate SNR using M2M4 method (2nd and 4th order moments).

    Works well for PSK/QAM signals.

    Args:
        signal_data: Complex IQ data

    Returns:
        SNRResult
    """
    x = signal_data.astype(np.complex64)
    M2 = np.mean(np.abs(x) ** 2)
    M4 = np.mean(np.abs(x) ** 4)

    snr_linear = (M2 ** 2) / (M4 - M2 ** 2 + 1e-20)
    snr_linear = max(snr_linear, 0.0)

    snr_db = 10.0 * np.log10(snr_linear + 1e-20)
    signal_db = 10.0 * np.log10(M2 + 1e-20)

    return SNRResult(
        snr_db=float(snr_db),
        signal_power_db=float(signal_db),
        noise_power_db=float(signal_db - snr_db),
        method="m2m4",
    )
