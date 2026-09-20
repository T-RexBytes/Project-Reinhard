"""
app/demodulation/pulseshape.py
--------------------------------
Square-root raised-cosine (RRC) matched filtering and rolloff estimation.

Real-world digital links shape the transmit spectrum with an RRC pulse (or a
root-raised-cosine / raised-cosine equivalent). A correctly matched RRC filter
at the receiver maximizes symbol SNR and completely eliminates inter-symbol
interference (ISI) at the optimum strobe points. Skipping this step is why naive
"slice every Nth sample" demodulation produces 85-90% EVM on real IQ captures.

This module also provides a bandwidth-based rolloff estimator so the filter can
be adapted automatically from physical measurements (BW_3dB / symbol_rate).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import fftconvolve


def estimate_rolloff(
    bandwidth_hz: Optional[float],
    symbol_rate_hz: Optional[float],
    default: float = 0.35,
) -> float:
    """Estimate the Nyquist rolloff factor alpha from measured bandwidth.

    For linear modulations (PSK/QAM) the occupied bandwidth is
    BW = (1 + alpha) * Rsym, so alpha = BW / Rsym - 1.

    Returns a value clamped to the physically sensible [0.05, 0.6] range,
    falling back to `default` when insufficient information is available.
    """
    if (
        bandwidth_hz
        and symbol_rate_hz
        and bandwidth_hz > 0
        and symbol_rate_hz > 0
        and bandwidth_hz > symbol_rate_hz
    ):
        alpha = bandwidth_hz / symbol_rate_hz - 1.0
        return float(np.clip(alpha, 0.05, 0.6))
    return float(default)


def design_rrc_taps(
    samples_per_symbol: float,
    alpha: float = 0.35,
    span_symbols: float = 8.0,
) -> np.ndarray:
    """Design a symmetric square-root raised-cosine FIR filter.

    Args:
        samples_per_symbol: Oversampling factor (samples per symbol).
        alpha: Rolloff factor in [0.05, 0.6].
        span_symbols: Filter span in symbol periods (non-integer allowed).

    Returns:
        Real, symmetric FIR taps normalized to unit L2 energy (true matched filter).
    """
    sps = float(samples_per_symbol)
    if sps < 2.0:
        return np.ones(1, dtype=np.float64)

    alpha = float(np.clip(alpha, 0.05, 0.6))
    span = max(2.0, float(span_symbols))

    num_taps = int(round(span * sps))
    if num_taps % 2 == 0:
        num_taps += 1

    n = (num_taps - 1) // 2
    tau = np.arange(-n, n + 1, dtype=np.float64) / sps  # normalized by symbol period

    h = np.empty_like(tau)
    a = alpha
    for i, t in enumerate(tau):
        if abs(t) < 1e-12:
            h[i] = (1.0 - a) + 4.0 * a / np.pi
        elif abs(abs(t) - 1.0 / (4.0 * a)) < 1e-6:
            h[i] = (a / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * a))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * a))
            )
        else:
            denom = 1.0 - (4.0 * a * t) ** 2
            numer = np.cos((1.0 + a) * np.pi * t) + np.sin((1.0 - a) * np.pi * t) / (4.0 * a * t)
            h[i] = 4.0 * a * numer / (np.pi * denom)

    # Unit L2 normalization => matched filter is energy-neutral.
    norm = float(np.sqrt(np.sum(h ** 2)))
    if norm > 1e-12:
        h = h / norm
    return h.astype(np.float64)


def apply_matched_filter(
    iq: np.ndarray,
    samples_per_symbol: float,
    alpha: float = 0.35,
    span_symbols: float = 8.0,
) -> np.ndarray:
    """Zero-phase matched filtering of a complex IQ burst with an RRC filter.

    The RRC kernel is symmetric, so `fftconvolve(..., mode="same")` yields a
    zero-phase (linear/symmetric) filter with no group delay distortion.

    If the signal is too short relative to the requested span, the span is
    shrunk so the filter fits inside the burst (keeps 1/4 of samples as steady
    state, which is relevant for ultra-short RadioML bursts of 128 samples).

    Returns:
        Matched-filtered complex IQ (same dtype/shape as input).
    """
    x = np.asarray(iq)
    sps = float(samples_per_symbol)
    if sps < 2.0 or len(x) < int(sps * 4):
        return x.astype(np.complex64)

    # Shrink span on short bursts: keep at least 1/4 of the burst as steady state.
    available_span = (len(x) // 4) / sps
    eff_span = min(float(span_symbols), max(2.0, float(available_span)))

    taps = design_rrc_taps(sps, alpha, span_symbols=eff_span)
    taps_c = taps.astype(np.complex64)
    return fftconvolve(x, taps_c, mode="same").astype(np.complex64)