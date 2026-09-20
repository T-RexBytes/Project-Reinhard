"""
app/characterization/symbol_rate.py
-------------------------------------
Symbol rate estimation from complex IQ.

Methods implemented:
  1. Squared-signal PSD  : |IQ|^2 → FFT → find spectral peak at 2*f_sym
     (classic for PSK/QAM — squaring removes carrier, exposes baud rate)
  2. Autocorrelation     : find period of |IQ|^2 autocorrelation
     (backup method, slower but works on shorter segments)
  3. Raised-cosine hint  : use bandwidth estimate to bound symbol rate

Works best on:
  - BPSK / QPSK / 8PSK  (squaring method is optimal)
  - QAM16               (squaring method works well)
  - FSK                 (less reliable — use bandwidth method instead)

Validated against: CommSignal2 (SigMF: QPSK, fs=25 MHz)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import welch, find_peaks


@dataclass
class SymbolRateResult:
    """Symbol rate estimation result."""
    symbol_rate_hz: float
    confidence: float           # 0–1 heuristic
    method: str
    spectral_peak_hz: Optional[float] = None
    bandwidth_bound_hz: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol_rate_hz":      self.symbol_rate_hz,
            "confidence":          self.confidence,
            "method":              self.method,
            "spectral_peak_hz":    self.spectral_peak_hz,
            "bandwidth_bound_hz":  self.bandwidth_bound_hz,
            "notes":               self.notes,
        }


def _peak_interp(db: np.ndarray, peak_idx: int) -> float:
    """Parabolic interpolation of a spectral peak for sub-bin frequency accuracy."""
    idx = int(peak_idx)
    if 1 <= idx < len(db) - 1:
        y0, y1, y2 = float(db[idx - 1]), float(db[idx]), float(db[idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            offset = 0.5 * (y0 - y2) / denom
            return idx + offset
    return float(idx)


def estimate_symbol_rate_envelope(
    iq: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096,
    min_rate_hz: float = 1e3,
    max_rate_hz: Optional[float] = None,
) -> SymbolRateResult:
    """Estimate symbol rate from the PSD of |IQ|^2 (envelope).

    For Nyquist-shaped PSK/QAM the instantaneous power |x|^2 varies periodically
    at the symbol rate, so the envelope PSD has a discrete line at exactly Rsym.
    Critically, |x|^2 removes the carrier entirely, making this estimator
    essentially immune to CFO (unlike the complex-squared method, which can
    falsely lock onto the 2*fc line for real BPSK).
    """
    x = np.asarray(iq, dtype=np.complex128)
    n = len(x)
    if n < 64:
        return SymbolRateResult(symbol_rate_hz=0.0, confidence=0.0, method="envelope",
                                notes="Signal too short for envelope PSD")

    env = np.abs(x) ** 2
    env = env - np.mean(env)

    if n < fft_size:
        fft_n = max(8, 2 ** int(np.floor(np.log2(n))))
    else:
        fft_n = fft_size
    nperseg = min(fft_n, n)
    noverlap = max(0, nperseg // 2)

    freqs, psd = welch(env, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(np.abs(psd))

    max_hz = max_rate_hz or sample_rate / 2.0
    mask = (freqs > min_rate_hz) & (freqs <= max_hz)
    if not np.any(mask):
        return SymbolRateResult(symbol_rate_hz=0.0, confidence=0.0, method="envelope",
                                notes="Empty search range")

    search_psd = psd[mask]
    search_freq = freqs[mask]
    psd_db = 10.0 * np.log10(search_psd + 1e-20)
    noise_floor = np.percentile(psd_db, 20)
    prominence = max(3.0, (np.max(psd_db) - noise_floor) * 0.3)
    peaks, props = find_peaks(psd_db, prominence=prominence)

    if len(peaks) == 0:
        idx = int(np.argmax(search_psd))
        peak_hz = float(search_freq[idx])
        return SymbolRateResult(
            symbol_rate_hz=peak_hz, confidence=0.2, method="envelope",
            spectral_peak_hz=peak_hz, notes="No distinct envelope peak; using argmax",
        )

    best_pos = int(np.argmax(props["prominences"]))
    best = int(peaks[best_pos])
    prominence_val = float(props["prominences"][best_pos])
    # Sub-bin interpolation refines the strobe period (reduces drift over long bursts)
    peak_hz = float(search_freq[int(np.clip(round(_peak_interp(psd_db, best)), 0, len(search_freq) - 1))])
    confidence = float(np.clip(prominence_val / 20.0, 0.10, 0.90))
    return SymbolRateResult(
        symbol_rate_hz=peak_hz, confidence=confidence, method="envelope",
        spectral_peak_hz=peak_hz, notes=f"Envelope PSD line at {peak_hz/1e3:.1f} kHz",
    )


def estimate_symbol_rate_freq_waveform(
    iq: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096,
    min_rate_hz: float = 1e3,
    max_rate_hz: Optional[float] = None,
) -> SymbolRateResult:
    """Estimate symbol rate from the PSD of the instantaneous-frequency waveform.

    For (continuous-phase) FSK the instantaneous frequency is a rectangular
    waveform with period T, so its PSD has a discrete line at Rsym = 1/T.
    Constant-envelope FSK has a flat envelope (no |x|^2 line), making this the
    natural complement of the envelope method.
    """
    x = np.asarray(iq, dtype=np.complex128)
    n = len(x)
    if n < 128:
        return SymbolRateResult(symbol_rate_hz=0.0, confidence=0.0, method="freq_waveform",
                                notes="Signal too short for freq-waveform PSD")

    phase = np.unwrap(np.angle(x))
    freq = np.diff(phase) / (2.0 * np.pi)  # cycles/sample, piecewise-constant for FSK
    # Light smoothing: suppresses sample-to-sample phase noise, keeps the symbol clock
    if len(freq) >= 8:
        k = np.ones(5, dtype=np.float64) / 5.0
        freq = np.convolve(freq, k, mode="same")
    freq = freq - np.mean(freq)

    if n < fft_size:
        fft_n = max(8, 2 ** int(np.floor(np.log2(n))))
    else:
        fft_n = fft_size
    nperseg = min(fft_n, len(freq))
    noverlap = max(0, nperseg // 2)

    freqs, psd = welch(freq, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(np.abs(psd))

    max_hz = max_rate_hz or sample_rate / 2.0
    mask = (freqs > min_rate_hz) & (freqs <= max_hz)
    if not np.any(mask):
        return SymbolRateResult(symbol_rate_hz=0.0, confidence=0.0, method="freq_waveform",
                                notes="Empty search range")

    search_psd = psd[mask]
    search_freq = freqs[mask]
    psd_db = 10.0 * np.log10(search_psd + 1e-20)
    noise_floor = np.percentile(psd_db, 20)
    prominence = max(3.0, (np.max(psd_db) - noise_floor) * 0.3)
    peaks, props = find_peaks(psd_db, prominence=prominence)

    if len(peaks) == 0:
        idx = int(np.argmax(search_psd))
        peak_hz = float(search_freq[idx])
        return SymbolRateResult(
            symbol_rate_hz=peak_hz, confidence=0.2, method="freq_waveform",
            spectral_peak_hz=peak_hz, notes="No distinct freq-waveform peak; using argmax",
        )

    best_pos = int(np.argmax(props["prominences"]))
    best = int(peaks[best_pos])
    prominence_val = float(props["prominences"][best_pos])
    peak_hz = float(search_freq[int(np.clip(round(_peak_interp(psd_db, best)), 0, len(search_freq) - 1))])
    confidence = float(np.clip(prominence_val / 20.0, 0.10, 0.90))
    return SymbolRateResult(
        symbol_rate_hz=peak_hz, confidence=confidence, method="freq_waveform",
        spectral_peak_hz=peak_hz, notes=f"Freq-waveform line at {peak_hz/1e3:.1f} kHz",
    )


def estimate_symbol_rate_squared_psd(
    iq: np.ndarray,
    sample_rate: float,
    fft_size: int = 4096,
    min_rate_hz: float = 1e3,
    max_rate_hz: Optional[float] = None,
) -> SymbolRateResult:
    """Estimate symbol rate via PSD of the squared signal.

    For PSK/QAM: x^2 removes the carrier phase, exposing a spectral line
    at 2 * symbol_rate. We look for the dominant spectral peak in |IQ|^2
    and infer symbol_rate = peak_freq / 2.

    Args:
        iq          : Complex IQ (1D)
        sample_rate : Sampling rate in Hz
        fft_size    : FFT size
        min_rate_hz : Minimum plausible symbol rate
        max_rate_hz : Maximum plausible symbol rate (default: sample_rate / 2)

    Returns:
        SymbolRateResult
    """
    x = np.asarray(iq, dtype=np.complex128)
    n = len(x)
    if n < fft_size:
        fft_size = max(8, 2 ** int(np.floor(np.log2(n))))

    # Squared signal removes carrier, exposes baud tones
    x_sq = x ** 2

    nperseg = min(fft_size, n)
    noverlap = max(0, nperseg // 2)
    freqs, psd = welch(
        x_sq,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
    )
    freqs = np.fft.fftshift(freqs)
    psd   = np.fft.fftshift(np.abs(psd))

    # Search only positive frequencies
    max_hz = max_rate_hz or sample_rate / 2.0
    pos_mask = (freqs > min_rate_hz * 2) & (freqs <= max_hz * 2)
    if not np.any(pos_mask):
        return SymbolRateResult(
            symbol_rate_hz=0.0, confidence=0.0, method="squared_psd",
            notes="No valid frequency range found",
        )

    search_psd  = psd[pos_mask]
    search_freq = freqs[pos_mask]

    # Find peaks
    psd_db = 10.0 * np.log10(search_psd + 1e-20)
    noise_floor = np.percentile(psd_db, 20)
    prominence  = max(3.0, (np.max(psd_db) - noise_floor) * 0.3)

    peaks, props = find_peaks(psd_db, prominence=prominence)

    if len(peaks) == 0:
        # Fallback: just take the max
        peak_idx = int(np.argmax(search_psd))
        peak_freq = float(search_freq[peak_idx])
        confidence = 0.25
        notes = "No distinct peak found; using argmax"
    else:
        # Take the most prominent peak
        best = peaks[np.argmax(props["prominences"])]
        peak_freq   = float(search_freq[best])
        prominence_val = float(props["prominences"][np.argmax(props["prominences"])])
        # Confidence based on prominence vs noise
        confidence = float(np.clip(prominence_val / 20.0, 0.1, 0.95))
        notes = f"Peak prominence={prominence_val:.1f} dB"

    symbol_rate = abs(peak_freq) / 2.0

    return SymbolRateResult(
        symbol_rate_hz=symbol_rate,
        confidence=confidence,
        method="squared_psd",
        spectral_peak_hz=peak_freq,
        notes=notes,
    )


def estimate_symbol_rate_autocorr(
    iq: np.ndarray,
    sample_rate: float,
    max_lag_samples: int = 2048,
    min_rate_hz: float = 1e3,
) -> SymbolRateResult:
    """Estimate symbol rate via autocorrelation of |IQ|^2.

    The power envelope of a baseband signal is periodic at the symbol rate.
    We find the first dominant lag in the autocorrelation of |IQ|^2.

    Args:
        iq               : Complex IQ (1D)
        sample_rate      : Sampling rate in Hz
        max_lag_samples  : Maximum lag to search
        min_rate_hz      : Minimum plausible symbol rate

    Returns:
        SymbolRateResult
    """
    x = np.asarray(iq, dtype=np.complex128)
    power = np.abs(x) ** 2
    power -= np.mean(power)  # remove DC

    n = len(power)
    max_lag = min(max_lag_samples, n // 2)

    # Compute autocorrelation via FFT (fast)
    fft_size = 2 ** int(np.ceil(np.log2(2 * n)))
    P = np.fft.fft(power, n=fft_size)
    acf = np.real(np.fft.ifft(P * np.conj(P)))[:max_lag]
    if acf[0] > 0:
        acf /= acf[0]  # normalize

    # Find first strong peak after lag=0 (skip lag < min_samples)
    min_lag = max(2, int(sample_rate / (min_rate_hz * 10)))
    search_acf = acf[min_lag:]

    peaks, props = find_peaks(search_acf, prominence=0.05, distance=min_lag)

    if len(peaks) == 0:
        return SymbolRateResult(
            symbol_rate_hz=0.0, confidence=0.0, method="autocorr",
            notes="No autocorrelation peak found",
        )

    best = peaks[np.argmax(props["prominences"])]
    lag_samples = best + min_lag
    symbol_rate = sample_rate / lag_samples

    confidence = float(np.clip(props["prominences"][np.argmax(props["prominences"])], 0.0, 0.9))

    return SymbolRateResult(
        symbol_rate_hz=symbol_rate,
        confidence=confidence,
        method="autocorr",
        notes=f"Period={lag_samples} samples",
    )


def estimate_symbol_rate(
    iq: np.ndarray,
    sample_rate: float,
    bandwidth_hz: Optional[float] = None,
    fft_size: int = 4096,
) -> SymbolRateResult:
    """Best-effort symbol rate estimation combining multiple methods.

    Strategy:
      1. Try squared_psd (best for PSK/QAM)
      2. Try autocorr as backup
      3. Use bandwidth as a sanity bound (symbol_rate <= bandwidth)
      4. Return the higher-confidence result

    Args:
        iq           : Complex IQ (1D)
        sample_rate  : Sampling rate in Hz
        bandwidth_hz : Optional 3dB bandwidth hint to bound search
        fft_size     : FFT size for PSD method

    Returns:
        SymbolRateResult (best estimate)
    """
    # max_rate floors at sample_rate/4 so it can never exclude a physically
    # plausible Rsym (even when the measured 3dB BW is tiny) while still
    # bounding the search against wideband out-of-band energy.
    max_rate = max(bandwidth_hz * 2.0, sample_rate / 4.0) if bandwidth_hz else None

    r_env = estimate_symbol_rate_envelope(iq, sample_rate, fft_size=fft_size, max_rate_hz=max_rate)
    r3 = estimate_symbol_rate_freq_waveform(iq, sample_rate, fft_size=fft_size, max_rate_hz=max_rate)
    r1 = estimate_symbol_rate_squared_psd(iq, sample_rate, fft_size=fft_size, max_rate_hz=max_rate)
    r2 = estimate_symbol_rate_autocorr(iq, sample_rate)

    # The envelope / freq-waveform lines sit exactly at Rsym and are CFO-immune,
    # so they are never vetoed by the bandwidth sanity bound. The messy
    # squared/autocorr methods are filtered so they cannot win with absurd rates.
    results = [r_env, r3]
    if bandwidth_hz:
        for r in (r1, r2):
            if r.symbol_rate_hz <= bandwidth_hz * 2.0 or r.symbol_rate_hz == 0:
                results.append(r)
    else:
        results += [r1, r2]

    if not results:
        return SymbolRateResult(
            symbol_rate_hz=bandwidth_hz or 0.0,
            confidence=0.1,
            method="bandwidth_fallback",
            bandwidth_bound_hz=bandwidth_hz,
            notes="Both spectral methods failed; using bandwidth as proxy",
        )

    # Arbitration: prefer a direct-Rsym method (envelope / freq-waveform are
    # CFO-immune and physically exact), then fall back to confidence alone.
    def _score(r: SymbolRateResult) -> float:
        prefix = {"envelope": 1.10, "freq_waveform": 1.10, "squared_psd": 1.0, "autocorr": 0.90}.get(r.method, 1.0)
        return r.confidence * prefix

    best = max(results, key=_score)
    best.bandwidth_bound_hz = bandwidth_hz

    return best
