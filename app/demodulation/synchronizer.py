"""
app/demodulation/synchronizer.py
---------------------------------
DSP synchronization routines for digital modulations.

Implements the receiver sync chain that converts an oversampled, CFO- and
timing-offset-impaired IQ burst into clean symbol samples:

  - Modulation-aware coarse carrier frequency offset (CFO) estimation.
  - Cubic Farrow interpolation (sub-sample-accurate resampling).
  - Symbol timing recovery: global max-energy strobe + continuous Gardner-Farrow
    tracking loop (with a max-symbol cap so per-burst cost stays bounded even on
    40k-sample captures).
  - Decision-directed phase recovery (per-symbol PLL) for fine residual CFO/phase.
  - Phase-ambiguity resolution against the reference constellation.

Scaling notes:
  - Heavy array operations are NumPy-vectorized.
  - The only Python loops are over the *recovered symbols* (at most max_symbols),
    never over the full oversampled sample stream, which keeps the pipeline fast
    across 25 MHz data0 frames, 61.44 MHz Zenodo frames, and 128-sample RadioML
    bursts alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from app.characterization.freq_offset import estimate_freq_offset


# ── Interpolation & sub-sample resampling ───────────────────────────────────

def farrow3_interp(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """3rd-order (cubic) Farrow interpolation at arbitrary real sample positions.

    For each position p, samples at floor(p)-1 .. floor(p)+2 are combined with
    the fractional part mu to form a continuously-differentiable interpolant.

    Args:
        x:   1D complex sample array.
        pos: 1D array of float sample positions.

    Returns:
        Interpolated complex samples (same length as pos).
    """
    x = np.asarray(x)
    n = len(x)
    pos = np.asarray(pos, dtype=np.float64)
    i0 = np.floor(pos).astype(np.int64)
    mu = pos - i0

    im1 = np.clip(i0 - 1, 0, n - 1)
    i00 = np.clip(i0, 0, n - 1)
    ip1 = np.clip(i0 + 1, 0, n - 1)
    ip2 = np.clip(i0 + 2, 0, n - 1)

    xm1 = x[im1]
    x0 = x[i00]
    x1 = x[ip1]
    x2 = x[ip2]

    a3 = -0.5 * xm1 + 1.5 * x0 - 1.5 * x1 + 0.5 * x2
    a2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2
    a1 = -0.5 * xm1 + 0.5 * x1
    a0 = x0

    return a0 + mu * (a1 + mu * (a2 + mu * a3))


def farrow_strobe(
    iq: np.ndarray,
    samples_per_symbol: float,
    mu: float = 0.0,
    max_symbols: Optional[int] = None,
) -> np.ndarray:
    """Downsample an oversampled IQ to 1 sample/symbol at position mu + k*sps."""
    x = np.asarray(iq)
    sps = float(samples_per_symbol)
    if sps <= 1.0:
        return x.astype(np.complex64)
    n = len(x)
    num_symbols = int(n / sps)
    if num_symbols <= 0:
        return x[:1].astype(np.complex64)
    if max_symbols is not None:
        num_symbols = min(num_symbols, int(max_symbols))
    pos = mu + np.arange(num_symbols, dtype=np.float64) * sps
    syms = farrow3_interp(x, pos)
    return syms.astype(np.complex64)


# ── Coarse carrier frequency offset ─────────────────────────────────────────

def _modulation_hint(modulation: Optional[str]) -> Optional[str]:
    hint = (modulation or "").upper()
    # Order matters: "AM" is a substring of "QAM", so QAM/PSK must be matched
    # before the generic AM/FM/CW branch or 16QAM gets misrouted to psd_peak.
    if "FSK" in hint:
        return "FSK"
    if "BPSK" in hint:
        return "BPSK"
    if "QAM" in hint:
        return "QAM"
    if "QPSK" in hint or "PSK" in hint:
        return "QPSK"
    # Constant-envelope / analog: psd_peak is reliable here.
    if "AM" in hint or "FM" in hint or "CW" in hint:
        return "FSK"
    return None


def estimate_coarse_cfo(
    iq: np.ndarray,
    sample_rate: float,
    modulation: Optional[str] = None,
) -> float:
    """Modulation-aware coarse CFO estimate (Hz). Returns 0.0 if unreliable."""
    hint = _modulation_hint(modulation)
    res = estimate_freq_offset(iq, sample_rate, mod_hint=hint)
    return float(res.offset_hz) if res.confidence > 0.4 else 0.0


def derotate_carrier(iq: np.ndarray, freq_offset_hz: float, sample_rate: float) -> np.ndarray:
    """Derotate IQ signal by a known or estimated carrier frequency offset (CFO)."""
    if abs(freq_offset_hz) < 1e-6 or sample_rate <= 0:
        return iq.copy()
    n = len(iq)
    t = np.arange(n) / sample_rate
    phasor = np.exp(-1j * 2.0 * np.pi * freq_offset_hz * t, dtype=np.complex64)
    return (iq * phasor).astype(np.complex64)


# ── Symbol timing recovery ──────────────────────────────────────────────────

@dataclass
class TimingRecoveryResult:
    """Result of symbol timing recovery."""
    symbols: np.ndarray
    timing_offset: float          # fractional-sample strobe offset within [0, 1)
    samples_per_symbol: float     # refined oversampling factor used
    method: str                   # 'gardner_farrow' | 'max_energy' | 'identity'
    num_symbols: int

    def to_dict(self) -> dict:
        return {
            "timing_offset_samples": round(float(self.timing_offset), 4),
            "samples_per_symbol": round(float(self.samples_per_symbol), 4),
            "method": self.method,
            "num_symbols": int(self.num_symbols),
        }


def max_energy_timing_offset(
    iq: np.ndarray,
    samples_per_symbol: float,
    searches: int = 64,
) -> float:
    """Blind timing offset search maximizing mean symbol energy.

    For Nyquist (RRC/RRC-matched) signaling the symbol-energy arc E|s(t)|^2 is
    maximized at the optimum strobe instants, making this a robust coarse timing
    estimator that works without any data decisions.
    """
    x = np.asarray(iq)
    sps = float(samples_per_symbol)
    n = len(x)
    if sps <= 1.0 or n < int(sps * 4):
        return 0.0
    num_symbols = max(8, min(int(n / sps), 2048))
    mus = np.arange(searches, dtype=np.float64) / searches
    energies = np.empty(searches, dtype=np.float64)
    for k, mu in enumerate(mus):
        s = farrow3_interp(x, mu + np.arange(num_symbols, dtype=np.float64) * sps)
        energies[k] = float(np.mean(np.abs(s) ** 2))
    return float(mus[int(np.argmax(energies))])


def rectify_sps_by_energy(
    iq: np.ndarray,
    samples_per_symbol: float,
    mu: float,
    candidates: int = 21,
) -> float:
    """Refine the samples-per-symbol estimate by maximizing strobed symbol energy.

    Symbol-rate estimators (4th-power/autocorr) are accurate to a fraction of a
    percent, but the residual error accumulates into timing drift over long
    bursts. Maximizing mean|s|^2 over a small sps grid absorbs most of that drift.
    """
    x = np.asarray(iq)
    sps = float(samples_per_symbol)
    n = len(x)
    if sps < 1.5 or n < int(sps * 4):
        return sps
    span = 0.012
    grid = np.linspace(sps * (1.0 - span), sps * (1.0 + span), candidates, dtype=np.float64)
    num_symbols = max(16, min(int(n / sps), 2048))
    energies = np.empty(candidates, dtype=np.float64)
    for k, cand in enumerate(grid):
        s = farrow3_interp(x, mu + np.arange(num_symbols, dtype=np.float64) * cand)
        energies[k] = float(np.mean(np.abs(s) ** 2))
    return float(grid[int(np.argmax(energies))])


def gardner_farrow_timing(
    iq: np.ndarray,
    samples_per_symbol: float,
    initial_mu: float = 0.0,
    ted_gain: float = 0.02,
    max_symbols: int = 4096,
) -> tuple[np.ndarray, float, float]:
    """Continuous Gardner timing-error tracking with cubic Farrow interpolation.

    Works in a decimated ~2-samples-per-symbol domain. The strobe advances by
    the (refined) symbol period and the Gardner TED perturbs the fractional
    interpolation offset, absorbing residual timing offset and jitter.

    Returns:
        (symbols, final_mu, used_sps)
    """
    x = np.asarray(iq)
    n = len(x)
    sps = float(samples_per_symbol)
    if sps < 3.0 or n < int(sps * 4):
        syms = farrow_strobe(x, sps, initial_mu, max_symbols=max_symbols)
        return syms, initial_mu, sps

    d = max(1, int(round(sps / 2.0)))
    xd = np.ascontiguousarray(x[::d])
    sps2 = sps / d

    mu = float(initial_mu) / float(d)
    pos = mu
    half = sps2 * 0.5
    symbols = []
    n_available = int((len(xd) - 4) / sps2)
    n_target = min(max_symbols, max(0, n_available))
    if n_target <= 0:
        return farrow_strobe(x, sps, initial_mu, max_symbols=max_symbols), initial_mu, sps

    for _ in range(n_target):
        p_even = pos
        p_mid = pos + half
        p_next = pos + sps2

        s_even = farrow3_interp(xd, np.array([p_even]))[0]
        s_mid = farrow3_interp(xd, np.array([p_mid]))[0]
        s_next = farrow3_interp(xd, np.array([p_next]))[0]

        # Gardner TED: Re{ (s'_{n+1} - s'_n) * conj(s'_{n+1/2}) }
        ted = float(((s_next - s_even) * np.conj(s_mid)).real)
        ted = float(np.clip(ted, -1.0, 1.0))
        mu += ted_gain * ted
        pos = p_even + sps2
        symbols.append(s_even)

    return (
        np.asarray(symbols, dtype=np.complex64),
        float((mu % 1.0) * float(d)),
        sps,
    )


def recover_symbol_timing(
    iq: np.ndarray,
    samples_per_symbol: float,
    max_symbols: int = 4096,
) -> TimingRecoveryResult:
    """End-to-end symbol timing recovery with graceful degradation.

    Strategy:
      1. sps <= 1 -> identity (signal is already symbol-sampled).
      2. Global max-energy fractional offset search.
      3. sps refinement by energy maximization (absorbs symbol-rate drift).
      4. Continuous Gardner-Farrow tracking if the signal is long enough;
         otherwise a single-shot farrow strobe at the optimal offset.
    """
    x = np.asarray(iq)
    sps = float(samples_per_symbol)
    n = len(x)
    if sps <= 1.0:
        return TimingRecoveryResult(
            symbols=x.astype(np.complex64), timing_offset=0.0,
            samples_per_symbol=1.0, method="identity", num_symbols=len(x),
        )

    mu0 = max_energy_timing_offset(x, sps)
    sps_r = rectify_sps_by_energy(x, sps, mu0)

    if sps_r >= 3.0 and n >= int(sps_r * 4):
        syms, mu_f, sps_u = gardner_farrow_timing(
            x, sps_r, initial_mu=mu0, max_symbols=max_symbols
        )
        if len(syms) >= 4:
            return TimingRecoveryResult(
                symbols=syms, timing_offset=mu_f, samples_per_symbol=sps_u,
                method="gardner_farrow", num_symbols=len(syms),
            )

    syms = farrow_strobe(x, sps_r, mu0, max_symbols=max_symbols)
    return TimingRecoveryResult(
        symbols=syms, timing_offset=mu0, samples_per_symbol=sps_r,
        method="max_energy", num_symbols=len(syms),
    )


# ── Fine phase & frequency recovery ─────────────────────────────────────────

def _decide(sample: complex, mod_upper: str) -> complex:
    """In-loop lightweight decision (reference lattice kept local to avoid imports)."""
    i_val = float(sample.real)
    q_val = float(sample.imag)
    if "BPSK" in mod_upper:
        return complex(1.0 if i_val >= 0 else -1.0, 0.0)
    if "QAM" in mod_upper:
        # Unit-power 16QAM levels are +/-1/sqrt(10) (inner) and +/-3/sqrt(10)
        # (outer). Quantize in the sqrt(10)-scaled lattice: round(v*sqrt(10))
        # maps inner->+/-(1..2) and outer->+/-(2..3) correctly. The previous
        # scale sqrt(10)/3 collapsed the inner ring (0.316) onto 0, which
        # corrupted decisions and made decision-directed phase recovery diverge.
        scale = float(math.sqrt(10.0))
        dec_i = float(np.clip(round(i_val * scale), -3, 3)) / scale
        dec_q = float(np.clip(round(q_val * scale), -3, 3)) / scale
        return complex(dec_i, dec_q)
    # QPSK (default)
    s = 1.0 / math.sqrt(2.0)
    dec_i = s if i_val >= 0 else -s
    dec_q = s if q_val >= 0 else -s
    return complex(dec_i, dec_q)


def dd_phase_recovery(
    symbols: np.ndarray,
    modulation: str = "QPSK",
    loop_bw: float = 0.01,
    damping: float = 0.707,
) -> Tuple[np.ndarray, np.ndarray]:
    """2nd-order decision-directed phase/frequency tracking loop over symbols.

    Operates at 1 sample per symbol (after timing recovery), so the Python loop
    is bounded by the symbol count (typically a few hundred to a few thousand).

    Returns:
        (phase_synced_symbols, phase_history)
    """
    syms = np.asarray(symbols, dtype=np.complex64)
    n = len(syms)
    if n == 0:
        return syms, np.array([], dtype=np.float32)

    denom = 1.0 + 2.0 * damping * loop_bw + loop_bw * loop_bw
    alpha = (4.0 * damping * loop_bw) / denom
    beta = (4.0 * loop_bw * loop_bw) / denom

    out = np.empty(n, dtype=np.complex64)
    phase_history = np.empty(n, dtype=np.float32)
    mod_upper = modulation.upper()

    # Unit-power normalization for stable decisions
    pwr = float(np.mean(np.abs(syms) ** 2))
    gain = 1.0 / math.sqrt(max(pwr, 1e-12))
    x = syms * gain

    phase = 0.0
    freq = 0.0

    for k in range(n):
        sample = x[k] * np.exp(-1j * phase)
        out[k] = sample
        phase_history[k] = phase

        dec = _decide(sample, mod_upper)
        # Decision-directed phase error: Im{ y * conj(d) }
        error = float(np.imag(sample * np.conj(dec)))
        error = float(np.clip(error, -0.8, 0.8))

        freq += beta * error
        phase += freq + alpha * error

        if phase > math.pi:
            phase -= 2.0 * math.pi
        elif phase < -math.pi:
            phase += 2.0 * math.pi

    return out, phase_history


def resolve_phase_ambiguity(
    symbols: np.ndarray,
    modulation: str,
    ref_constellation: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Remove the M-fold rotational ambiguity left by decision-directed recovery.

    BPSK: 2-fold, QPSK/16QAM: 4-fold. Every candidate rotation is applied and
    the one minimizing RMS distance to the reference lattice is kept.

    Returns:
        (ambiguity_resolved_symbols, applied_rotation_rad)
    """
    syms = np.asarray(symbols, dtype=np.complex64)
    if len(syms) == 0 or len(ref_constellation) == 0:
        return syms, 0.0

    mod_upper = modulation.upper()
    m_fold = 2 if "BPSK" in mod_upper else 4

    angles = np.linspace(0.0, 2.0 * np.pi, int(m_fold), endpoint=False, dtype=np.float64)
    rot = np.exp(1j * angles)

    # Normalize power so distances are directly comparable
    pwr = float(np.mean(np.abs(syms) ** 2))
    s_norm = syms * (1.0 / math.sqrt(max(pwr, 1e-12)))

    best = None
    best_rms = np.inf
    for r in rot:
        cand = s_norm * r
        d = np.abs(cand[:, None] - ref_constellation[None, :])
        nearest = ref_constellation[np.argmin(d, axis=1)]
        err = cand - nearest
        rms = float(np.sqrt(np.mean(np.abs(err) ** 2)))
        if rms < best_rms:
            best_rms = rms
            best = float(np.angle(r))

    return (syms * np.exp(1j * best)).astype(np.complex64), best


# ── Backwards-compatible wrappers ───────────────────────────────────────────

def costas_loop(
    iq: np.ndarray,
    modulation: str = "QPSK",
    loop_bw: float = 0.01,
    damping: float = 0.707,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy-compatible Costas loop: decision-directed phase recovery.

    Keep signature stable for existing tests/API. Does NOT resolve the M-fold
    phase ambiguity (legacy behavior); use the full demodulation chain for that.
    """
    return dd_phase_recovery(iq, modulation, loop_bw=loop_bw, damping=damping)


def gardner_timing_recovery(
    iq: np.ndarray,
    sps_nominal: float,
    gain: float = 0.01,
) -> np.ndarray:
    """Legacy-compatible Gardner wrapper returning symbol samples only."""
    if sps_nominal < 3.0 or len(iq) < int(sps_nominal * 4):
        return farrow_strobe(iq, sps_nominal)
    syms, _, _ = gardner_farrow_timing(iq, sps_nominal, ted_gain=gain, max_symbols=8192)
    return syms


def strobe_symbols(
    iq: np.ndarray,
    samples_per_symbol: float,
    timing_offset: float = 0.0,
) -> np.ndarray:
    """Strobe symbols at optimal points using cubic interpolation (legacy API)."""
    return farrow_strobe(iq, samples_per_symbol, mu=timing_offset)