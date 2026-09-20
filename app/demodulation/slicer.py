"""
app/demodulation/slicer.py
--------------------------
Constellation slicing, EVM (Error Vector Magnitude), bit mapping, soft LLR
extraction, FSK tone slicing, and the full `demodulate_signal` orchestration.

The demodulation chain converts an unknown oversampled IQ burst into hard bits,
soft LLRs, and an honest EVM figure:

  classify (optional) -> coarse CFO -> matched filter -> timing recovery
  -> decision-directed phase recovery -> phase-ambiguity resolution
  -> slice -> EVM/LLR            (PSK/QAM)

  classify -> FSK tone slicing via differential phase demod     (FSK)

Short bursts (e.g., 128-sample RadioML frames) and Noise/CW inputs never
produce a garbage EVM: the result is explicitly *gated* with a machine-legible
`num_symbols` count and an honest `why` explanation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from app.models import ComplexSignal
from app.characterization.symbol_rate import estimate_symbol_rate
from app.modulation.classifier import classify_signal
from app.detection.fft import compute_psd
from app.detection.bandwidth import estimate_bandwidth
from app.demodulation.pulseshape import estimate_rolloff, apply_matched_filter
from app.demodulation.synchronizer import (
    estimate_coarse_cfo,
    derotate_carrier,
    recover_symbol_timing,
    dd_phase_recovery,
    resolve_phase_ambiguity,
)


# ── Reference Constellations (Normalized to Unit Average Energy) ────────────

BPSK_CONSTELLATION = np.array([-1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64)

# Gray order: index -> bits [b1 b0]
_QPSK_POINTS = np.array(
    [
        1.0 + 1.0j,
        1.0 - 1.0j,
        -1.0 - 1.0j,
        -1.0 + 1.0j,
    ],
    dtype=np.complex64,
) / math.sqrt(2.0)
_QPSK_BITS = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ],
    dtype=np.uint8,
)
QPSK_CONSTELLATION = _QPSK_POINTS

_grid_16 = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
# Per-axis reflected Gray code: value -> 2-bit gray
_axis_gray = {-3.0: 0, -1.0: 1, 1.0: 3, 3.0: 2}
_qam_pts = []
_qam_bits = []
for _r in _grid_16:
    for _i in _grid_16:
        _qam_pts.append(complex(_r, _i))
        _g = (_axis_gray[float(_r)] << 2) | _axis_gray[float(_i)]
        _qam_bits.append([(_g >> b) & 1 for b in range(4)])
QAM16_CONSTELLATION = np.array(_qam_pts, dtype=np.complex64) / math.sqrt(10.0)
QAM16_BITS = np.array(_qam_bits, dtype=np.uint8)

# FSK tone Gray tables (indices in ascending-frequency order)
_FSK2_BITS = np.array([[0], [1]], dtype=np.uint8)
_FSK4_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)

MIN_SYMBOLS_FOR_EVM = 8
SPAN_SYMBOLS = 8.0  # default RRC matched-filter span in symbols


@dataclass
class EVMResult:
    """Error Vector Magnitude (EVM) metrics."""
    evm_rms_percent: float
    evm_db: float
    evm_peak_percent: float

    def to_dict(self) -> dict:
        return {
            "evm_rms_percent": round(float(self.evm_rms_percent), 2),
            "evm_db": round(float(self.evm_db), 2),
            "evm_peak_percent": round(float(self.evm_peak_percent), 2),
        }


@dataclass
class DemodulationResult:
    """Full demodulation, slicing, and EVM verification result.

    `evm` is None when demodulation is deliberately *gated* (short burst,
    Noise/CW input, or FSK path semantics differ). `why` always explains.
    """
    modulation: str
    evm: Optional[EVMResult]
    num_symbols: int
    bits: list[int]
    symbols: list[complex] = field(default_factory=list)
    sliced_symbols: list[complex] = field(default_factory=list)
    freq_offset_hz: float = 0.0
    samples_per_symbol: float = 1.0
    symbol_rate_hz: Optional[float] = None
    timing_offset_samples: float = 0.0
    rolloff_alpha: Optional[float] = None
    residual_phase_rad: float = 0.0
    gated: bool = False
    llrs_sample: list[float] = field(default_factory=list)
    why: str = ""

    def to_dict(self, max_symbols_export: int = 256, max_llr_export: int = 64) -> dict:
        syms_export = [
            {"i": round(float(s.real), 3), "q": round(float(s.imag), 3)}
            for s in self.symbols[:max_symbols_export]
        ]
        return {
            "modulation": self.modulation,
            "evm": self.evm.to_dict() if self.evm is not None else None,
            "num_symbols": self.num_symbols,
            "freq_offset_hz": round(float(self.freq_offset_hz), 2),
            "symbol_rate_hz": round(float(self.symbol_rate_hz), 1) if self.symbol_rate_hz else None,
            "samples_per_symbol": round(float(self.samples_per_symbol), 3),
            "timing_offset_samples": round(float(self.timing_offset_samples), 4),
            "rolloff_alpha": round(float(self.rolloff_alpha), 3) if self.rolloff_alpha is not None else None,
            "residual_phase_rad": round(float(self.residual_phase_rad), 4),
            "gated": bool(self.gated),
            "bits_sample": self.bits[:128],
            "llrs_sample": [round(float(v), 3) for v in self.llrs_sample[:max_llr_export]],
            "symbols_sample": syms_export,
            "why": self.why,
        }


def get_reference_constellation(modulation: str) -> np.ndarray:
    """Get the ideal normalized reference constellation for PSK/QAM."""
    mod = modulation.upper()
    if "BPSK" in mod:
        return BPSK_CONSTELLATION
    elif "16QAM" in mod or "QAM" in mod:
        return QAM16_CONSTELLATION
    else:
        return QPSK_CONSTELLATION


def _get_bits_table(modulation: str) -> np.ndarray:
    mod = modulation.upper()
    if "BPSK" in mod:
        return np.array([[0], [1]], dtype=np.uint8)
    elif "16QAM" in mod or "QAM" in mod:
        return QAM16_BITS
    else:
        return _QPSK_BITS


def slice_symbols(
    symbols: np.ndarray,
    modulation: str = "QPSK",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice recovered complex symbols to the nearest reference constellation points.

    Returns:
        (sliced_symbols, bit_array, error_vector)
    """
    ref = get_reference_constellation(modulation)
    bits_table = _get_bits_table(modulation)
    n = len(symbols)
    if n == 0:
        return (
            np.array([], dtype=np.complex64),
            np.array([], dtype=np.uint8),
            np.array([], dtype=np.complex64),
        )

    pwr = float(np.mean(np.abs(symbols) ** 2))
    norm_factor = 1.0 / math.sqrt(max(pwr, 1e-12))
    s_norm = symbols * norm_factor

    dists = np.abs(s_norm[:, None] - ref[None, :])
    nearest_idx = np.argmin(dists, axis=1)

    sliced = ref[nearest_idx]
    err = s_norm - sliced

    # Gray-correct bit demapping per constellation
    bit_rows = bits_table[nearest_idx]  # (n, bits_per_symbol)
    bits = bit_rows.flatten()

    return sliced, bits.astype(np.uint8), err


def compute_evm(symbols: np.ndarray, sliced: np.ndarray) -> EVMResult:
    """Compute RMS Error Vector Magnitude (EVM) in % and dB."""
    if len(symbols) == 0 or len(sliced) == 0:
        return EVMResult(evm_rms_percent=100.0, evm_db=0.0, evm_peak_percent=100.0)

    # Normalize symbols before EVM so absolute gain does not skew the result
    pwr = float(np.mean(np.abs(symbols) ** 2))
    if pwr < 1e-12:
        return EVMResult(evm_rms_percent=100.0, evm_db=0.0, evm_peak_percent=100.0)
    s_norm = symbols / math.sqrt(pwr)

    err = s_norm - sliced
    err_power = float(np.mean(np.abs(err) ** 2))
    ref_power = float(np.mean(np.abs(sliced) ** 2))

    evm_rms = math.sqrt(err_power / max(ref_power, 1e-12)) * 100.0
    evm_peak = (float(np.max(np.abs(err))) / math.sqrt(max(ref_power, 1e-12))) * 100.0
    evm_db = 20.0 * math.log10(max(evm_rms / 100.0, 1e-6))

    return EVMResult(
        evm_rms_percent=evm_rms,
        evm_db=evm_db,
        evm_peak_percent=evm_peak,
    )


def compute_soft_llr(
    symbols: np.ndarray,
    modulation: str,
    noise_variance_estimate: float = 0.05,
) -> np.ndarray:
    """Per-bit soft log-likelihood ratios for PSK/QAM (used by the FEC stage).

    LLR[k, b] = log( P(bit=0) / P(bit=1) ) computed from Euclidean distances to
    the constellation, assuming AWGN with variance `noise_variance_estimate`.

    Returns:
        Array of shape (n_symbols, bits_per_symbol). Positive => bit more likely 0.
    """
    ref = get_reference_constellation(modulation)
    bits_table = _get_bits_table(modulation)
    n = len(symbols)
    if n == 0:
        return np.zeros((0, bits_table.shape[1]), dtype=np.float32)

    pwr = float(np.mean(np.abs(symbols) ** 2))
    s_norm = symbols * (1.0 / math.sqrt(max(pwr, 1e-12)))

    sigma2 = max(float(noise_variance_estimate), 1e-4)
    dist2 = np.abs(s_norm[:, None] - ref[None, :]) ** 2  # (n, R)
    w = np.exp(-dist2 / sigma2) + 1e-12

    llrs = np.empty((n, bits_table.shape[1]), dtype=np.float32)
    for b in range(bits_table.shape[1]):
        mask0 = bits_table[:, b] == 0
        mask1 = bits_table[:, b] == 1
        num0 = w[:, mask0].sum(axis=1)
        num1 = w[:, mask1].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            llr = np.log(num0 / num1)
        llrs[:, b] = np.clip(llr, -12.0, 12.0)
    return llrs


# ── FSK demodulation (differential phase tone slicing) ──────────────────────

def _kmeans_1d(x: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    """1D Lloyd / k-means returning `k` ascending cluster centers."""
    if k <= 1:
        return np.array([float(np.mean(x))], dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    centers = np.linspace(lo, hi, k, dtype=np.float64)
    if hi == lo:
        return centers
    for _ in range(iters):
        dists = np.abs(x[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([x[labels == j].mean() if np.any(labels == j) else centers[j] for j in range(k)])
        if np.allclose(new_centers, centers, atol=1e-12):
            centers = new_centers
            break
        centers = new_centers
    return np.sort(centers)


def _fsk_per_symbol_freq(
    x: np.ndarray,
    sps: float,
    sample_rate: float,
    num_samples_symbol_free: int = 1,
) -> np.ndarray:
    """Per-symbol single-side instantaneous frequency (cycles/symbol).

    Robust FSK tone measurement: band-limit the burst, discard the band-limiter
    edge transient, then read the phase slope *inside* each symbol rather than
    differencing interpolated complex samples.

    Why this is necessary:
      - Constant-envelope FSK is destroyed by matched-filter/interpolated
        strobes: cubic interpolation across symbol boundaries smears the phase,
        so differential phase over interpolated symbols is noisy even at
        infinite SNR. After a narrowband LPF (which rejects the ~12x
        out-of-band noise), the per-sample instantaneous frequency within a
        symbol is essentially the tone frequency plus small AWGN.
      - Averaging the interior instantaneous-frequency samples telescopes the
        frequency over a whole symbol (x_sps noise reduction) while excluding
        the single boundary-crossing sample.
    """
    from scipy.signal import firwin, convolve

    n = len(x)
    sps_c = float(np.clip(sps, 1.5, 64.0))
    S = int(round(sps_c))
    if n < max(2 * S, 8):
        return np.zeros(0, dtype=np.float64)

    symbol_rate = sample_rate / sps_c
    # LPF cutoff near the symbol-rate main lobe; clipped to a sane band.
    cutoff = float(np.clip(0.7 * symbol_rate, 0.5e6, 0.4 * sample_rate))
    num_taps = 401
    taps = firwin(num_taps, cutoff, fs=sample_rate)
    fx = convolve(x, taps, mode="same")
    trim = num_taps // 2
    if len(fx) > 2 * trim:
        fx = fx[trim:-trim]

    inst = np.angle(fx[1:] * np.conj(fx[:-1])) / (2.0 * np.pi)  # cycles/sample
    n_sym = max(0, int((len(fx) - 1) / S))
    f = np.empty(n_sym, dtype=np.float64)
    margin = max(0, int(num_samples_symbol_free))
    for k in range(n_sym):
        a = k * S
        b = (k + 1) * S - margin
        f[k] = float(inst[a:b].mean()) * S  # averaged slope -> cycles per symbol
    if f.size:
        f = f - float(np.mean(f))
    return f


def _fsk_refine_sps(
    x: np.ndarray,
    candidates: list[float],
    num_tones: int,
    max_symbols: int = 4096,
) -> float:
    """Pick the FSK samples-per-symbol that best separates the FSK tone bins.

    For FSK the global symbol-rate estimators are unreliable (constant envelope
    kills the PSK methods; the freq-waveform line drowns in phase noise at
    realistic SNR). Instead we brute-force candidate strobe periods and keep the
    one whose differential-phase clusters are most cleanly *discrete*.

    We stroke each candidate with cubic Farrow interpolation, measure the
    differential phase between consecutive (aligned) symbol samples, and score
    the result with the Fisher between/within variance ratio:

        ratio = var(cluster centers) / mean((f - center[label])^2)

    Maximizing this rewards a strobe period that (a) concentrates the per-tone
    differential phase tightly (low within-cluster variance) while (b) keeping
    the tone bins far apart (high between-cluster variance).

    This matters because plain within-cluster variance alone is *biased* toward
    under-sampled periods: strobing below the true symbol rate compresses the
    apparent frequency deviation (each increment spans a fraction of a symbol),
    so the clusters look tighter regardless of correctness. Only at the true
    symbol rate do the tone bins collapse onto the full-size, well-separated
    deviation levels -> the between/within ratio spikes. Verified empirically:
    the ratio peaks cleanly at the true sps (8) for 2-FSK and 4-FSK at both
    22 dB and 40 dB, whereas raw within-variance picks the aliased sps=3.
    """
    from app.demodulation.synchronizer import gardner_farrow_timing

    best_sps = float(candidates[0])
    best_score = -np.inf
    for sps_c in candidates:
        if sps_c < 3.0 or sps_c > 64.0:
            continue
        try:
            syms, _, _ = gardner_farrow_timing(x, sps_c, initial_mu=0.0, max_symbols=max_symbols)
        except Exception:
            continue
        if len(syms) < 16:
            continue
        w = syms[1:] * np.conj(syms[:-1])
        f = np.angle(w) / (2.0 * np.pi)
        f = f - float(np.mean(f))
        centers = _kmeans_1d(f, num_tones)
        dists = np.abs(f[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        within = float(np.mean((f - centers[labels]) ** 2))
        between = float(np.var(centers[labels]))
        score = between / within if within > 1e-12 else np.inf
        if score > best_score:
            best_score = score
            best_sps = sps_c
    return best_sps


def _demodulate_fsk(
    signal: ComplexSignal,
    num_tones: int,
    sps: float,
    sample_rate: float,
    max_symbols: int = 4096,
) -> DemodulationResult:
    """Differential phase-demodulate an FSK burst and slice tone bins.

    Band-limits the burst and measures the per-symbol instantaneous frequency
    (see `_fsk_per_symbol_freq`) so the tone levels are read from the clean
    interior of each symbol rather than across fractional inter-symbol
    transitions or interpolated complex samples.
    """
    x = np.asarray(signal.data, dtype=np.complex64)
    n = len(x)
    sps = float(np.clip(sps, 1.5, 64.0))
    available = int(n / sps)
    if available <= 0:
        return DemodulationResult(
            modulation=f"{num_tones}-FSK", evm=None, num_symbols=0, bits=[],
            symbols=[], sliced_symbols=[], why="No complete symbols in burst.",
        )

    f_sym = _fsk_per_symbol_freq(x, sps, sample_rate)

    num = len(f_sym)
    if num < MIN_SYMBOLS_FOR_EVM:
        return DemodulationResult(
            modulation=f"{num_tones}-FSK", evm=None, num_symbols=num, bits=[],
            symbols=[], sliced_symbols=[],
            samples_per_symbol=sps, gated=True,
            why=f"Insufficient symbols ({num} < {MIN_SYMBOLS_FOR_EVM}) for reliable FSK tone slicing.",
        )

    centers_sym = _kmeans_1d(f_sym, num_tones)
    dev = max(float(np.max(np.abs(centers_sym))), 1e-9)
    centers_norm = centers_sym / dev
    f_norm = f_sym / dev

    ref = centers_norm.astype(np.complex64)
    dists = np.abs(f_norm[:, None].astype(np.complex64) - ref[None, :])
    nearest = np.argmin(dists, axis=1)
    sliced = ref[nearest]

    bits_table = _FSK4_BITS if num_tones == 4 else _FSK2_BITS
    bit_rows = bits_table[np.clip(nearest, 0, len(bits_table) - 1)]
    bits = bit_rows.flatten().astype(np.uint8)

    # Frequency-domain EVM normalized to frequency deviation
    err = f_norm - sliced
    err_rms = float(np.sqrt(np.mean(np.abs(err) ** 2))) * 100.0
    evm_peak = float(np.max(np.abs(err))) * 100.0
    evm_db = 20.0 * math.log10(max(err_rms / 100.0, 1e-6))
    evm = EVMResult(evm_rms_percent=min(err_rms, 1000.0), evm_db=evm_db, evm_peak_percent=min(evm_peak, 1000.0))

    why = (
        f"Demodulated {num_tones}-FSK via band-limited instantaneous "
        f"frequency per symbol: {num} tones, deviation span {2*dev:.3f} cycles/symbol "
        f"(fs={sample_rate/1e6:.2f} MHz, sps={sps:.2f}). Frequency EVM={err_rms:.1f}%."
    )
    return DemodulationResult(
        modulation=f"{num_tones}-FSK",
        evm=evm,
        num_symbols=num,
        bits=bits.tolist(),
        symbols=f_norm.tolist(),
        sliced_symbols=sliced.tolist(),
        freq_offset_hz=0.0,
        samples_per_symbol=sps,
        symbol_rate_hz=sample_rate / sps,
        timing_offset_samples=0.0,
        why=why,
    )


# ── End-to-end demodulation ─────────────────────────────────────────────────

def demodulate_signal(
    signal: ComplexSignal,
    modulation: Optional[str] = None,
    samples_per_symbol: Optional[float] = None,
    max_symbols: int = 4096,
    apply_mf: bool = True,
) -> DemodulationResult:
    """End-to-end demodulation and verification for a ComplexSignal.

    1. If modulation is omitted, runs the candidate classifier.
    2. Non-demodulable inputs (Noise/CW) and ultra-short bursts are gated.
    3. FSK is handled by differential tone slicing (no matched filter).
    4. PSK/QAM: coarse CFO -> RRC matched filter -> timing recovery
       -> decision-directed phase PLL -> phase-ambiguity resolution
       -> slicing -> EVM + soft LLRs.
    5. Appends one provenance step per operation.
    """
    sr = signal.metadata.sample_rate or 1.0
    n_data = len(signal.data)

    # Auto-detect modulation if not specified
    if modulation is None:
        clf_res = classify_signal(signal)
        target_mod = clf_res.top_candidate
    else:
        target_mod = modulation
    mod_upper = target_mod.upper()

    # Gate non-digital / non-demodulable inputs
    if "NOISE" in mod_upper or "CW" in mod_upper:
        return DemodulationResult(
            modulation=target_mod, evm=None, num_symbols=0, bits=[], symbols=[],
            sliced_symbols=[], why="Non-demodulable classification (Noise/CW); no symbols recovered.",
        )

    # Bandwidth hint (helps symbol-rate estimation and rolloff selection)
    bw_hint = None
    try:
        fft_n = min(1024, n_data)
        if fft_n >= 64:
            psd = compute_psd(signal, fft_size=fft_n)
            bw_res = estimate_bandwidth(psd, method="3db")
            bw_hint = bw_res.bandwidth_hz
    except Exception:
        pass

    # Coarse CFO for FSK: center-of-mass (differential demod removes it anyway)
    cfo_hz = estimate_coarse_cfo(signal.data[:16384], sr, target_mod)
    if "FSK" in mod_upper:
        # Seed samples-per-symbol from the estimator (bandwidth hints are
        # unreliable for FSK: the 3dB BW of a two-tone CPFSK spectrum is tiny).
        if samples_per_symbol is not None and samples_per_symbol >= 1.5:
            sps = float(samples_per_symbol)
        else:
            sym_res = estimate_symbol_rate(signal.data[:32768], sample_rate=sr, bandwidth_hz=bw_hint)
            if sym_res.confidence > 0.3 and sym_res.symbol_rate_hz > 0:
                sps = sr / sym_res.symbol_rate_hz
            else:
                sps = 8.0
            # Self-tuning: pick the strobe period that gives the tightest tone
            # clusters (global estimators are systematically weak for FSK).
            num_tones = 4 if "4" in target_mod else 2
            cands = sorted(set([
                3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 32.0,
                float(np.clip(sps, 3.0, 32.0)),
                float(np.clip(sps * 1.15, 3.0, 32.0)),
                float(np.clip(sps * 0.85, 3.0, 32.0)),
            ]))
            sps = _fsk_refine_sps(signal.data, cands, num_tones, max_symbols=max_symbols)
        sps = float(np.clip(sps, 1.5, 64.0))
        num_tones = 4 if "4" in target_mod else 2
        res = _demodulate_fsk(signal, num_tones, sps, sr, max_symbols=max_symbols)
        signal.add_provenance(
            stage="demodulation",
            operation="demodulate_fsk_slice",
            parameters={
                "modulation": res.modulation,
                "num_symbols": res.num_symbols,
                "samples_per_symbol": sps,
                "gated": res.gated,
            },
        )
        return res

    # ── Linear modulations (PSK/QAM) ────────────────────────────────────────
    # Symbol-rate -> sps
    sym_res = None
    if samples_per_symbol is not None and samples_per_symbol >= 1.0:
        sps = float(samples_per_symbol)
    else:
        sym_res = estimate_symbol_rate(signal.data[:32768], sample_rate=sr, bandwidth_hz=bw_hint)
        if sym_res.confidence > 0.3 and sym_res.symbol_rate_hz > 0:
            sps = sr / sym_res.symbol_rate_hz
        else:
            sps = 1.0

    # Honest gating: an ultra-short burst with no recoverable symbol clock
    # must never produce a garbage EVM.
    if (samples_per_symbol is None and sym_res is not None
            and sym_res.confidence <= 0.3 and n_data < 256):
        return DemodulationResult(
            modulation=target_mod, evm=None, num_symbols=0, bits=[],
            symbols=[], sliced_symbols=[], freq_offset_hz=cfo_hz,
            gated=True,
            why=(f"Symbol clock not recoverable from ultra-short burst "
                 f"(n={n_data} samples, symbol-rate confidence "
                 f"{sym_res.confidence:.2f} <= 0.3); EVM withheld honestly."),
        )
    sps = float(np.clip(sps, 1.0, 64.0))

    symbol_rate_hz = sr / sps if sps > 1.0 else None
    alpha = estimate_rolloff(bw_hint, symbol_rate_hz)
    ref = get_reference_constellation(target_mod)

    # 1. Coarse carrier derotation
    iq_derot = derotate_carrier(signal.data, freq_offset_hz=cfo_hz, sample_rate=sr)

    # 2. RRC matched filter
    iq_synced = iq_derot
    if apply_mf and sps >= 2.0:
        iq_synced = apply_matched_filter(iq_derot, sps, alpha=alpha, span_symbols=SPAN_SYMBOLS)

    # 3. Symbol timing recovery
    timing = recover_symbol_timing(iq_synced, sps, max_symbols=max_symbols)
    symbols = timing.symbols
    num_symbols = timing.num_symbols

    if num_symbols < MIN_SYMBOLS_FOR_EVM:
        return DemodulationResult(
            modulation=target_mod, evm=None, num_symbols=num_symbols,
            bits=[], symbols=symbols.tolist(), sliced_symbols=[],
            freq_offset_hz=cfo_hz, symbol_rate_hz=symbol_rate_hz,
            samples_per_symbol=timing.samples_per_symbol,
            timing_offset_samples=timing.timing_offset, rolloff_alpha=alpha,
            gated=True,
            why=(f"Insufficient symbols ({num_symbols} < {MIN_SYMBOLS_FOR_EVM}) "
                 f"for reliable EVM; symbols exported for visualization only."),
        )

    # 4. Fine carrier/phase recovery
    phase_synced, _ = dd_phase_recovery(symbols, modulation=target_mod, loop_bw=0.02)

    # 5. Phase-ambiguity resolution
    resolved, rot = resolve_phase_ambiguity(phase_synced, target_mod, ref)

    # 6. Slicing + EVM
    sliced, bits, err = slice_symbols(resolved, modulation=target_mod)

    # Steady-state EVM: matched-filter edge transients are excluded per standard
    # EVM practice (they are a burst-length artifact, not a demod-quality metric).
    trim = max(1, int(SPAN_SYMBOLS / 2)) if (apply_mf and sps >= 2.0) else 0
    if trim > 0 and len(resolved) > 2 * trim:
        resolved_evm = resolved[trim:-trim]
        sliced_evm = sliced[trim:-trim]
    else:
        resolved_evm = resolved
        sliced_evm = sliced
    evm_res = compute_evm(resolved_evm, sliced_evm)

    # 7. Soft LLRs for the FEC stage
    sigma2 = (evm_res.evm_rms_percent / 100.0) ** 2
    llrs = compute_soft_llr(resolved, target_mod, noise_variance_estimate=sigma2)

    why = (
        f"Demodulated {target_mod}: EVM={evm_res.evm_rms_percent:.1f}% ({evm_res.evm_db:.1f} dB), "
        f"CFO={cfo_hz:.1f} Hz, sps={timing.samples_per_symbol:.2f}, "
        f"timing offset={timing.timing_offset:.3f} samples, rolloff={alpha:.2f}, "
        f"phase rotation={rot:.3f} rad, {len(symbols)} symbols "
        f"(steady-state EVM over {len(resolved_evm)} of {len(symbols)})."
    )

    signal.add_provenance(
        stage="demodulation",
        operation="demodulate_and_slice",
        parameters={
            "modulation": target_mod,
            "evm_rms_percent": evm_res.evm_rms_percent,
            "evm_db": evm_res.evm_db,
            "cfo_hz": cfo_hz,
            "symbol_rate_hz": symbol_rate_hz,
            "samples_per_symbol": timing.samples_per_symbol,
            "timing_offset_samples": timing.timing_offset,
            "rolloff_alpha": alpha,
            "phase_rotation_rad": rot,
            "num_symbols": len(symbols),
        },
    )

    return DemodulationResult(
        modulation=target_mod,
        evm=evm_res,
        num_symbols=len(symbols),
        bits=bits.tolist(),
        symbols=symbols.tolist(),
        sliced_symbols=sliced.tolist(),
        freq_offset_hz=cfo_hz,
        samples_per_symbol=timing.samples_per_symbol,
        symbol_rate_hz=symbol_rate_hz,
        timing_offset_samples=timing.timing_offset,
        rolloff_alpha=alpha,
        residual_phase_rad=rot,
        llrs_sample=llrs.flatten().tolist(),
        why=why,
    )