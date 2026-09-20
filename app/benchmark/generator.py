"""
app/benchmark/generator.py
--------------------------
Synthetic signal + verified-chain generator for the benchmark corpus
(Build order §9.2).

Produces ~200 standardized IQ bursts with FULL hidden ground truth
(modulation × FEC × interleaver × frame × SNR) through the exact encoder chain
the decoder consumes, so a corpus entry's ``verified_chain`` is honest:

    payload + CRC[preamble]  ->  FEC (none | viterbi | rs255_223)
    ->  interleave (none | block(4,8))  ->  symbol map (BPSK/QPSK/16QAM/2FSK)
    ->  upsample by sps  ->  RRC pulse shape  ->  CFO + fractional delay
    ->  AWGN at target SNR

Feature vectors are the SAME characterization estimates the main pipeline
produces at query time (PSD SNR, -3dB bandwidth, 4-method symbol rate,
22-feature extraction), cached content-addressed so regenerating the corpus is
cheap. The generator itself is a hard prerequisite for the advisor (§ "no
corpus, no advisor").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve, resample_poly

from app.models import ComplexSignal, SignalMetadata, DataType, IQOrdering
from app.decoding.convolutional import encode_convolutional
from app.decoding.reed_solomon import RS255_223, rs_encode
from app.decoding.interleaver import none_spec, block_spec, interleave
from app.decoding.frame import crc_bytes, bytes_to_bits, CRC16_IBM, CRC32_IEEE
from app.decoding.decoder import (
    DecodingSpec,
    FEC_NONE,
    FEC_VITERBI,
    FEC_RS223,
)
from app.decoding.frame import FrameDescriptor
from app.demodulation.pulseshape import design_rrc_taps
from app.characterization.features import extract_features, FeatureVector

from app.benchmark.cache import get_cache
from app.benchmark.corpus import (
    BenchmarkCorpus,
    CorpusEntry,
    build_query_vector,
)

# Canonical TX options exposed by the generator.
MODULATIONS = ("BPSK", "QPSK", "16QAM", "2FSK")
INTERLEAVERS = ("none", "block(4,8)")
FECS = ("none", "viterbi", "rs255_223")
FRAMES = ("preamble+crc16", "preamble+crc32")
SNR_POINTS = (5.0, 10.0, 15.0, 20.0)

_PREAMBLE = [0, 1, 1, 1, 1, 1, 1, 0] * 2

_CRC_VARIANTS = {"preamble+crc16": CRC16_IBM, "preamble+crc32": CRC32_IEEE}

# QPSK unit-power lattice helpers.
_QPSK_PTS = (np.array([1, 1, -1, -1]) + 1j * np.array([1, -1, 1, -1])) / np.sqrt(2.0)
_16QAM_LVL = np.array([-3, -1, 1, 3]) / np.sqrt(10.0)


def _build_frame_bits(payload: bytes, crc_variant: str, fec: str) -> np.ndarray:
    """TX bitstream before interleaving: preamble|payload|CRC, FEC-encoded."""
    crc_poly = _CRC_VARIANTS[crc_variant]
    frame_bits = np.concatenate([
        np.asarray(_PREAMBLE, dtype=np.int64),
        bytes_to_bits(payload),
        bytes_to_bits(crc_bytes(payload, variant=crc_poly)),
    ])
    if fec == FEC_VITERBI:
        return encode_convolutional(frame_bits)
    if fec == FEC_RS223:
        msg = np.frombuffer(payload + crc_bytes(payload, variant=crc_poly),
                            dtype=np.uint8)
        padded = np.zeros(RS255_223.k, dtype=np.uint8)
        padded[: len(msg)] = msg
        return bytes_to_bits(rs_encode(padded, RS255_223))
    return frame_bits


def _interleave_bits(bits: np.ndarray, ilv_name: str) -> np.ndarray:
    spec = none_spec() if ilv_name == "none" else block_spec(4, 8)
    if spec.type == "block":
        pad = (-len(bits)) % spec.size
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.int64)])
    return interleave(bits, spec), spec


def _map_to_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    """Map (truncated) bits to unit-power complex symbols."""
    b = bits.astype(np.int64)
    if modulation == "BPSK":
        if len(b) % 1:
            b = b[:len(b) - len(b) % 1]
        return (1.0 - 2.0 * b).astype(np.complex64)
    if modulation == "QPSK":
        groups = b[:len(b) - len(b) % 2].reshape(-1, 2)
        idx = groups[:, 0] * 2 + groups[:, 1]
        return _QPSK_PTS[idx].astype(np.complex64)
    if modulation == "16QAM":
        groups = b[:len(b) - len(b) % 4].reshape(-1, 4)
        i_bits = groups[:, 0] * 2 + groups[:, 1]
        q_bits = groups[:, 2] * 2 + groups[:, 3]
        return (_16QAM_LVL[i_bits] + 1j * _16QAM_LVL[q_bits]).astype(np.complex64)
    raise ValueError(f"unsupported linear modulation: {modulation}")


def _pulse_shape(symbols: np.ndarray, sps: int, alpha: float = 0.35) -> np.ndarray:
    """Upsample + RRC pulse shaping (matched-filter-compatible)."""
    shaped = np.zeros(len(symbols) * sps, dtype=np.complex64)
    shaped[::sps] = symbols
    taps = design_rrc_taps(sps, alpha=alpha, span_symbols=8)
    return fftconvolve(shaped, taps.astype(np.complex64), mode="same").astype(np.complex64)


def _binary_fsk(symbols_bits: np.ndarray, sps: int, sample_rate: float,
                fdev_ratio: float = 0.5) -> np.ndarray:
    """CPFSK over the bit symbols at ±fdev (band-limited continuous phase)."""
    fdev = sample_rate / sps * fdev_ratio
    bits = np.asarray(symbols_bits, dtype=np.int64)
    freq = (2.0 * bits - 1.0) * fdev          # ±fdev square wave
    freq_up = np.repeat(freq, sps)
    # Integrate frequency → phase (approximates continuous phase FSK).
    phase = np.cumsum(2.0 * np.pi * freq_up / sample_rate)
    return np.exp(1j * phase).astype(np.complex64)


def _apply_impairments(iq: np.ndarray, sample_rate: float, rng) -> np.ndarray:
    n = len(iq)
    cfo = rng.uniform(-0.004, 0.004)            # ±0.4% of fs
    tau = rng.uniform(-1.5, 1.5)                 # fractional-sample delay
    t = np.linspace(0.0, n / sample_rate, n, endpoint=False)
    iq = iq * np.exp(1j * 2.0 * np.pi * cfo * t)
    pos = np.clip(np.arange(n) + tau, 0, n - 1)
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    iq = iq[i0] * (1.0 - frac) + iq[np.clip(i0 + 1, 0, n - 1)] * frac
    # Unit power
    p = float(np.mean(np.abs(iq) ** 2))
    iq = iq / np.sqrt(max(p, 1e-12))
    return iq.astype(np.complex64)


def _add_awgn(iq: np.ndarray, snr_db: float, rng) -> np.ndarray:
    sig_p = float(np.mean(np.abs(iq) ** 2))
    noise_p = sig_p / (10.0 ** (snr_db / 10.0))
    noise = (rng.standard_normal(len(iq)) + 1j * rng.standard_normal(len(iq)))
    noise = noise / np.sqrt(2.0) * np.sqrt(noise_p)
    return (iq + noise).astype(np.complex64)


@dataclass
class SyntheticGroundTruth:
    """Hidden ground truth for one generated burst."""
    modulation: str
    ilverleaver: str
    fec: str
    frame: str
    snr_true_db: float
    bit_rate_hz: float
    sample_rate: float
    chain_name: str
    payload: bytes
    num_symbols: int

    def canonical_chain(self) -> str:
        return f"{self.fec}|{self.ilverleaver}|{self.frame}"


def generate_synthetic_sample(
    modulation: str,
    ilv_name: str,
    fec_name: str,
    frame_name: str,
    snr_db: float,
    sample_rate: float = 1.0e6,
    sps: int = 8,
    n_symbols_target: int = 1024,
    payload_size: int = 48,
    seed: int = 0,
) -> tuple[np.ndarray, SyntheticGroundTruth]:
    """Generate one impaired synthetic burst and its hidden ground truth."""
    rng = np.random.default_rng(seed)
    payload = rng.bytes(payload_size)
    bits = _build_frame_bits(payload, frame_name, fec_name)
    bits, ilv_spec = _interleave_bits(bits, ilv_name)

    bps = {"BPSK": 1, "QPSK": 2, "16QAM": 4}[modulation] if modulation != "2FSK" else 1
    bits = bits[: bps * n_symbols_target] if len(bits) > bps * n_symbols_target else bits
    n_sym = len(bits) // bps

    if modulation == "2FSK":
        sym_bits = bits[:n_sym]
        iq = _binary_fsk(sym_bits, sps, sample_rate, fdev_ratio=0.5)
    else:
        symbols = _map_to_symbols(bits[: bps * n_sym], modulation)
        iq = _pulse_shape(symbols, sps)

    iq = _apply_impairments(iq, sample_rate, rng)
    iq = _add_awgn(iq, snr_db, rng)

    gt = SyntheticGroundTruth(
        modulation=modulation,
        ilverleaver=str(ilv_spec),
        fec=fec_name,
        frame=frame_name,
        snr_true_db=float(snr_db),
        bit_rate_hz=sample_rate / sps * bps,
        sample_rate=sample_rate,
        chain_name=f"{fec_name}|{ilv_spec}|{frame_name}",
        payload=payload,
        num_symbols=n_sym,
    )
    return iq, gt


# ── Feature estimation (same path the live pipeline uses) ──────────────────

def estimate_physical_features(
    iq: np.ndarray, sample_rate: float, use_cache: bool = True
) -> dict:
    """Pipeline-identical estimates: snr_db, bw_hz, symbol_rate, FeatureVector."""
    cache = get_cache()
    if use_cache:
        hit = cache.get("physical_features", iq, {"sr": sample_rate})
        if hit is not None:
            return hit

    signal = ComplexSignal(
        data=iq.astype(np.complex64),
        metadata=SignalMetadata(
            filename="<synthetic>", file_size_bytes=0, data_type=DataType.CF32,
            sample_rate=float(sample_rate), iq_ordering=IQOrdering.INTERLEAVED,
            num_samples=len(iq),
        ),
    )
    from app.detection.fft import compute_psd
    from app.detection.noise_floor import estimate_noise_floor
    from app.detection.snr import estimate_snr_from_psd
    from app.detection.bandwidth import estimate_bandwidth
    from app.characterization.symbol_rate import estimate_symbol_rate

    fft_n = min(1024, len(iq))
    psd = compute_psd(signal, fft_size=fft_n) if fft_n >= 64 else None
    nf = estimate_noise_floor(psd) if psd is not None else None
    snr = estimate_snr_from_psd(psd, nf) if psd is not None and nf is not None else None
    bw = estimate_bandwidth(psd, method="3db") if psd is not None else None
    sym = estimate_symbol_rate(iq[:32768], sample_rate=sample_rate,
                               bandwidth_hz=bw.bandwidth_hz if bw else None)
    fv = extract_features(iq, sample_rate=sample_rate,
                          snr_db=snr.snr_db if snr else None)

    result = {
        "snr_db": float(snr.snr_db) if snr else 0.0,
        "bandwidth_hz": float(bw.bandwidth_hz) if bw else 0.0,
        "symbol_rate_hz": float(sym.symbol_rate_hz),
        "feature_vector": fv.to_array().tolist(),
        "feature_dict": {k: float(v) for k, v in fv.to_dict().items()
                         if isinstance(v, (int, float))},
    }
    if use_cache:
        cache.put("physical_features", iq, {"sr": sample_rate}, result)
    return result


def corpus_query_vector(feats: dict) -> np.ndarray:
    """Query vector for a corpus entry from the estimation result dict."""
    fv = FeatureVector()
    for k, v in feats.get("feature_dict", {}).items():
        if hasattr(fv, k):
            setattr(fv, k, v)
    return build_query_vector(
        snr_db=feats.get("snr_db"),
        bandwidth_hz=feats.get("bandwidth_hz"),
        symbol_rate_hz=feats.get("symbol_rate_hz"),
        sample_rate=fv.sample_rate or 1.0,
        fv=fv,
    )


def generate_corpus(
    path: str | Path,
    modulations: tuple[str, ...] = MODULATIONS,
    interleavers: tuple[str, ...] = INTERLEAVERS,
    fes: tuple[str, ...] = FECS,
    frames: tuple[str, ...] = FRAMES,
    snrs: tuple[float, ...] = SNR_POINTS,
    samples_per_config: int = 1,
    seed: int = 42,
    sample_rate: float = 1.0e6,
    sps: int = 8,
) -> BenchmarkCorpus:
    """Build the full synthetic corpus (default grid ≈ 4×2×3×2×4 = 192 runs)."""
    corpus = BenchmarkCorpus()
    rng_global = np.random.default_rng(seed)
    idx = 0
    for mod in modulations:
        for ilv in interleavers:
            for fec in fes:
                for frame in frames:
                    for snr in snrs:
                        for rep in range(samples_per_config):
                            sig_seed = int(rng_global.integers(0, 2 ** 31))
                            iq, gt = generate_synthetic_sample(
                                modulation=mod, ilv_name=ilv, fec_name=fec,
                                frame_name=frame, snr_db=float(snr), seed=sig_seed,
                                sample_rate=sample_rate, sps=sps,
                            )
                            feats = estimate_physical_features(iq, sample_rate)
                            qv = corpus_query_vector(feats)
                            corpus.add_from_run(
                                feature_vector=qv,
                                verified_modulation="2-FSK" if mod == "2FSK" else mod,
                                verified_chain=gt.canonical_chain(),
                                crc_validated=True,
                                snr_db=float(snr),
                                truth_source="synthetic",
                                run_id=f"synth-{idx}",
                            )
                            idx += 1
    corpus.save(path)
    return corpus


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build the synthetic HITL benchmark corpus")
    p.add_argument("-o", "--output", default="corpus.json")
    p.add_argument("--samples-per-config", type=int, default=1)
    args = p.parse_args()
    corpus = generate_corpus(args.output, samples_per_config=args.samples_per_config)
    print(f"corpus: {corpus.n} entries -> {corpus.save(args.output)}")
    print(f"source mix: {corpus.source_mix()}")