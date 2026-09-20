#!/usr/bin/env python3
"""Run the full RF pipeline on a signal and build the Evidence Graph.

The Evidence Graph converts every pipeline measurement (detection facts,
characterization features, modulation classifier scores, demodulation EVM,
decode-chain CRC validity) into a weighted, provenance-linked graph per
hypothesis. This script is the end-to-end demonstration of
``app/evidence``:

    1. load + normalize a real IQ file (or synthesize a QPSK burst)
    2. detection   -> PSD / noise floor / SNR / bandwidth / segments
    3. characterize-> features / symbol rate / CFO
    4. classify    -> ranked modulation candidates
    5. demodulate  -> EVM / symbols / LLRs (gate noise & short bursts)
    6. decode      -> constrained candidate chains (Viterbi / RS / CRC)
    7. evidence    -> cross-stage weighted evidence graph + leaderboard
    8. export      -> JSON + GraphML (Gephi / yEd) under outputs/evidence/

Usage:
    python scripts/run_evidence_graph.py -f <signal.iq> --sample-rate 2500000
    python scripts/run_evidence_graph.py --synthetic            # QPSK burst
    python scripts/run_evidence_graph.py -f <frame.sigmf-data>  # Zenodo frame
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from app.models import ComplexSignal, SignalMetadata
from app.ingestion.parser import load_signal
from app.preprocessing.normalize import full_normalize
from app.detection.fft import compute_psd
from app.detection.noise_floor import estimate_noise_floor
from app.detection.segmentation import build_segments
from app.detection.snr import estimate_snr_from_psd
from app.detection.bandwidth import estimate_bandwidth
from app.characterization.features import extract_features
from app.characterization.symbol_rate import estimate_symbol_rate
from app.characterization.freq_offset import estimate_freq_offset
from app.modulation.classifier import classify_signal
from app.demodulation.slicer import demodulate_signal
from app.decoding.decoder import search_decode

from app.evidence import (
    build_evidence_graph,
    export_evidence_graph,
)


def make_synthetic_qpsk(sample_rate: float = 2.0e6, n_samples: int = 16384,
                        snr_db: float = 20.0, seed: int = 7, sps: int = 16):
    """Synthesize a clean QPSK burst (no interleaver, raw bits) for demo."""
    rng = np.random.default_rng(seed)
    n_sym = n_samples // sps
    bits = rng.integers(0, 2, n_sym * 2)
    sym_map = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=np.complex64)
    symbols = sym_map[(bits[0::2] * 2 + bits[1::2]).astype(int)] / np.sqrt(2.0)
    t = np.arange(n_sym) * sps
    iq = np.zeros(n_samples, dtype=np.complex64)
    for k, s in enumerate(symbols):
        iq[t[k]:(t[k] + sps)] = s
    # roll on a smooth RRC-ish pulse to keep spectrum realistic
    from scipy.signal import firwin, lfilter
    taps = firwin(63, 0.55, window="hann")
    iq = lfilter(taps, 1.0, iq).astype(np.complex64)
    iq /= np.sqrt(np.mean(np.abs(iq) ** 2))
    noise_pow = 10.0 ** (-snr_db / 10.0)
    noise = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    noise *= np.sqrt(noise_pow / 2.0)
    iq = (iq + noise.astype(np.complex64))
    return iq.astype(np.complex64)


def make_synthetic_signal(sample_rate: float = 2.0e6) -> ComplexSignal:
    data = make_synthetic_qpsk(sample_rate=sample_rate)
    meta = SignalMetadata(
        filename="synthetic_qpsk.iq", file_size_bytes=data.nbytes,
        data_type="cf32", sample_rate=sample_rate,
        center_frequency=100.0e6, iq_ordering="interleaved",
        num_samples=len(data),
    )
    sig = ComplexSignal(data=data, metadata=meta)
    sig.add_provenance("synthetic", "generate_qpsk_burst", {"snr_db": 20.0})
    return sig


def run_full_pipeline(sig: ComplexSignal):
    """Run detection -> characterization -> classification -> demod -> decode."""
    sig = full_normalize(sig, remove_dc_offset=True, normalize_to="power")

    psd = compute_psd(sig, fft_size=1024)
    nf = estimate_noise_floor(psd, method="mad")
    seg = build_segments(psd, nf, sig)
    snr = estimate_snr_from_psd(psd, nf)
    bw = estimate_bandwidth(psd, method="3db")

    fv = extract_features(sig.data[:65536], sample_rate=sig.metadata.sample_rate)
    sym = estimate_symbol_rate(sig.data[:65536], sample_rate=sig.metadata.sample_rate)
    cfo = estimate_freq_offset(sig.data[:16384], sample_rate=sig.metadata.sample_rate)

    mod = classify_signal(sig, max_samples=65536)
    demod = demodulate_signal(sig, modulation=mod.top_candidate, max_symbols=1024)
    decode = []
    if demod.bits:
        decode = search_decode(
            demod.bits, llrs=(demod.llrs_sample or None),
            use_soft_viterbi=True, top_k=10,
        )

    return dict(snr=snr, bandwidth=bw, noise_floor=nf, segmentation=seg,
                features=fv, symbol_rate=sym, freq_offset=cfo,
                modulation=mod, demodulation=demod, decoding=decode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-f", "--file", help="Signal file (.iq/.wav/.sigmf-data)")
    ap.add_argument("--sample-rate", type=float, default=None,
                    help="Override sample rate (Hz)")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use an in-memory synthetic QPSK burst instead of a file")
    ap.add_argument("-o", "--out-dir", default=ROOT / "outputs" / "evidence",
                    help="Output directory for JSON + GraphML")
    args = ap.parse_args()

    if args.synthetic:
        print("Using synthetic QPSK burst (SNR=20 dB, fs=2 MHz)")
        sig = make_synthetic_signal(sample_rate=args.sample_rate or 2.0e6)
    elif args.file:
        print(f"Loading {args.file}")
        sig = load_signal(args.file, sample_rate=args.sample_rate)
    else:
        ap.error("provide --file or --synthetic")

    print("\n[1] full pipeline (detection/characterization/classification/demod/decode)")
    stages = run_full_pipeline(sig)

    print("\n[2] detection summary")
    print(f"    SNR={stages['snr'].snr_db:.1f} dB  BW={stages['bandwidth'].bandwidth_hz:.0f} Hz  "
          f"segments={stages['segmentation'].num_segments}")

    print("\n[3] characterization summary")
    print(f"    symbol_rate={stages['symbol_rate'].symbol_rate_hz:.0f} Hz "
          f"(conf {stages['symbol_rate'].confidence:.2f})  "
          f"CFO={stages['freq_offset'].offset_hz:.0f} Hz")

    print("\n[4] modulation classification")
    for c in stages["modulation"].candidates[:3]:
        print(f"    {c.modulation:8s} score={c.score:.3f} conf={c.confidence:.3f}")

    demod = stages["demodulation"]
    print("\n[5] demodulation")
    if demod.gated:
        print(f"    GATED: {demod.why}")
    else:
        print(f"    {demod.modulation}: EVM={demod.evm.evm_rms_percent if demod.evm else 'n/a'}%  "
              f"symbols={demod.num_symbols}  residual_phase={demod.residual_phase_rad:.3f} rad")

    print("\n[6] decode chain leaderboard")
    for r in stages["decoding"][:5]:
        print(f"    crc={str(r.crc_valid):5s} verr={r.num_viterbi_errors:3d} "
              f"rs={r.num_rs_symbols_corrected:3d} corr={r.preamble_correlation:.2f}  {r.spec_name}")

    print("\n[7] evidence graph")
    graph = build_evidence_graph(
        snr=stages["snr"], bandwidth=stages["bandwidth"],
        noise_floor=stages["noise_floor"], segmentation=stages["segmentation"],
        features=stages["features"], symbol_rate=stages["symbol_rate"],
        freq_offset=stages["freq_offset"], mod_classification=stages["modulation"],
        demodulation=stages["demodulation"], decoding=stages["decoding"],
        metadata={"filename": sig.metadata.filename,
                  "sample_rate": sig.metadata.sample_rate,
                  "center_frequency": sig.metadata.center_frequency},
    )

    for h in graph.top_hypotheses(k=12):
        print(f"    {h.kind:13s} {h.name[:48]:48s} score={h.score:.3f} "
              f"quality={h.quality} items={h.num_items}")

    out = export_evidence_graph(graph, args.out_dir, basename="evidence")
    print(f"\n[8] exported\n    JSON:    {out['json']}\n    GraphML: {out['graphml']}")
    if not args.synthetic:
        print(f"\n    graph covers {len(graph.items)} items / {len(graph.edges)} edges "
              f"/ {len(graph.hypotheses)} hypotheses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())