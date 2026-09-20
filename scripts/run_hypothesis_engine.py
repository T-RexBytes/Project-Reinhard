#!/usr/bin/env python3
"""Run the hypothesis engine over a signal: DAG search + scoring + WHY.

The Hypothesis Engine consumes the Evidence Graph and expands the top
modulation candidates into a bounded (Modulation -> Demodulation ->
Deinterleaver -> FEC -> Frame) DAG. Every live chain is verified by actually
demodulating and decoding the signal; each chain is scored per stage, its
composite score computed with uncertainty quantification, and a natural
language WHY explanation is generated. This script demonstrates the full
workflow:

    1. load + normalize a real IQ file (or synthesize a QPSK burst)
    2. detection   -> PSD / noise floor / SNR / bandwidth / segments
    3. characterize-> features / symbol rate / CFO
    4. classify    -> ranked modulation candidates
    5. demodulate  -> EVM / symbols / LLRs (gate noise & short bursts)
    6. decode      -> constrained candidate chains (Viterbi / RS / CRC)
    7. evidence    -> cross-stage weighted evidence graph + leaderboard
    8. hypothesize -> bounded DAG search, live verification, ranked chains
    9. explain     -> WHY report for the top hypotheses
    10. export     -> JSON chain summaries under outputs/hypotheses/

Usage:
    python scripts/run_hypothesis_engine.py -f <signal.iq> --sample-rate 2500000
    python scripts/run_hypothesis_engine.py --synthetic            # QPSK burst
    python scripts/run_hypothesis_engine.py -f <frame.sigmf-data>  # Zenodo frame
"""

from __future__ import annotations

import argparse
import sys
import time
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

from app.evidence import build_evidence_graph
from app.hypothesis import (
    HypothesisEngineConfig,
    run_hypothesis_search,
    generate_why,
    generate_short_why,
    generate_hypothesis_report,
    generate_chain_json,
)

DEFAULT_OUT_DIR = ROOT / "outputs" / "hypotheses"


def make_synthetic_qpsk(n_samples: int = 16384, snr_db: float = 20.0,
                        seed: int = 7, sps: int = 16):
    """Synthesize a clean QPSK burst (raw bits, RRC-ish shaping) for demo."""
    rng = np.random.default_rng(seed)
    n_sym = n_samples // sps
    bits = rng.integers(0, 2, n_sym * 2)
    sym_map = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=np.complex64)
    symbols = sym_map[(bits[0::2] * 2 + bits[1::2]).astype(int)] / np.sqrt(2.0)
    t = np.arange(n_sym) * sps
    iq = np.zeros(n_samples, dtype=np.complex64)
    for k, s in enumerate(symbols):
        iq[t[k]:(t[k] + sps)] = s
    from scipy.signal import firwin, lfilter
    taps = firwin(63, 0.55, window="hann")
    iq = lfilter(taps, 1.0, iq).astype(np.complex64)
    iq /= np.sqrt(np.mean(np.abs(iq) ** 2))
    noise_pow = 10.0 ** (-snr_db / 10.0)
    noise = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    noise *= np.sqrt(noise_pow / 2.0)
    return (iq + noise.astype(np.complex64)).astype(np.complex64)


def make_synthetic_signal(sample_rate: float = 2.0e6) -> ComplexSignal:
    data = make_synthetic_qpsk()
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
    ap.add_argument("-o", "--out-dir", default=DEFAULT_OUT_DIR,
                    help="Output directory for JSON summaries")
    ap.add_argument("--top-modulations", type=int, default=3,
                    help="Number of modulation candidates to expand (default 3)")
    ap.add_argument("--sps", default=None,
                    help="Comma-separated samples-per-symbol grid (default 1,2,4)")
    ap.add_argument("--max-chains", type=int, default=None,
                    help="Bound total chains evaluated (default 200)")
    ap.add_argument("--no-live", action="store_true",
                    help="Skip live demodulation/decoding (score from evidence only)")
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

    print("\n[5] evidence graph")
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
    for h in graph.top_hypotheses(k=6):
        print(f"    {h.kind:13s} {h.name[:48]:48s} score={h.score:.3f} "
              f"quality={h.quality} items={h.num_items}")

    # bounded hypothesis-engine config built from CLI switches
    config = HypothesisEngineConfig(top_modulations=args.top_modulations)
    if args.sps:
        config.sps_candidates = [float(x) for x in args.sps.split(",")]
    if args.max_chains:
        config.max_chains = args.max_chains
    if args.no_live:
        config.run_demodulation = False
        config.run_decoding = False

    print(f"\n[6] hypothesis engine (top {config.top_modulations} modulation(s), "
          f"sps={config.sps_candidates}, max_chains={config.max_chains})")
    t0 = time.perf_counter()
    chains = run_hypothesis_search(sig, graph, config)
    dt = time.perf_counter() - t0
    print(f"    evaluated {len(chains)} chains in {dt:.1f}s")

    print("\n[7] ranked leaderboard")
    for i, c in enumerate(chains[:8]):
        print(f"    #{i+1:2d} {c.modulation:8s} sps={c.demod_params.samples_per_symbol:4.1f} "
              f"{c.deinterleaver!s:14s} {c.fec.mode:14s} {c.frame.frame_name:18s} "
              f"composite={c.composite_score:.3f} unc={c.score_uncertainty:.3f}")
        print(f"        {generate_short_why(c)}")

    print("\n[8] top hypothesis WHY")
    print(generate_why(chains[0], graph))

    out_dir = Path(args.out_dir)
    chains_json = generate_chain_json(chains, graph)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hypotheses.json"
    out_path.write_text(
        __import__("json").dumps({
            "filename": sig.metadata.filename,
            "top_modulations": args.top_modulations,
            "num_chains": len(chains),
            "chains": chains_json,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\n[9] exported\n    JSON: {out_path}")

    report = generate_hypothesis_report(chains, graph)
    report_path = out_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"    Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())