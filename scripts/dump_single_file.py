#!/usr/bin/env python3
"""Dump the full pipeline output for ONE signal file into a directory.

Runs load -> normalize -> detection -> characterization -> classification ->
demodulation -> decoding -> evidence graph -> hypothesis engine for a single
.IQ/.WAV/.sigmf-data file and writes everything into an output directory so an
analyst can inspect exactly what the system decides and why.

Usage:
    python scripts/dump_single_file.py -f <signal.iq> [--sample-rate HZ] [-o test]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    generate_hypothesis_report,
    generate_chain_json,
)


def _to_jsonable(obj):
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.complexfloating):
        return [obj.real, obj.imag]
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return _to_jsonable(obj.to_dict())
    return str(obj)


def run_pipeline(sig):
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
    demod = demodulate_signal(sig, modulation=mod.top_candidate, max_symbols=4096)
    decode = []
    if demod.bits:
        decode = search_decode(
            demod.bits, llrs=(demod.llrs_sample or None),
            use_soft_viterbi=True, top_k=10,
        )

    graph = build_evidence_graph(
        snr=snr, bandwidth=bw, noise_floor=nf, segmentation=seg,
        features=fv, symbol_rate=sym, freq_offset=cfo,
        mod_classification=mod, demodulation=demod, decoding=decode,
        metadata={"filename": sig.metadata.filename,
                  "sample_rate": sig.metadata.sample_rate,
                  "center_frequency": sig.metadata.center_frequency},
    )

    config = HypothesisEngineConfig(top_modulations=3)
    t0 = time.perf_counter()
    chains = run_hypothesis_search(sig, graph, config)
    search_time = time.perf_counter() - t0

    return dict(snr=snr, bandwidth=bw, noise_floor=nf, segmentation=seg,
                features=fv, symbol_rate=sym, freq_offset=cfo,
                modulation=mod, demodulation=demod, decoding=decode,
                graph=graph, chains=chains, search_time=search_time,
                metadata=dict(filename=sig.metadata.filename,
                              sample_rate=sig.metadata.sample_rate,
                              center_frequency=sig.metadata.center_frequency,
                              data_type=sig.metadata.data_type,
                              num_samples=sig.metadata.num_samples,
                              provenance=[p.operation for p in sig.provenance]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-f", "--file", required=True, help="Signal file (.iq/.wav/.sigmf-data)")
    ap.add_argument("--sample-rate", type=float, default=None, help="Override sample rate (Hz)")
    ap.add_argument("-o", "--out-dir", default="test", help="Output directory (default: test)")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.file}")
    sig = load_signal(args.file, sample_rate=args.sample_rate)
    print(f"  {sig.metadata.data_type}  {sig.metadata.num_samples} samples  "
          f"fs={sig.metadata.sample_rate}")

    p = run_pipeline(sig)

    data = {
        "input": p["metadata"],
        "detection": {
            "snr": _to_jsonable(p["snr"]),
            "bandwidth": _to_jsonable(p["bandwidth"]),
            "noise_floor": _to_jsonable(p["noise_floor"]),
            "segments": len(p["segmentation"].segments),
        },
        "characterization": {
            "features": _to_jsonable(p["features"]),
            "symbol_rate": _to_jsonable(p["symbol_rate"]),
            "freq_offset": _to_jsonable(p["freq_offset"]),
        },
        "modulation": {
            "top": p["modulation"].top_candidate,
            "confidence": p["modulation"].confidence,
            "candidates": _to_jsonable(p["modulation"].candidates),
        },
        "demodulation": _to_jsonable(p["demodulation"]),
        "decoding": _to_jsonable(p["decoding"]),
        "evidence_graph": {"top_hypotheses": _to_jsonable(
            [h for h in p["graph"].top_hypotheses(k=10)]
        )},
        "hypothesis_engine": {
            "search_time_s": round(p["search_time"], 2),
            "num_chains": len(p["chains"]),
            "chains": generate_chain_json(p["chains"], p["graph"]),
        },
    }

    json_path = out_dir / "output.json"
    json_path.write_text(json.dumps(data, indent=2, default=_to_jsonable),
                         encoding="utf-8")
    print(f"Wrote {json_path}")

    lines = []
    lines.append(f"FILE: {p['metadata']['filename']}")
    lines.append(f"FORMAT: {p['metadata']['data_type']}  "
                 f"{p['metadata']['num_samples']} samples  "
                 f"fs={p['metadata']['sample_rate']}  "
                 f"fc={p['metadata']['center_frequency']}\n")
    lines.append("=== DETECTION ===")
    lines.append(f"  SNR           = {p['snr'].snr_db:.2f} dB ({p['snr'].method})")
    lines.append(f"  Bandwidth     = {p['bandwidth'].bandwidth_hz:.1f} Hz ({p['bandwidth'].method})")
    lines.append(f"  Noise floor   = {p['noise_floor'].noise_floor_db:.2f} dBFS")
    lines.append(f"  Segments      = {len(p['segmentation'].segments)}")
    lines.append("\n=== CHARACTERIZATION ===")
    lines.append(f"  Symbol rate   = {p['symbol_rate'].symbol_rate_hz:.1f} Hz "
                 f"(conf {p['symbol_rate'].confidence:.2f}, {p['symbol_rate'].method})")
    lines.append(f"  CFO           = {p['freq_offset'].offset_hz:.1f} Hz "
                 f"(conf {p['freq_offset'].confidence:.2f}, {p['freq_offset'].method})")
    lines.append(f"  C42_norm      = {p['features'].C42_norm:.3f}   "
                 f"C21_sym = {p['features'].C21_sym:.3f}   "
                 f"PAPR = {p['features'].papr_db:.2f} dB   "
                 f"kurt_amp = {p['features'].kurtosis_amplitude:.3f}   "
                 f"inst_freq_std_norm = {p['features'].instantaneous_freq_std_norm:.4f}")
    lines.append("\n=== MODULATION CLASSIFICATION ===")
    for c in p["modulation"].candidates[:6]:
        lines.append(f"  {c.modulation:8s} score={c.score:.3f} conf={c.confidence:.3f}")
    lines.append(f"\n  WHY: {p['modulation'].candidates[0].why}"[:400]
                 if p["modulation"].candidates[0].why else "  WHY: (none)")
    lines.append("\n=== DEMODULATION ===")
    d = p["demodulation"]
    if d.gated:
        lines.append(f"  GATED: {d.why}")
    else:
        evm = d.evm
        lines.append(f"  modulation={d.modulation}  symbols={d.num_symbols}  "
                     f"EVM RMS={evm.evm_rms_percent:.1f}% ({evm.evm_db:.1f} dB)"
                     if evm else f"  modulation={d.modulation}  symbols={d.num_symbols}  EVM=(none)")
        lines.append(f"  sps={d.samples_per_symbol:.2f}  CFO_rem={d.freq_offset_hz:.1f}  "
                     f"timing_off={d.timing_offset_samples:.2f}  "
                     f"phase_res={d.residual_phase_rad:.4f} rad  alpha={d.rolloff_alpha:.2f}")
        lines.append(f"  bits={len(d.bits)}  why={d.why}")
    lines.append("\n=== DECODING LEADERBOARD (search_decode) ===")
    if p["decoding"]:
        for i, res in enumerate(p["decoding"][:5]):
            lines.append(f"  #{i+1} spec={res.spec_name} ok={res.ok} "
                         f"viterbi_errs={res.num_viterbi_errors} "
                         f"rs_corr={res.num_rs_symbols_corrected} "
                         f"crc_valid={res.crc_valid} frame_offset={res.frame_offset}")
    else:
        lines.append("  (no bits to decode)")
    lines.append("\n=== EVIDENCE GRAPH TOP HYPOTHESES ===")
    for h in p["graph"].top_hypotheses(k=6):
        lines.append(f"  {h.kind:13s} {h.name[:48]:48s} score={h.score:.3f} "
                     f"quality={h.quality} items={h.num_items}")
    lines.append("\n=== HYPOTHESIS ENGINE (ranked chains) ===")
    lines.append(f"  evaluated {len(p['chains'])} chains in {p['search_time']:.1f}s")
    for i, c in enumerate(p["chains"][:8]):
        lines.append(f"  #{i+1:2d} {c.modulation:8s} sps={c.demod_params.samples_per_symbol:4.1f} "
                     f"{c.deinterleaver!s:14s} {c.fec.mode:14s} {c.frame.frame_name:18s} "
                     f"composite={c.composite_score:.3f} unc={c.score_uncertainty:.3f}")
    lines.append("\n=== TOP HYPOTHESIS WHY ===")
    lines.append(generate_why(p["chains"][0], p["graph"]))

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())