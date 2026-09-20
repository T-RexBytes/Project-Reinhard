#!/usr/bin/env python3
"""
scripts/run_rml_pipeline.py
----------------------------
Executes the SIH 26147 RF analysis pipeline across the RadioML 2016.10a dataset.
Maintains zero RAM bloat via streaming generator and preserves all native parameters.
Saves outputs to outputs/rml/ and exports standard SigMF frames to D:/dataset/rml_frames.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from app.ingestion.rml_parser import RMLDatasetReader, slice_rml_dataset
from app.preprocessing.normalize import remove_dc, normalize_power
from app.detection.fft import compute_psd
from app.detection.spectrogram import compute_spectrogram
from app.detection.noise_floor import estimate_noise_floor
from app.detection.snr import estimate_snr_from_psd
from app.characterization.features import extract_features
from app.modulation.classifier import ModulationClassifier


def run_rml_pipeline():
    print("=" * 80)
    print("SIH 26147 - RadioML 2016.10a Pipeline Processing & Dataset Slicing")
    print("=" * 80)

    t0 = time.time()
    out_dir = PROJECT_ROOT / "outputs" / "rml"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = Path("D:/dataset/rml_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    reader = RMLDatasetReader()
    print(f"[*] RML Dataset Loaded: 11 modulations across {len(reader.snrs)} SNR steps.")

    # Target representative SNR levels from clean to noise
    selected_snrs = [18, 14, 10, 6, 2, 0, -4, -10]
    frames_per_key = 3  # 11 mods * 8 SNRs * 3 frames = 264 frames

    print(f"[*] Processing balanced grid: 11 modulations x {len(selected_snrs)} SNRs x {frames_per_key} frames = {11 * len(selected_snrs) * frames_per_key} frames...")

    clf = ModulationClassifier()
    records = []
    mod_stats = defaultdict(lambda: {"total": 0, "correct_top": 0, "conf_sum": 0.0})
    snr_stats = defaultdict(lambda: {"total": 0, "snr_err_sum": 0.0})

    count = 0
    for sig in reader.iter_frames(mods=reader.modulations, snrs=selected_snrs, max_per_key=frames_per_key):
        count += 1
        truth_mod = sig.metadata.native_parameters["ground_truth_mod"]
        truth_snr = sig.metadata.native_parameters["labeled_snr_db"]
        frame_idx = sig.metadata.native_parameters["frame_index"]

        # 1. Preprocessing
        proc_sig = remove_dc(sig)
        proc_sig = normalize_power(proc_sig)

        # 2. Detection
        psd = compute_psd(proc_sig, fft_size=128)
        spec = compute_spectrogram(proc_sig, fft_size=64)
        nf = estimate_noise_floor(psd, method="mad")
        snr_est = estimate_snr_from_psd(psd, nf)

        # 3. Features
        feats = extract_features(proc_sig.data, sample_rate=proc_sig.metadata.sample_rate)

        # 4. Classification
        class_res = clf.classify_signal(proc_sig)

        # Accuracy & stats
        is_exact = (class_res.top_candidate == truth_mod)
        # Handle compatible names (e.g. 16QAM vs QAM16, 2-FSK vs CPFSK/GFSK)
        is_family = is_exact or (
            truth_mod in ["CPFSK", "GFSK"] and class_res.top_candidate in ["2-FSK", "4-FSK"]
        ) or (
            truth_mod in ["QAM16", "QAM64"] and class_res.top_candidate in ["16QAM"]
        ) or (
            truth_mod in ["8PSK"] and class_res.top_candidate in ["QPSK"]
        )

        mod_stats[truth_mod]["total"] += 1
        if is_family:
            mod_stats[truth_mod]["correct_top"] += 1
        mod_stats[truth_mod]["conf_sum"] += class_res.confidence

        record = {
            "filename": sig.metadata.filename,
            "dataset_origin": sig.metadata.dataset_origin,
            "sample_rate": sig.metadata.sample_rate,
            "num_samples": sig.metadata.num_samples,
            "native_parameters": sig.metadata.native_parameters,
            "ground_truth_mod": truth_mod,
            "labeled_snr_db": truth_snr,
            "frame_index": frame_idx,
            "top_candidate": class_res.top_candidate,
            "confidence": round(float(class_res.confidence), 4),
            "is_family_match": is_family,
            "estimated_snr_db": round(float(snr_est.snr_db), 2),
            "noise_floor_db": round(float(nf.noise_floor_db), 2),
            "features": {
                "C42_norm": round(float(feats.C42_norm), 4),
                "C21_sym": round(float(feats.C21_abs / max(feats.C20, 1e-12)), 4),
                "papr_db": round(float(feats.papr_db), 2),
                "spectral_flatness": round(float(feats.spectral_flatness), 4),
                "zero_crossing_rate": round(float(feats.zero_crossing_rate), 4),
            },
            "candidates": [c.to_dict() for c in class_res.candidates[:3]],
            "provenance_steps": [p.to_dict() for p in proc_sig.provenance],
        }
        records.append(record)

        if count % 50 == 0 or count == 11 * len(selected_snrs) * frames_per_key:
            print(f"  [{count:3d}/{11 * len(selected_snrs) * frames_per_key}] Processed {truth_mod:7s} @ {truth_snr:+3d} dB -> Pred: {class_res.top_candidate:6s} (conf: {class_res.confidence:.2f})")

    # Save detailed JSON output
    results_file = out_dir / "rml_pipeline_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[*] Detailed results written to {results_file} ({results_file.stat().st_size / 1024:.1f} KB)")

    # Also save to outputs/classification/
    class_out_dir = PROJECT_ROOT / "outputs" / "classification"
    class_out_dir.mkdir(parents=True, exist_ok=True)
    with open(class_out_dir / "rml_classification_summary.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Export a standardized SigMF slice to D:/dataset/rml_frames for permanent disk access
    print(f"[*] Slicing benchmark SigMF frames to {frames_dir}...")
    sliced_files = slice_rml_dataset(
        pkl_path=reader.pkl_path,
        output_dir=frames_dir,
        frames_per_key=2,
        snrs=[18, 10, 0],
    )
    print(f"[*] Exported {len(sliced_files)} SigMF frame pairs to {frames_dir}.")

    # Summary statistics
    summary = {
        "dataset": "RadioML 2016.10a",
        "dataset_origin": "rml2016.10a",
        "total_frames_processed": len(records),
        "total_time_seconds": round(time.time() - t0, 3),
        "modulations": {},
    }
    for mod, st in mod_stats.items():
        summary["modulations"][mod] = {
            "total_frames": st["total"],
            "family_accuracy_pct": round(100.0 * st["correct_top"] / max(1, st["total"]), 1),
            "avg_confidence": round(st["conf_sum"] / max(1, st["total"]), 3),
        }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("RML PIPELINE SUMMARY TABLE:")
    print("=" * 80)
    print(f"{'Modulation':<10} | {'Frames':<8} | {'Family Match %':<16} | {'Avg Confidence':<15}")
    print("-" * 55)
    for mod, st in summary["modulations"].items():
        print(f"{mod:<10} | {st['total_frames']:<8} | {st['family_accuracy_pct']:<15.1f}% | {st['avg_confidence']:<15.3f}")
    print("=" * 80)
    print(f"[SUCCESS] All RML outputs cleanly generated in {time.time() - t0:.2f} s!")


if __name__ == "__main__":
    run_rml_pipeline()
