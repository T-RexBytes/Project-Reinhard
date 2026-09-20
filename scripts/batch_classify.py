"""
scripts/batch_classify.py
-------------------------
Headless CLI tool for batch modulation classification and verification across
dataset directories (MIT RF Challenge data0, Zenodo ATA RFI frames, etc.).

Outputs:
  - Terminal summary table
  - outputs/classification/batch_results.csv
  - outputs/classification/batch_results.json

Usage:
  python scripts/batch_classify.py --input-dir D:\\dataset\\data0 --max-files 20
  python scripts/batch_classify.py --input-dir D:\\dataset\\zenodo_frames --max-files 10 --demodulate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure repository root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ingestion.parser import load_signal
from app.models import DataType
from app.modulation.classifier import classify_signal, ModulationClassifier
from app.demodulation.slicer import demodulate_signal


def scan_input_dir(directory: Path, max_files: int = 50) -> list[Path]:
    """Find signal files in input directory (recursively searching subdirectories)."""
    files = []
    # Prioritize SigMF data files
    for p in directory.rglob("*.sigmf-data"):
        files.append(p)
        if len(files) >= max_files:
            return files

    # Fallback to standard binary extensions
    for ext in ("*.cf32", "*.c32", "*.ci16", "*.i16", "*.iq", "*.wav"):
        for p in directory.rglob(ext):
            if p not in files:
                files.append(p)
                if len(files) >= max_files:
                    return files

    return files[:max_files]


def run_batch_classification(
    input_dir: str,
    max_files: int = 30,
    output_dir: str = "outputs/classification",
    data_type: Optional[str] = None,
    sample_rate: Optional[float] = None,
    run_demod: bool = False,
):
    inp_path = Path(input_dir)
    if not inp_path.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    signal_files = scan_input_dir(inp_path, max_files=max_files)
    if not signal_files:
        print(f"No valid signal files found in {input_dir}")
        return

    print(f"\n================================================================================")
    print(f"   SIH 26147 BATCH MODULATION CLASSIFICATION & HYPOTHESIS RANKING")
    print(f"================================================================================")
    print(f"Input Directory : {inp_path}")
    print(f"Total Files     : {len(signal_files)}")
    print(f"Demodulation    : {'ENABLED' if run_demod else 'DISABLED'}")
    print(f"Output Directory: {out_path}\n")

    results = []
    clf = ModulationClassifier()

    print(f"{'Filename':<32} {'SR (MHz)':<10} {'SNR (dB)':<10} {'Top Mod':<10} {'Conf %':<8} {'EVM %':<8} {'Top Candidate Why'}")
    print("-" * 115)

    t0 = time.time()
    for p in signal_files:
        dt = DataType(data_type) if data_type else None
        try:
            sig = load_signal(str(p), data_type=dt, sample_rate=sample_rate)
        except Exception as e:
            print(f"{p.name:<32} [ERROR LOADING: {e}]")
            continue

        res = clf.classify_signal(sig, max_samples=65536)

        evm_val = None
        evm_db = None
        if run_demod and res.top_candidate not in ("Noise/CW", "Unknown"):
            try:
                demod_res = demodulate_signal(sig, modulation=res.top_candidate, max_symbols=512)
                evm_val = demod_res.evm.evm_rms_percent
                evm_db = demod_res.evm.evm_db
            except Exception:
                pass

        sr_mhz = (sig.metadata.sample_rate or 1.0) / 1e6
        snr_str = f"{res.snr_db:.1f}" if res.snr_db is not None else "N/A"
        conf_str = f"{res.confidence * 100:.1f}%"
        evm_str = f"{evm_val:.1f}%" if evm_val is not None else "N/A"

        top_why = res.candidates[0].why if res.candidates else ""
        # Shorten why for table display
        why_short = (top_why[:45] + "...") if len(top_why) > 45 else top_why

        print(f"{sig.metadata.filename:<32} {sr_mhz:<10.2f} {snr_str:<10} {res.top_candidate:<10} {conf_str:<8} {evm_str:<8} {why_short}")

        row = {
            "filename": sig.metadata.filename,
            "filepath": str(p),
            "data_type": sig.metadata.data_type.value,
            "sample_rate_hz": sig.metadata.sample_rate,
            "center_frequency_hz": sig.metadata.center_frequency,
            "num_samples": sig.num_samples,
            "snr_db": res.snr_db,
            "top_candidate": res.top_candidate,
            "confidence": res.confidence,
            "evm_rms_percent": evm_val,
            "evm_db": evm_db,
            "candidates": [c.to_dict() for c in res.candidates],
            "features_summary": res.features_summary,
        }
        results.append(row)

    elapsed = time.time() - t0
    print("-" * 115)
    print(f"Processed {len(results)} signals in {elapsed:.2f}s ({elapsed / max(1, len(results)):.3f}s / signal)\n")

    # Save CSV summary
    csv_path = out_path / "batch_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename", "data_type", "sample_rate_hz", "center_frequency_hz",
            "snr_db", "top_candidate", "confidence", "evm_rms_percent",
            "score_BPSK", "score_QPSK", "score_16QAM", "score_2FSK", "score_4FSK", "score_Noise",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            c_dict = {c["modulation"]: c["score"] for c in r["candidates"]}
            writer.writerow({
                "filename": r["filename"],
                "data_type": r["data_type"],
                "sample_rate_hz": r["sample_rate_hz"],
                "center_frequency_hz": r["center_frequency_hz"],
                "snr_db": r["snr_db"],
                "top_candidate": r["top_candidate"],
                "confidence": r["confidence"],
                "evm_rms_percent": r["evm_rms_percent"],
                "score_BPSK": c_dict.get("BPSK", 0.0),
                "score_QPSK": c_dict.get("QPSK", 0.0),
                "score_16QAM": c_dict.get("16QAM", 0.0),
                "score_2FSK": c_dict.get("2-FSK", 0.0),
                "score_4FSK": c_dict.get("4-FSK", 0.0),
                "score_Noise": c_dict.get("Noise/CW", 0.0),
            })

    # Save JSON details
    json_path = out_path / "batch_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Summary written to: {csv_path}")
    print(f"Detailed results written to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Modulation Classification CLI")
    parser.add_argument("--input-dir", "-i", type=str, default=r"D:\dataset\data0", help="Directory containing signal files")
    parser.add_argument("--max-files", "-n", type=int, default=15, help="Maximum number of files to process")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/classification", help="Output directory for reports")
    parser.add_argument("--data-type", "-t", type=str, default=None, help="Override data type (cf32, ci16, etc.)")
    parser.add_argument("--sample-rate", "-r", type=float, default=None, help="Sample rate in Hz")
    parser.add_argument("--demodulate", "-d", action="store_true", help="Run DSP demodulation and EVM calculation")

    args = parser.parse_args()
    run_batch_classification(
        input_dir=args.input_dir,
        max_files=args.max_files,
        output_dir=args.output_dir,
        data_type=args.data_type,
        sample_rate=args.sample_rate,
        run_demod=args.demodulate,
    )
