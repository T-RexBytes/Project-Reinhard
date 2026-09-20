#!/usr/bin/env python3
"""Batch analyze files from the MIT RF Challenge dataset.

Usage:
    python scripts/batch_analyze.py --task demod --signal CommSignal2 --split train --max 10
    python scripts/batch_analyze.py --task sep --signal CommSignal3 --split val --max 5
    python scripts/batch_analyze.py --stats
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dataset.loader import DatasetScanner, TaskType, SignalType, SplitType
from app.detection.fft import compute_psd
from app.detection.noise_floor import estimate_noise_floor
from app.detection.segmentation import build_segments
from app.detection.snr import estimate_snr_from_psd
from app.detection.bandwidth import estimate_bandwidth


def show_stats(scanner: DatasetScanner) -> None:
    stats = scanner.get_stats()
    print(f"\nDataset: {stats.root_path}")
    print(f"Total files: {stats.total_files}")
    print(f"Total size: {stats.total_size_gb} GB")
    print(f"Sample rate: {stats.sample_rate / 1e6} MHz")
    print(f"Samples per file: {stats.samples_per_file}")
    print(f"\nBy task: {json.dumps(stats.by_task, indent=2)}")
    print(f"By signal: {json.dumps(stats.by_signal, indent=2)}")
    print(f"By split: {json.dumps(stats.by_split, indent=2)}")


def analyze_entries(
    scanner: DatasetScanner,
    task: TaskType | None,
    signal: SignalType | None,
    split: SplitType | None,
    max_files: int,
    fft_size: int,
) -> list[dict]:
    entries = scanner.filter(task=task, signal=signal, split=split, max_entries=max_files)
    print(f"\nAnalyzing {len(entries)} files...\n")

    results = []
    for i, entry in enumerate(entries):
        t0 = time.time()
        try:
            sig = scanner.load_entry(entry)
            psd = compute_psd(sig, fft_size=fft_size)
            nf = estimate_noise_floor(psd, method="mad")
            snr = estimate_snr_from_psd(psd, nf)
            bw = estimate_bandwidth(psd, method="3db")
            seg = build_segments(psd, nf, sig)

            elapsed = time.time() - t0
            result = {
                "index": entry.index,
                "signal": entry.signal.value,
                "component": entry.component_type,
                "num_samples": sig.num_samples,
                "snr_db": round(snr.snr_db, 2),
                "bandwidth_hz": round(bw.bandwidth_hz, 0),
                "noise_floor_db": round(nf.noise_floor_db, 2),
                "elapsed_ms": round(elapsed * 1000, 1),
            }
            results.append(result)
            print(
                f"  [{i+1}/{len(entries)}] {entry.signal.value} #{entry.index:04d} | "
                f"SNR={snr.snr_db:.1f} dB | BW={bw.bandwidth_hz/1e3:.1f} kHz | "
                f"{elapsed*1000:.0f}ms"
            )
        except Exception as e:
            results.append({"index": entry.index, "signal": entry.signal.value, "error": str(e)})
            print(f"  [{i+1}/{len(entries)}] {entry.signal.value} #{entry.index:04d} | ERROR: {e}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch analyze RF dataset files")
    parser.add_argument("--dataset", default=r"D:\dataset", help="Dataset root path")
    parser.add_argument("--task", choices=["demod", "sep", "frame"], help="Filter by task type")
    parser.add_argument("--signal", choices=["CommSignal2", "CommSignal3", "EMISignal1"], help="Filter by signal type")
    parser.add_argument("--split", choices=["train", "val"], help="Filter by split")
    parser.add_argument("--max", type=int, default=10, help="Max files to analyze")
    parser.add_argument("--fft-size", type=int, default=1024, help="FFT size")
    parser.add_argument("--stats", action="store_true", help="Show dataset stats only")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    scanner = DatasetScanner(args.dataset)
    scanner.scan()

    if args.stats:
        show_stats(scanner)
        return

    task = TaskType(args.task) if args.task else None
    signal = SignalType(args.signal) if args.signal else None
    split = SplitType(args.split) if args.split else None

    results = analyze_entries(scanner, task, signal, split, args.max, args.fft_size)

    print(f"\n--- Summary ---")
    print(f"Processed: {len(results)} files")
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"Errors: {len(errors)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
