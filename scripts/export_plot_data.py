#!/usr/bin/env python3
"""Export pipeline outputs for plotting — lean, no storage flood.

Saves every stage output till bandwidth/segmentation for a small
representative subset (not the full 33k files).

Outputs per file are split into:
  - psd.npz          -> frequencies, psd_db, psd_linear  (for spectrum plot)
  - spectrogram.npz  -> frequencies, time_bins, spectrogram_db (for waterfall)
  - iq_sample.npz    -> first 4096 raw + normalized IQ samples (for time/constellation plot)
  - metrics.json     -> noise_floor, snr, bandwidth, segments (scalars + segments)
  - metadata.json    -> signal metadata + provenance

Plus top-level summary.csv / summary.json for aggregate plots.

Usage:
  python scripts/export_plot_data.py                          # default: 3 per signal
  python scripts/export_plot_data.py --per-signal 5           # 5 per signal
  python scripts/export_plot_data.py --out outputs/plot_data  # custom out dir
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Ensure app is importable when run as `python scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.dataset.loader import DatasetScanner, TaskType, SignalType, SplitType
from app.preprocessing.normalize import full_normalize
from app.detection.fft import compute_psd
from app.detection.spectrogram import compute_spectrogram
from app.detection.noise_floor import estimate_noise_floor
from app.detection.segmentation import build_segments
from app.detection.snr import estimate_snr_from_psd, estimate_snr_m2m4
from app.detection.bandwidth import estimate_bandwidth

# ---------------------------------------------------------------------------
# Representative selection — lean but diverse
# ---------------------------------------------------------------------------
# Pick indices spread across the range to capture variation without saving all 33k.
# 3 per main signal covers low/mid/high; sep/frame add 1-2 each.
DEFAULT_SELECTION = [
    # demod_train — main signals
    (TaskType.DEMOD, SignalType.COMMSIGNAL2, SplitType.TRAIN, 0),
    (TaskType.DEMOD, SignalType.COMMSIGNAL2, SplitType.TRAIN, 500),
    (TaskType.DEMOD, SignalType.COMMSIGNAL2, SplitType.TRAIN, 1000),
    (TaskType.DEMOD, SignalType.COMMSIGNAL3, SplitType.TRAIN, 0),
    (TaskType.DEMOD, SignalType.COMMSIGNAL3, SplitType.TRAIN, 500),
    (TaskType.DEMOD, SignalType.COMMSIGNAL3, SplitType.TRAIN, 1000),
    (TaskType.DEMOD, SignalType.EMISIGNAL1, SplitType.TRAIN, 0),
    (TaskType.DEMOD, SignalType.EMISIGNAL1, SplitType.TRAIN, 500),
    (TaskType.DEMOD, SignalType.EMISIGNAL1, SplitType.TRAIN, 1000),
    # sep_train — mixed signals
    (TaskType.SEPARATION, SignalType.COMMSIGNAL3, SplitType.TRAIN, 0),
    (TaskType.SEPARATION, SignalType.EMISIGNAL1, SplitType.TRAIN, 0),
    # train_frame — longer captures
    (TaskType.FRAME, SignalType.COMMSIGNAL2, SplitType.TRAIN, 0),
    (TaskType.FRAME, SignalType.COMMSIGNAL3, SplitType.TRAIN, 0),
    (TaskType.FRAME, SignalType.EMISIGNAL1, SplitType.TRAIN, 0),
]

IQ_SAMPLE_LEN = 4096  # only first 4k samples for time/const plot (not full 40k)


def process_one(scanner: DatasetScanner, task, signal, split, index, out_root: Path):
    entries = scanner.filter(task=task, signal=signal, split=split)
    target = next((e for e in entries if e.index == index), None)
    if target is None:
        print(f"  SKIP  {signal.value} {task.value} #{index:04d} — not found")
        return None

    sig = scanner.load_entry(target)
    sig_norm = full_normalize(sig)

    # --- PSD ---
    psd = compute_psd(sig_norm, fft_size=1024)

    # --- Spectrogram ---
    spec = compute_spectrogram(sig_norm, fft_size=512, max_time_bins=128)

    # --- Noise floor, segmentation, SNR, bandwidth ---
    nf = estimate_noise_floor(psd, method="mad")
    seg = build_segments(psd, nf, sig_norm)
    snr_psd = estimate_snr_from_psd(psd, nf)
    try:
        snr_m2m4 = estimate_snr_m2m4(sig_norm.data)
        snr_m2m4_db = float(snr_m2m4.snr_db)
    except Exception:
        snr_m2m4_db = None
    bw_3db = estimate_bandwidth(psd, method="3db")
    bw_frac = estimate_bandwidth(psd, method="fractional")

    # --- Output folder per file ---
    folder_name = f"{signal.value}_{task.value}_{split.value}_{index:04d}"
    out_dir = out_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # psd.npz
    np.savez_compressed(
        out_dir / "psd.npz",
        frequencies=psd.frequencies,
        psd_db=psd.psd_db,
        psd_linear=psd.psd_linear if hasattr(psd, "psd_linear") else np.power(10, psd.psd_db / 10),
    )

    # spectrogram.npz
    np.savez_compressed(
        out_dir / "spectrogram.npz",
        frequencies=spec.frequencies,
        time_bins=spec.time_bins,
        spectrogram_db=spec.spectrogram_db,
    )

    # iq_sample.npz — tiny, only first 4k for plotting
    n = min(IQ_SAMPLE_LEN, len(sig.data))
    np.savez_compressed(
        out_dir / "iq_sample.npz",
        iq_raw=sig.data[:n],
        iq_normalized=sig_norm.data[:n],
        sample_rate=np.array(sig.metadata.sample_rate or 25e6),
    )

    # metrics.json — scalars + segments (plottable overlay data)
    metrics = {
        "file": target.data_path.name,
        "task": task.value,
        "signal": signal.value,
        "split": split.value,
        "index": index,
        "num_samples": sig.num_samples,
        "duration_ms": sig.duration_seconds * 1e3 if sig.duration_seconds else None,
        "noise_floor": {"method": nf.method, "noise_floor_db": float(nf.noise_floor_db)},
        "snr": {
            "psd_based_db": float(snr_psd.snr_db),
            "psd_signal_power_db": float(snr_psd.signal_power_db),
            "psd_noise_power_db": float(snr_psd.noise_power_db),
            "m2m4_db": snr_m2m4_db,
        },
        "bandwidth": {
            "bw_3db_hz": float(bw_3db.bandwidth_hz),
            "bw_3db_lower_hz": float(bw_3db.lower_3db_hz) if bw_3db.lower_3db_hz is not None else None,
            "bw_3db_upper_hz": float(bw_3db.upper_3db_hz) if bw_3db.upper_3db_hz is not None else None,
            "bw_frac_hz": float(bw_frac.bandwidth_hz),
            "occupancy_percent": float(bw_frac.occupancy_percent) if hasattr(bw_frac, "occupancy_percent") else None,
        },
        "segmentation": {
            "num_segments": seg.num_segments,
            "detection_threshold_db": float(seg.detection_threshold_db) if hasattr(seg, "detection_threshold_db") else None,
            "segments": [s.to_dict() for s in seg.segments],
        },
        "psd_stats": {
            "max_db": float(psd.psd_db.max()),
            "min_db": float(psd.psd_db.min()),
            "mean_db": float(psd.psd_db.mean()),
            "freq_range_khz": [float(psd.frequencies[0] / 1e3), float(psd.frequencies[-1] / 1e3)],
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # metadata.json
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "metadata": sig.metadata.to_dict(),
                "provenance": [p.to_dict() for p in sig_norm.provenance],
                "sigmf": sig.metadata.sigmf_metadata,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"  OK    {folder_name} -> {out_dir.name}/  (SNR {snr_psd.snr_db:.1f} dB, {seg.num_segments} segs, BW {bw_3db.bandwidth_hz/1e3:.0f} kHz)")
    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export pipeline outputs for plotting")
    parser.add_argument("--dataset", default=r"D:\dataset", help="Dataset root")
    parser.add_argument("--out", default="outputs/plot_data", help="Output directory")
    parser.add_argument("--per-signal", type=int, default=None, help="Override: N files per signal (default uses curated 14)")
    args = parser.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = Path(__file__).resolve().parent.parent / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    scanner = DatasetScanner(args.dataset)
    scanner.scan()
    print(f"Dataset: {args.dataset} — {len(scanner.filter())} files indexed")
    print(f"Exporting to: {out_root}")

    selection = DEFAULT_SELECTION
    if args.per_signal is not None:
        # Build simple per-signal selection
        selection = []
        for sig in [SignalType.COMMSIGNAL2, SignalType.COMMSIGNAL3, SignalType.EMISIGNAL1]:
            for idx in [0, 500, 1000][: args.per_signal]:
                selection.append((TaskType.DEMOD, sig, SplitType.TRAIN, idx))

    all_metrics = []
    for task, signal, split, idx in selection:
        m = process_one(scanner, task, signal, split, idx, out_root)
        if m:
            all_metrics.append(m)

    # --- Top-level summary.csv ---
    csv_path = out_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "file", "task", "signal", "split", "index",
                "snr_psd_db", "snr_m2m4_db", "noise_floor_db",
                "bw_3db_khz", "bw_frac_khz", "num_segments", "psd_max_db",
            ],
        )
        w.writeheader()
        for m in all_metrics:
            w.writerow({
                "file": m["file"],
                "task": m["task"],
                "signal": m["signal"],
                "split": m["split"],
                "index": m["index"],
                "snr_psd_db": round(m["snr"]["psd_based_db"], 2),
                "snr_m2m4_db": round(m["snr"]["m2m4_db"], 2) if m["snr"]["m2m4_db"] is not None else "",
                "noise_floor_db": round(m["noise_floor"]["noise_floor_db"], 2),
                "bw_3db_khz": round(m["bandwidth"]["bw_3db_hz"] / 1e3, 2),
                "bw_frac_khz": round(m["bandwidth"]["bw_frac_hz"] / 1e3, 2),
                "num_segments": m["segmentation"]["num_segments"],
                "psd_max_db": round(m["psd_stats"]["max_db"], 2),
            })

    # --- summary.json ---
    with open(out_root / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # --- README ---
    readme = f"""# Plot Data — Pipeline Outputs (till segmentation/bandwidth)

Generated from: {args.dataset}
Files exported: {len(all_metrics)} (curated subset, not full 33k — to save storage)
Total size: ~{sum(f.stat().st_size for f in out_root.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB

## Structure
```
plot_data/
  summary.csv          <- one row per file, for bar/scatter plots
  summary.json         <- same as above, full detail
  <signal>_<task>_<split>_<idx>/
    psd.npz            -> frequencies, psd_db, psd_linear  (plot: freq vs PSD)
    spectrogram.npz    -> frequencies, time_bins, spectrogram_db (plot: waterfall)
    iq_sample.npz      -> iq_raw, iq_normalized, sample_rate (plot: time domain / constellation)
    metrics.json       -> noise_floor, snr, bandwidth, segments, psd_stats
    metadata.json      -> SigMF + provenance
```

## Quick plotting snippets

```python
import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

base = Path("outputs/plot_data/CommSignal2_demod_train_0000")

# 1) PSD + segments overlay
d = np.load(base / "psd.npz")
with open(base / "metrics.json") as f: m = json.load(f)
plt.plot(d["frequencies"]/1e3, d["psd_db"])
plt.axhline(m["noise_floor"]["noise_floor_db"], ls="--", label="noise floor")
for s in m["segmentation"]["segments"]:
    plt.axvspan(s["start_freq_hz"]/1e3, s["end_freq_hz"]/1e3, alpha=0.2)
plt.xlabel("Freq (kHz)"); plt.ylabel("PSD (dB)"); plt.legend(); plt.show()

# 2) Waterfall
s = np.load(base / "spectrogram.npz")
plt.imshow(s["spectrogram_db"], aspect="auto", origin="lower",
           extent=[s["time_bins"][0]*1e3, s["time_bins"][-1]*1e3,
                   s["frequencies"][0]/1e3, s["frequencies"][-1]/1e3])
plt.xlabel("Time (ms)"); plt.ylabel("Freq (kHz)"); plt.colorbar(label="dB"); plt.show()

# 3) Summary bar chart
import pandas as pd
df = pd.read_csv("outputs/plot_data/summary.csv")
df.plot(x="file", y=["snr_psd_db","noise_floor_db"], kind="bar"); plt.show()
```
"""
    (out_root / "README.md").write_text(readme)

    print(f"\nDone. Exported {len(all_metrics)} files -> {out_root}")
    print(f"  - summary.csv / summary.json")
    print(f"  - {len(all_metrics)} folders, each with psd.npz, spectrogram.npz, iq_sample.npz, metrics.json, metadata.json")
    total_mb = sum(f.stat().st_size for f in out_root.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  Total size: {total_mb:.2f} MB (lean, not full 33k)")


if __name__ == "__main__":
    main()
