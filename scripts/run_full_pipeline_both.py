#!/usr/bin/env python3
"""Run full detection+characterization pipeline for BOTH datasets.

Datasets:
  - data0 : MIT RF Challenge at D:\\dataset\\data0  (40k samples @25MHz, 14 curated files)
  - zenodo: ATA RFI captures at D:\\dataset\\zenodo (6 tar .sigmf, 61.44MHz ci16)

Pipeline per file (till now, as before + characterization):
  ingestion -> full_normalize -> PSD(1024) -> spectrogram(512/128) ->
  noise_floor MAD -> segmentation -> SNR(PSD+M2M4) -> BW(3dB+frac) ->
  features(C42 etc) -> symbol_rate -> freq_offset

Outputs:
  outputs/data0/<file_folder>/{psd.npz,spectrogram.npz,iq_sample.npz,metrics.json,metadata.json}
  outputs/data0/summary.json/csv + README
  outputs/zenodo/<capture_folder>/{same}
  outputs/zenodo/summary.json/csv + README
"""

from __future__ import annotations

import csv
import json
import sys
import tarfile
import shutil
import warnings
from pathlib import Path
import time

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from app.ingestion.parser import load_signal
from app.dataset.loader import DatasetScanner, TaskType, SignalType, SplitType
from app.models import ComplexSignal, SignalMetadata, DataType, IQOrdering, ProvenanceStep
from app.preprocessing.normalize import full_normalize
from app.detection.fft import compute_psd
from app.detection.spectrogram import compute_spectrogram
from app.detection.noise_floor import estimate_noise_floor
from app.detection.segmentation import build_segments
from app.detection.snr import estimate_snr_from_psd, estimate_snr_m2m4
from app.detection.bandwidth import estimate_bandwidth
from app.characterization.features import extract_features
from app.characterization.symbol_rate import estimate_symbol_rate
from app.characterization.freq_offset import estimate_freq_offset
from app.modulation.classifier import ModulationClassifier

# ---------------------------------------------------------------------------
# Selection for data0 (same 14 as before)
# ---------------------------------------------------------------------------
DEFAULT_SELECTION = [
    (TaskType.DEMOD, SignalType.COMMSIGNAL2, SplitType.TRAIN, 0),
    (TaskType.DEMOD, SignalType.COMMSIGNAL2, SplitType.TRAIN, 500),
    (TaskType.DEMOD, SignalType.COMMSIGNAL2, SplitType.TRAIN, 1000),
    (TaskType.DEMOD, SignalType.COMMSIGNAL3, SplitType.TRAIN, 0),
    (TaskType.DEMOD, SignalType.COMMSIGNAL3, SplitType.TRAIN, 500),
    (TaskType.DEMOD, SignalType.COMMSIGNAL3, SplitType.TRAIN, 1000),
    (TaskType.DEMOD, SignalType.EMISIGNAL1, SplitType.TRAIN, 0),
    (TaskType.DEMOD, SignalType.EMISIGNAL1, SplitType.TRAIN, 500),
    (TaskType.DEMOD, SignalType.EMISIGNAL1, SplitType.TRAIN, 1000),
    (TaskType.SEPARATION, SignalType.COMMSIGNAL3, SplitType.TRAIN, 0),
    (TaskType.SEPARATION, SignalType.EMISIGNAL1, SplitType.TRAIN, 0),
    (TaskType.FRAME, SignalType.COMMSIGNAL2, SplitType.TRAIN, 0),
    (TaskType.FRAME, SignalType.COMMSIGNAL3, SplitType.TRAIN, 0),
    (TaskType.FRAME, SignalType.EMISIGNAL1, SplitType.TRAIN, 0),
]

IQ_SAMPLE_LEN = 4096
CHAR_MAX_SAMPLES = 65536  # for features/symbol/freq on large zenodo files

# ---------------------------------------------------------------------------
# Helpers: core pipeline
# ---------------------------------------------------------------------------

def pipeline_core(sig: ComplexSignal, sig_norm: ComplexSignal, out_dir: Path, file_label: str, extra_meta: dict | None = None):
    """Run PSD/spectrogram/seg/SNR/BW + characterization + modulation classification and save outputs."""
    # Compute PSD, Noise Floor, and Segmentation on the raw signal to preserve native physical power
    psd = compute_psd(sig, fft_size=1024)
    spec = compute_spectrogram(sig_norm, fft_size=512, max_time_bins=128)
    nf = estimate_noise_floor(psd, method="mad")
    seg = build_segments(psd, nf, sig)
    snr_psd = estimate_snr_from_psd(psd, nf)
    try:
        snr_m2m4 = estimate_snr_m2m4(sig.data)
        snr_m2m4_db = float(snr_m2m4.snr_db)
    except Exception:
        snr_m2m4_db = None
    bw_3db = estimate_bandwidth(psd, method="3db")
    bw_frac = estimate_bandwidth(psd, method="fractional")

    # --- Characterization (use raw signal to preserve native physical moments) ---
    iq_char = sig.data
    if len(iq_char) > CHAR_MAX_SAMPLES:
        iq_char = iq_char[:CHAR_MAX_SAMPLES]
    sample_rate = sig.metadata.sample_rate
    if sample_rate is None:
        raise ValueError(f"Signal {file_label} is missing required sample_rate metadata!")
    sr_est = sample_rate
    try:
        fv = extract_features(iq_char, sample_rate=sr_est, snr_db=snr_psd.snr_db)
    except Exception as e:
        print(f"    WARN features failed: {e}")
        fv = None
    try:
        bw_hint = bw_3db.bandwidth_hz if bw_3db else None
        sym = estimate_symbol_rate(iq_char, sample_rate=sr_est, bandwidth_hz=bw_hint)
    except Exception as e:
        print(f"    WARN symbol_rate failed: {e}")
        sym = None
    try:
        cfo = estimate_freq_offset(iq_char, sample_rate=sr_est)
    except Exception as e:
        print(f"    WARN freq_offset failed: {e}")
        cfo = None

    # --- Modulation Classification ---
    clf = ModulationClassifier()
    mod_res = clf.classify(
        fv,
        snr_db=snr_psd.snr_db,
        symbol_rate_hz=sym.symbol_rate_hz if sym else None,
        bandwidth_hz=bw_3db.bandwidth_hz if bw_3db else None,
        sample_rate=sr_est,
    ) if fv else None

    out_dir.mkdir(parents=True, exist_ok=True)

    # psd.npz
    np.savez_compressed(
        out_dir / "psd.npz",
        frequencies=psd.frequencies,
        rf_frequencies=psd.rf_frequencies if psd.rf_frequencies is not None else psd.frequencies,
        psd_db=psd.psd_db,
        psd_linear=psd.psd_linear,
    )
    # spectrogram.npz
    np.savez_compressed(
        out_dir / "spectrogram.npz",
        frequencies=spec.frequencies,
        time_bins=spec.time_bins,
        spectrogram_db=spec.spectrogram_db,
    )
    # iq_sample.npz — first 4k
    n = min(IQ_SAMPLE_LEN, len(sig.data))
    np.savez_compressed(
        out_dir / "iq_sample.npz",
        iq_raw=sig.data[:n],
        iq_normalized=sig_norm.data[:n],
        sample_rate=np.array(sig.metadata.sample_rate),
    )

    metrics = {
        "file": file_label,
        "task": extra_meta.get("task", "unknown") if extra_meta else "unknown",
        "signal": extra_meta.get("signal", "unknown") if extra_meta else "unknown",
        "split": extra_meta.get("split", "unknown") if extra_meta else "unknown",
        "index": extra_meta.get("index", -1) if extra_meta else -1,
        "data_type": sig.metadata.data_type.value,
        "num_samples": sig.num_samples,
        "duration_ms": sig.duration_seconds * 1e3 if sig.duration_seconds else None,
        "sample_rate_hz": sig.metadata.sample_rate,
        "center_frequency_hz": sig.metadata.center_frequency,
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
            "occupancy_percent": float(bw_frac.occupancy_percent) if hasattr(bw_frac, "occupancy_percent") and bw_frac.occupancy_percent is not None else None,
        },
        "segmentation": {
            "num_segments": seg.num_segments,
            "detection_threshold_db": float(seg.detection_threshold_db),
            "segments": [s.to_dict() for s in seg.segments],
        },
        "psd_stats": {
            "max_db": float(psd.psd_db.max()),
            "min_db": float(psd.psd_db.min()),
            "mean_db": float(psd.psd_db.mean()),
            "freq_range_khz": [float(psd.frequencies[0] / 1e3), float(psd.frequencies[-1] / 1e3)],
            "rf_freq_range_mhz": [
                float((psd.frequencies[0] + sig.metadata.center_frequency) / 1e6),
                float((psd.frequencies[-1] + sig.metadata.center_frequency) / 1e6)
            ] if sig.metadata.center_frequency is not None else None,
        },
        "characterization": {
            "features": fv.to_dict() if fv else None,
            "symbol_rate": sym.to_dict() if sym else None,
            "freq_offset": cfo.to_dict() if cfo else None,
            "char_samples_used": len(iq_char),
        },
        "modulation_classification": mod_res.to_dict() if mod_res else None,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "metadata": sig.metadata.to_dict(),
                "provenance": [p.to_dict() for p in sig_norm.provenance],
                "sigmf": sig.metadata.sigmf_metadata,
                "extra": extra_meta,
            },
            f,
            indent=2,
            default=str,
        )
    mod_summary = f"Mod={mod_res.top_candidate}({mod_res.confidence:.2f})" if mod_res else ""
    print(f"  OK {out_dir.name} SNR {snr_psd.snr_db:.1f}dB segs={seg.num_segments} BW={bw_3db.bandwidth_hz/1e3:.0f}kHz C42={fv.C42_norm:.3f} {mod_summary}" if fv else f"  OK {out_dir.name} SNR {snr_psd.snr_db:.1f}dB segs={seg.num_segments}")
    return metrics


# ---------------------------------------------------------------------------
# data0 processing
# ---------------------------------------------------------------------------

def process_data0(out_root: Path, smoke: bool = False):
    out_root.mkdir(parents=True, exist_ok=True)
    mode_label = "SMOKE TEST (2 files)" if smoke else "FULL RUN"
    print(f"\n{'='*70}\n  DATA0 [{mode_label}] — MIT RF Challenge @ D:\\dataset\\data0 -> {out_root}\n{'='*70}")
    scanner = DatasetScanner(r"D:\dataset\data0")
    scanner.scan()
    print(f"Indexed {len(scanner.filter())} files")
    
    selection = DEFAULT_SELECTION[:2] if smoke else DEFAULT_SELECTION
    all_metrics = []
    for task, signal, split, idx in selection:
        entries = scanner.filter(task=task, signal=signal, split=split)
        target = next((e for e in entries if e.index == idx), None)
        if target is None:
            print(f"  SKIP {signal.value} {task.value} #{idx:04d} not found")
            continue
        t0 = time.time()
        sig = scanner.load_entry(target)
        sig_norm = full_normalize(sig)
        folder_name = f"{signal.value}_{task.value}_{split.value}_{idx:04d}"
        out_dir = out_root / folder_name
        extra = {"task": task.value, "signal": signal.value, "split": split.value, "index": idx, "source": "data0", "data_path": str(target.data_path)}
        m = pipeline_core(sig, sig_norm, out_dir, target.data_path.name, extra)
        m["elapsed_sec"] = time.time() - t0
        all_metrics.append(m)
    write_summaries(out_root, all_metrics, source="data0 MIT RF Challenge (D:\\dataset\\data0)")
    return all_metrics


# ---------------------------------------------------------------------------
# zenodo processing (uses lightweight D:\dataset\zenodo_frames)
# ---------------------------------------------------------------------------

def process_zenodo(out_root: Path, smoke: bool = False):
    out_root.mkdir(parents=True, exist_ok=True)
    zen_frames_root = Path(r"D:\dataset\zenodo_frames")
    
    if not zen_frames_root.exists() or not any(zen_frames_root.iterdir()):
        raise FileNotFoundError("D:\\dataset\\zenodo_frames does not exist or is empty. Run scripts/slice_zenodo.py first.")

    capture_dirs = sorted([d for d in zen_frames_root.iterdir() if d.is_dir()])
    mode_label = "SMOKE TEST (2 files)" if smoke else f"FULL RUN ({len(capture_dirs)} captures)"
    print(f"\n{'='*70}\n  ZENODO [{mode_label}] — ATA RFI @ D:\\dataset\\zenodo_frames -> {out_root}\n  Found {len(capture_dirs)} capture folders\n{'='*70}")
    
    all_metrics = []
    
    if smoke:
        targets = []
        for cdir in capture_dirs[:2]:
            frame_files = sorted(cdir.glob("*.sigmf-data"))
            if frame_files:
                targets.append((cdir.name, frame_files[0], 0))
    else:
        targets = []
        for cdir in capture_dirs:
            frame_files = sorted(cdir.glob("*.sigmf-data"))
            for idx, ff in enumerate(frame_files[:2]):  # 2 frames per capture = 12 frames
                targets.append((cdir.name, ff, idx))

    for cap_name, data_path, f_idx in targets:
        t0 = time.time()
        print(f"\nProcessing {cap_name} / {data_path.name}...")
        sig = load_signal(str(data_path))
        sig_norm = full_normalize(sig)
        
        folder_name = f"{cap_name}_frame{f_idx:04d}"
        out_dir = out_root / folder_name
        extra = {
            "task": "zenodo_rfi",
            "signal": "RFI",
            "split": "capture",
            "index": f_idx,
            "source": "zenodo_frames",
            "data_path": str(data_path),
            "center_freq": sig.metadata.center_frequency,
        }
        m = pipeline_core(sig, sig_norm, out_dir, data_path.name, extra)
        m["elapsed_sec"] = time.time() - t0
        m["zenodo_meta"] = {
            "capture": cap_name,
            "sample_rate": sig.metadata.sample_rate,
            "center_freq": sig.metadata.center_frequency,
            "num_samples": sig.num_samples,
        }
        all_metrics.append(m)

    write_summaries(out_root, all_metrics, source="zenodo ATA RFI (D:\\dataset\\zenodo_frames, 61.44MHz ci16)")
    return all_metrics


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def write_summaries(out_root: Path, all_metrics: list[dict], source: str):
    # summary.json
    with open(out_root / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    # summary.csv
    csv_path = out_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "file","task","signal","split","index","data_type",
                "num_samples","duration_ms","sample_rate_mhz","center_freq_mhz",
                "rf_freq_min_mhz","rf_freq_max_mhz",
                "top_modulation","mod_confidence",
                "snr_psd_db","snr_m2m4_db","noise_floor_db",
                "bw_3db_khz","bw_frac_khz","occupancy_percent",
                "num_segments","psd_max_db",
                "C42_norm","phase_var","phase_std","amp_var_norm","papr_db",
                "symbol_rate_khz","symbol_conf","freq_offset_khz","freq_conf",
            ],
        )
        w.writeheader()
        for m in all_metrics:
            char = m.get("characterization", {})
            fv = char.get("features") or {}
            sym = char.get("symbol_rate") or {}
            cfo = char.get("freq_offset") or {}
            mod = m.get("modulation_classification") or {}
            rf_range = m.get("psd_stats", {}).get("rf_freq_range_mhz")
            w.writerow({
                "file": m.get("file",""),
                "task": m.get("task",""),
                "signal": m.get("signal",""),
                "split": m.get("split",""),
                "index": m.get("index",""),
                "data_type": m.get("data_type",""),
                "num_samples": m.get("num_samples",""),
                "duration_ms": round(m.get("duration_ms",0),2) if m.get("duration_ms") else "",
                "sample_rate_mhz": round(m.get("sample_rate_hz",0)/1e6,4) if m.get("sample_rate_hz") else "",
                "center_freq_mhz": round(m.get("center_frequency_hz",0)/1e6,2) if m.get("center_frequency_hz") else "",
                "rf_freq_min_mhz": round(rf_range[0], 2) if rf_range else "",
                "rf_freq_max_mhz": round(rf_range[1], 2) if rf_range else "",
                "top_modulation": mod.get("top_candidate", ""),
                "mod_confidence": round(mod.get("confidence", 0), 2) if mod.get("confidence") else "",
                "snr_psd_db": round(m["snr"]["psd_based_db"],2) if m.get("snr") else "",
                "snr_m2m4_db": round(m["snr"]["m2m4_db"],2) if m.get("snr") and m["snr"].get("m2m4_db") is not None else "",
                "noise_floor_db": round(m["noise_floor"]["noise_floor_db"],2) if m.get("noise_floor") else "",
                "bw_3db_khz": round(m["bandwidth"]["bw_3db_hz"]/1e3,2) if m.get("bandwidth") else "",
                "bw_frac_khz": round(m["bandwidth"]["bw_frac_hz"]/1e3,2) if m.get("bandwidth") else "",
                "occupancy_percent": round(m["bandwidth"]["occupancy_percent"],1) if m.get("bandwidth") and m["bandwidth"].get("occupancy_percent") else "",
                "num_segments": m.get("segmentation",{}).get("num_segments",""),
                "psd_max_db": round(m.get("psd_stats",{}).get("max_db",0),2) if m.get("psd_stats") else "",
                "C42_norm": round(fv.get("C42_norm",0),4) if fv and fv.get("C42_norm") is not None else "",
                "phase_var": round(fv.get("phase_variance",0),3) if fv else "",
                "phase_std": round(fv.get("phase_std",0),4) if fv and fv.get("phase_std") is not None else "",
                "amp_var_norm": round(fv.get("amplitude_variance_norm",0),4) if fv else "",
                "papr_db": round(fv.get("papr_db",0),2) if fv else "",
                "symbol_rate_khz": round(sym.get("symbol_rate_hz",0)/1e3,2) if sym and sym.get("symbol_rate_hz") else "",
                "symbol_conf": round(sym.get("confidence",0),2) if sym else "",
                "freq_offset_khz": round(cfo.get("offset_hz",0)/1e3,2) if cfo else "",
                "freq_conf": round(cfo.get("confidence",0),2) if cfo else "",
            })
    # README
    total_mb = sum(f.stat().st_size for f in out_root.rglob("*") if f.is_file())/1024/1024
    readme = f"""# Pipeline Outputs — {source}

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Source: {source}
Files exported: {len(all_metrics)}
Total size: {total_mb:.1f} MB

## Structure
```
{out_root.name}/
  summary.csv/json   <- aggregate metrics
  <file_folder>/
    psd.npz            -> frequencies, psd_db, psd_linear
    spectrogram.npz    -> frequencies, time_bins, spectrogram_db
    iq_sample.npz      -> iq_raw, iq_normalized (4096), sample_rate
    metrics.json       -> noise_floor, snr, bandwidth, segments, psd_stats, characterization
    metadata.json      -> metadata + provenance + sigmf
```
"""
    (out_root / "README.md").write_text(readme)
    print(f"\nDone {out_root.name}: {len(all_metrics)} files -> {out_root} ({total_mb:.1f} MB)")
    print(f"  summary.json/csv ready")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run full pipeline for both datasets into separate output folders")
    parser.add_argument("--data0-out", default=None, help="output for data0 (default: outputs/data0 or outputs/smoke_data0 in smoke mode)")
    parser.add_argument("--zenodo-out", default=None, help="output for zenodo (default: outputs/zenodo or outputs/smoke_zenodo in smoke mode)")
    parser.add_argument("--smoke", action="store_true", help="Run small smoke test with 2 files per dataset into separate smoke folders")
    parser.add_argument("--skip-data0", action="store_true")
    parser.add_argument("--skip-zenodo", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        data0_out = Path(args.data0_out or "outputs/smoke_data0")
        zenodo_out = Path(args.zenodo_out or "outputs/smoke_zenodo")
        print("\n" + "=" * 70)
        print("  SMOKE TEST MODE: 2 files per dataset into separate output folders")
        print(f"  data0 target:  {data0_out}")
        print(f"  zenodo target: {zenodo_out}")
        print("=" * 70)
    else:
        data0_out = Path(args.data0_out or "outputs/data0")
        zenodo_out = Path(args.zenodo_out or "outputs/zenodo")

    if not data0_out.is_absolute():
        data0_out = ROOT / data0_out
    if not zenodo_out.is_absolute():
        zenodo_out = ROOT / zenodo_out

    if not args.skip_data0:
        process_data0(data0_out, smoke=args.smoke)
    else:
        print("Skipping data0")

    if not args.skip_zenodo:
        process_zenodo(zenodo_out, smoke=args.smoke)
    else:
        print("Skipping zenodo")

    print("\nAll pipeline processing completed successfully.")
    print(f"Outputs saved to:\n  - {data0_out}\n  - {zenodo_out}")


if __name__ == "__main__":
    main()
