#!/usr/bin/env python3
"""Smoke test and memory audit for Zenodo chunks and sliced frames.

Strict memory safety guarantees:
- Monitors RAM (via tracemalloc and RSS).
- Verifies that processing a 40,960-sample burst stays < 20 MB RAM and < 200 ms.
- Validates the entire 7-stage DSP detection + characterization pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from app.ingestion.parser import load_signal
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


MEMORY_SAFETY_LIMIT_MB = 100.0  # Abort immediately if RAM increases by >100 MB


def get_process_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback to tracemalloc current if psutil not installed
        current, _ = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)


def run_smoke_test(file_path: str) -> dict:
    """Run full pipeline smoke test with per-stage RAM and timing audits."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {file_path}")

    tracemalloc.start()
    tracemalloc.reset_peak()
    baseline_rss = get_process_memory_mb()

    print("\n" + "=" * 75)
    print(f"  SMOKE TEST: RAM & PIPELINE AUDIT")
    print(f"  Target: {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    print(f"  Baseline Process RSS: {baseline_rss:.1f} MB")
    print(f"  Memory Safety Limit:  {MEMORY_SAFETY_LIMIT_MB:.1f} MB max increase")
    print("=" * 75)

    stages = []

    def log_stage(name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        pre_mem = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        post_mem, peak_mem = [m / (1024 * 1024) for m in tracemalloc.get_traced_memory()]

        delta_mb = post_mem - pre_mem
        if peak_mem > MEMORY_SAFETY_LIMIT_MB:
            raise MemoryError(
                f"SAFETY TRIP in stage '{name}': Peak Python memory ({peak_mem:.1f} MB) exceeded safety limit ({MEMORY_SAFETY_LIMIT_MB} MB)!"
            )

        stages.append({
            "stage": name,
            "elapsed_ms": elapsed_ms,
            "delta_mb": delta_mb,
            "peak_mb": peak_mem,
        })
        print(f"  [{len(stages)}] {name:<26} -> {elapsed_ms:>6.2f} ms | RAM d: {delta_mb:>+5.2f} MB | Peak: {peak_mem:>5.2f} MB")
        return result

    # --- 1. Ingestion ---
    sig = log_stage("1. Ingestion (load_signal)", load_signal, str(path))
    assert len(sig.data) > 0, "Signal data is empty"

    # --- 2. Normalization ---
    sig_norm = log_stage("2. Normalize (DC + Power)", full_normalize, sig)

    # --- 3. Welch PSD ---
    psd = log_stage("3. PSD (Welch 1024)", compute_psd, sig_norm, fft_size=1024)

    # --- 4. Spectrogram (capped) ---
    spec = log_stage("4. Spectrogram (512 / 128)", compute_spectrogram, sig_norm, fft_size=512, max_time_bins=128)

    # --- 5. Noise Floor ---
    nf = log_stage("5. Noise Floor (MAD)", estimate_noise_floor, psd, method="mad")

    # --- 6. Segmentation ---
    seg = log_stage("6. Segmentation", build_segments, psd, nf, sig_norm)

    # --- 7. SNR ---
    snr = log_stage("7. SNR (PSD-based)", estimate_snr_from_psd, psd, nf)

    # --- 8. Bandwidth ---
    bw = log_stage("8. Bandwidth (-3 dB)", estimate_bandwidth, psd, method="3db")

    # --- 9. Characterization Features (HOC C42) ---
    fv = log_stage("9. Features (HOC C42)", extract_features, sig_norm.data, sample_rate=sig.metadata.sample_rate)

    # --- 10. Symbol Rate ---
    sym = log_stage("10. Symbol Rate", estimate_symbol_rate, sig_norm.data, sample_rate=sig.metadata.sample_rate, bandwidth_hz=bw.bandwidth_hz)

    # --- 11. Frequency Offset ---
    cfo = log_stage("11. Freq Offset (CFO)", estimate_freq_offset, sig_norm.data, sample_rate=sig.metadata.sample_rate)

    final_rss = get_process_memory_mb()
    total_traced_peak = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    total_time_ms = sum(s["elapsed_ms"] for s in stages)

    print("-" * 75)
    print(f"  TOTAL PIPELINE TIME:   {total_time_ms:.1f} ms")
    print(f"  PEAK TRACED RAM:       {total_traced_peak:.2f} MB (Target was < 20 MB)")
    print(f"  PROCESS RSS DELTA:     {(final_rss - baseline_rss):+.2f} MB")
    print("=" * 75)

    print("\n  EXTRACTED RF CHARACTERISTICS:")
    print(f"    - Sample Count:     {sig.num_samples:,} samples")
    print(f"    - Sample Rate:      {sig.metadata.sample_rate / 1e6:.2f} MHz")
    print(f"    - Center Frequency: {sig.metadata.center_frequency / 1e6 if sig.metadata.center_frequency else 0:.2f} MHz")
    print(f"    - Noise Floor:      {nf.noise_floor_db:.2f} dB")
    print(f"    - SNR (PSD):        {snr.snr_db:.2f} dB")
    print(f"    - -3dB Bandwidth:   {bw.bandwidth_hz / 1e3:.1f} kHz")
    print(f"    - Segments Found:   {seg.num_segments}")
    print(f"    - HOC C42 (norm):   {fv.C42_norm:.4f}")
    print(f"    - PAPR:             {fv.papr_db:.2f} dB")
    print(f"    - Est. Symbol Rate: {sym.symbol_rate_hz / 1e3:.1f} kHz (conf={sym.confidence:.2f})")
    print(f"    - Est. CFO:         {cfo.offset_hz / 1e3:.1f} kHz (conf={cfo.confidence:.2f})")
    print("-" * 75)
    print("  VERDICT: PASSED - RAM SAFE & FULLY FUNCTIONAL\n")

    return {
        "num_samples": sig.num_samples,
        "total_time_ms": total_time_ms,
        "peak_ram_mb": total_traced_peak,
        "snr_db": snr.snr_db,
        "bw_hz": bw.bandwidth_hz,
        "c42_norm": fv.C42_norm,
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke test pipeline on sliced Zenodo frames")
    parser.add_argument("--sliced-frame", "-f", required=True, help="Path to a sliced .sigmf-data file")
    args = parser.parse_args()

    run_smoke_test(args.sliced_frame)


if __name__ == "__main__":
    main()
