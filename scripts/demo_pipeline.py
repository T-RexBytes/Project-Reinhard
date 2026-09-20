#!/usr/bin/env python3
"""Full pipeline demo: input data -> pipeline output."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pickle
import numpy as np
from app.dataset.loader import DatasetScanner, TaskType, SignalType, SplitType
from app.preprocessing.normalize import full_normalize
from app.detection.fft import compute_psd
from app.detection.noise_floor import estimate_noise_floor
from app.detection.segmentation import build_segments
from app.detection.snr import estimate_snr_from_psd, estimate_snr_m2m4
from app.detection.bandwidth import estimate_bandwidth

scanner = DatasetScanner(r"D:\dataset")
scanner.scan()

entries = scanner.filter(task=TaskType.DEMOD, signal=SignalType.COMMSIGNAL2, split=SplitType.TRAIN, max_entries=1)
entry = entries[0]
sig = scanner.load_entry(entry)

print("=" * 70)
print("  INPUT DATA")
print("=" * 70)
print(f"File: {entry.data_path.name}")
print(f"Task: {entry.task.value} | Signal: {entry.signal.value} | Split: {entry.split.value}")
print()
print("--- SigMF Metadata ---")
with open(entry.meta_path) as f:
    print(json.dumps(json.load(f), indent=2))
print()
print("--- Parsed Signal ---")
m = sig.metadata
print(f"Data type:    {m.data_type.value}")
print(f"Sample rate:  {m.sample_rate/1e6:.1f} MHz")
print(f"Samples:      {sig.num_samples:,}  |  Duration: {sig.duration_seconds*1e3:.4f} ms")
print()
print("--- Raw IQ (first 5 samples) ---")
for i in range(5):
    print(f"  [{i}] I={sig.data[i].real:+.6f}  Q={sig.data[i].imag:+.6f}  |A|={abs(sig.data[i]):.6f}")
print(f"  Mean power: {np.mean(np.abs(sig.data)**2):.4f}  |  Peak: {np.max(np.abs(sig.data)):.4f}")

print()
print("=" * 70)
print("  PIPELINE OUTPUT")
print("=" * 70)

sig_norm = full_normalize(sig)
psd = compute_psd(sig_norm, fft_size=1024)
nf = estimate_noise_floor(psd, method="mad")
seg = build_segments(psd, nf, sig_norm)
snr_psd = estimate_snr_from_psd(psd, nf)
snr_m2m4 = estimate_snr_m2m4(sig_norm.data)
bw_3db = estimate_bandwidth(psd, method="3db")
bw_frac = estimate_bandwidth(psd, method="fractional")

print()
print("  [Stage 1] Normalization")
print(f"    DC offset removed -> mean I={sig_norm.data.real.mean():.2e}, Q={sig_norm.data.imag.mean():.2e}")
print(f"    Power normalized -> mean power={np.mean(np.abs(sig_norm.data)**2):.4f}")
print()
print("  [Stage 2] PSD (Welch, 1024-pt FFT)")
print(f"    {len(psd.frequencies)} freq bins  |  range: {psd.frequencies[0]/1e3:.0f} to {psd.frequencies[-1]/1e3:.0f} kHz")
print(f"    Max PSD: {psd.psd_db.max():.1f} dB  |  Min PSD: {psd.psd_db.min():.1f} dB")
print()
print("  [Stage 3] Noise Floor (MAD)")
print(f"    Noise floor: {nf.noise_floor_db:.2f} dB")
print()
print("  [Stage 4] Segmentation")
print(f"    {seg.num_segments} segments detected:")
for i, s in enumerate(seg.segments):
    print(f"      #{i}: center={s.center_freq_hz/1e3:+.0f} kHz  bw={s.bandwidth_hz/1e3:.1f} kHz  snr={s.snr_db:.1f} dB")
print()
print("  [Stage 5] SNR")
print(f"    PSD-based:  {snr_psd.snr_db:.2f} dB  (signal={snr_psd.signal_power_db:.1f} dB, noise={snr_psd.noise_power_db:.1f} dB)")
print(f"    M2M4:       {snr_m2m4.snr_db:.2f} dB")
print()
print("  [Stage 6] Bandwidth")
print(f"    -3dB:       {bw_3db.bandwidth_hz/1e3:.2f} kHz")
print(f"    Fractional: {bw_frac.bandwidth_hz/1e3:.2f} kHz  ({bw_frac.occupancy_percent:.1f}% occupancy)")

print()
print("=" * 70)
print("  GROUND TRUTH")
print("=" * 70)
gt_entries = scanner.filter(task=TaskType.DEMOD, signal=SignalType.COMMSIGNAL2, split=SplitType.TRAIN, component_type="QPSK", max_entries=1)
if gt_entries and gt_entries[0].ground_truth_bits_path:
    with open(gt_entries[0].ground_truth_bits_path, "rb") as f:
        bits, info = pickle.load(f)
    print(f"  QPSK bits:     {len(bits)} bits  (first 32: {bits[:32].tolist()})")
    print(f"  Bit 1 ratio:   {bits.mean():.2%}")
    print(f"  Symbols:       {len(info['symbols'])} QPSK symbols")
    zero_padded = info["symbols (zero-padded)"]
    print(f"  Zero-padded:   {len(zero_padded)} samples")

print()
print("=" * 70)
print("  PROVENANCE")
print("=" * 70)
for p in sig_norm.provenance:
    print(f"  [{p.stage}] {p.operation}")
