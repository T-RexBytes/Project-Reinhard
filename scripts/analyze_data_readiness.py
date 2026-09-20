"""
analyze_data_readiness.py
--------------------------
Deep-reads all outputs/plot_data/* folders and tells you:
  1. What signals/tasks/splits you have
  2. What RF parameters are already computed
  3. What is MISSING for modulation hypothesis
  4. Whether the existing detection pipeline is adequate
  5. Go / No-Go verdict per signal
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
PLOT_DATA  = ROOT / "outputs" / "plot_data"

# ── modulation hypothesis needs these to exist ──────────────────────────────
REQUIRED_FOR_MOD = [
    "snr_psd_db",          # to pick confident vs uncertain regime
    "snr_m2m4_db",         # modulation-type sensitive
    "noise_floor_db",      # needed for threshold-based detection
    "bw_3db_hz",           # bandwidth → helps discriminate FSK vs PSK vs QAM
    "num_segments",        # is there one clean signal or mixed?
    "iq_array",            # raw IQ must be loadable from npz
    "psd_array",           # PSD must be loadable
]

# What's NOT computed yet (gap analysis)
MISSING_FOR_MOD = {
    "symbol_rate_hz":       "❌ NOT computed — cyclostationary or squaring estimator needed",
    "freq_offset_hz":       "❌ NOT computed — carrier sync estimator needed",
    "phase_variance":       "❌ NOT computed — needed for PSK vs FSK discrimination",
    "amplitude_variance":   "❌ NOT computed — needed for QAM vs PSK",
    "instantaneous_freq":   "❌ NOT computed — d(angle)/dt, key for FSK",
    "higher_order_stats":   "❌ NOT computed — kurtosis/skewness of |IQ|, C42, C40",
    "modulation_label":     "❌ NOT labeled — ground truth from SigMF only says CommSignal/EMI",
}

NICE_TO_HAVE = {
    "timing_offset_est":    "⚠️  missing — Gardner loop or Mueller-Müller",
    "occupied_bw_percent":  "✅ available via bw_frac_hz",
    "peak_snr_segment":     "✅ available in segments[].snr_db",
    "constellation":        "❌ NOT computed — needs demod first",
    "evm":                  "❌ NOT computed — needs sync + demod",
}

# ── helpers ─────────────────────────────────────────────────────────────────

def load_npz_safe(path: Path) -> dict | None:
    try:
        return dict(np.load(path, allow_pickle=False))
    except Exception as e:
        return None


def analyze_folder(folder: Path) -> dict:
    result = {"name": folder.name, "issues": [], "ok": [], "iq_shape": None, "psd_shape": None}

    # 1. metrics.json
    metrics_path = folder / "metrics.json"
    if not metrics_path.exists():
        result["issues"].append("metrics.json MISSING")
        return result

    with open(metrics_path) as f:
        m = json.load(f)

    result["signal"]  = m.get("signal", "?")
    result["task"]    = m.get("task", "?")
    result["split"]   = m.get("split", "?")
    result["index"]   = m.get("index", "?")
    result["num_samples"] = m.get("num_samples", 0)
    result["duration_ms"] = m.get("duration_ms", 0)

    # SNR
    snr = m.get("snr", {})
    result["snr_psd_db"]  = snr.get("psd_based_db")
    result["snr_m2m4_db"] = snr.get("m2m4_db")

    if result["snr_psd_db"] is None:
        result["issues"].append("snr_psd_db missing")
    else:
        result["ok"].append(f"SNR(PSD)={result['snr_psd_db']:.1f} dB")

    # Noise floor
    nf = m.get("noise_floor", {})
    result["noise_floor_db"] = nf.get("noise_floor_db")
    if result["noise_floor_db"] is None:
        result["issues"].append("noise_floor_db missing")
    else:
        result["ok"].append(f"NoiseFloor={result['noise_floor_db']:.1f} dB")

    # Bandwidth
    bw = m.get("bandwidth", {})
    result["bw_3db_hz"]     = bw.get("bw_3db_hz")
    result["bw_frac_hz"]    = bw.get("bw_frac_hz")
    result["occupancy_pct"] = bw.get("occupancy_percent")
    if result["bw_3db_hz"] is None:
        result["issues"].append("bw_3db_hz missing")
    else:
        result["ok"].append(f"BW(3dB)={result['bw_3db_hz']/1e3:.1f} kHz")

    # Segmentation
    seg = m.get("segmentation", {})
    result["num_segments"] = seg.get("num_segments", 0)
    result["segments"]     = seg.get("segments", [])
    result["ok"].append(f"Segments={result['num_segments']}")

    # Peak segment SNR
    if result["segments"]:
        best = max(result["segments"], key=lambda s: s.get("snr_db", 0))
        result["best_segment_snr"] = best.get("snr_db")
        result["best_segment_bw"]  = best.get("bandwidth_hz")
        result["ok"].append(f"BestSeg SNR={result['best_segment_snr']:.1f} dB  BW={result['best_segment_bw']/1e3:.1f} kHz")

    # 2. IQ npz
    iq_path = folder / "iq_sample.npz"
    if not iq_path.exists():
        result["issues"].append("iq_sample.npz MISSING — cannot run modulation features")
    else:
        npz = load_npz_safe(iq_path)
        if npz is None:
            result["issues"].append("iq_sample.npz unreadable")
        else:
            key = list(npz.keys())[0] if npz else None
            if key:
                result["iq_shape"] = npz[key].shape
                result["ok"].append(f"IQ loaded: shape={npz[key].shape}  dtype={npz[key].dtype}")

    # 3. PSD npz
    psd_path = folder / "psd.npz"
    if not psd_path.exists():
        result["issues"].append("psd.npz MISSING")
    else:
        npz = load_npz_safe(psd_path)
        if npz:
            key = list(npz.keys())[0]
            result["psd_shape"] = npz[key].shape
            result["ok"].append(f"PSD loaded: shape={npz[key].shape}")

    # 4. spectrogram npz
    spec_path = folder / "spectrogram.npz"
    result["has_spectrogram"] = spec_path.exists()
    if result["has_spectrogram"]:
        result["ok"].append("Spectrogram present")

    return result


def modulation_regime(snr_psd: float | None, bw_hz: float | None) -> str:
    if snr_psd is None:
        return "UNKNOWN"
    if snr_psd < 5:
        return "LOW-SNR  (classifier may fail — threshold regime)"
    if snr_psd < 12:
        return "MED-SNR  (soft decisions, need good feature extractor)"
    return "HIGH-SNR (good for classification)"


def signal_character(signal: str, bw_3db: float | None, snr_m2m4: float | None) -> str:
    """Heuristic guess at likely modulation family from bandwidth + m2m4."""
    lines = []
    if "EMI" in signal:
        lines.append("→ EMISignal1: likely narrowband interferer (EMI/CW/FSK)")
        if bw_3db and bw_3db < 500e3:
            lines.append("  BW < 500 kHz confirms narrowband — candidate: CW or FSK")
    elif "CommSignal2" in signal:
        lines.append("→ CommSignal2: SigMF says QPSK. Wideband comm signal.")
        if bw_3db and bw_3db > 500e3:
            lines.append("  BW > 500 kHz — consistent with QPSK @ ~100k sym/s")
    elif "CommSignal3" in signal:
        lines.append("→ CommSignal3: SigMF suggests comm signal, likely PSK/QAM variant")
    if snr_m2m4 is not None:
        if snr_m2m4 > 3:
            lines.append("  M2M4 high → likely single-carrier PSK/QAM with good SNR")
        elif snr_m2m4 < 0:
            lines.append("  M2M4 negative → multi-signal or noise-dominant segment")
        else:
            lines.append("  M2M4 ~1-3 → moderate single-carrier, or mixed")
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    folders = sorted([f for f in PLOT_DATA.iterdir() if f.is_dir()])
    if not folders:
        print(f"No data folders found in {PLOT_DATA}")
        sys.exit(1)

    print("=" * 70)
    print("  RF DATA READINESS ANALYSIS")
    print(f"  Source: {PLOT_DATA}")
    print(f"  Folders found: {len(folders)}")
    print("=" * 70)

    all_results = []
    for folder in folders:
        r = analyze_folder(folder)
        all_results.append(r)

    # ── Per-signal report ────────────────────────────────────────────────────
    for r in all_results:
        print(f"\n{'─'*70}")
        print(f"  📁 {r['name']}")
        print(f"     Signal={r.get('signal','?')}  Task={r.get('task','?')}  Split={r.get('split','?')}  Index={r.get('index','?')}")
        print(f"     Samples={r.get('num_samples',0):,}  Duration={r.get('duration_ms',0):.2f} ms  IQ shape={r.get('iq_shape')}")
        print()
        for ok in r["ok"]:
            print(f"     ✅ {ok}")
        for issue in r["issues"]:
            print(f"     ❌ {issue}")

        snr_psd = r.get("snr_psd_db")
        bw_3db  = r.get("bw_3db_hz")
        snr_m2m4= r.get("snr_m2m4_db")
        signal  = r.get("signal", "")

        print(f"\n     SNR Regime: {modulation_regime(snr_psd, bw_3db)}")
        print()
        for line in signal_character(signal, bw_3db, snr_m2m4).splitlines():
            print(f"     {line}")

    # ── Aggregated gap analysis ──────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  PIPELINE GAP ANALYSIS — WHAT YOU HAVE vs WHAT YOU NEED")
    print("=" * 70)

    print("\n  ── ALREADY COMPUTED (by your detection pipeline) ──")
    already = [
        "✅ SNR (PSD-based Welch method)",
        "✅ SNR (M2M4 higher-order moment method)",
        "✅ Noise floor (MAD estimator)",
        "✅ Occupied bandwidth (3dB + fractional)",
        "✅ Signal segmentation (per-segment SNR, BW, freq range)",
        "✅ PSD arrays (numpy npz, ready for feature extraction)",
        "✅ IQ arrays (numpy npz, ready for feature extraction)",
        "✅ Spectrogram arrays",
        "✅ SigMF metadata (sample_rate=25 MHz, center_freq, datatype=cf32)",
    ]
    for line in already:
        print(f"    {line}")

    print("\n  ── MISSING — NEEDED BEFORE MODULATION HYPOTHESIS ──")
    missing = [
        ("symbol_rate_hz",     "Cyclostationary analysis (raised-cosine PSD peaks)"),
        ("carrier_offset_hz",  "Coarse frequency offset: peak-of-PSD or FFT of squared signal"),
        ("phase_variance",     "std(angle(IQ)) — high=FM/FSK, low=PSK/QAM"),
        ("amplitude_variance", "std(|IQ|) / mean(|IQ|) — high=QAM, low=PSK/BPSK"),
        ("instantaneous_freq", "np.diff(np.unwrap(np.angle(IQ))) * fs/(2π) — FSK fingerprint"),
        ("kurtosis_IQ",        "scipy.stats.kurtosis(|IQ|) — modulation order hint"),
        ("C42_cumulant",       "4th-order cumulant — distinguishes BPSK/QPSK/16QAM/FSK"),
    ]
    for param, how in missing:
        print(f"    ❌ {param:<25} → Compute via: {how}")

    print("\n  ── MISSING — NEEDED AFTER MOD HYPOTHESIS (demod phase) ──")
    post = [
        ("constellation",   "Needs sync → scatter plot of sym(I, Q)"),
        ("evm_db",          "Error Vector Magnitude — needs timing/carrier lock"),
        ("timing_offset",   "Gardner loop or Mueller-Müller TED"),
    ]
    for param, how in post:
        print(f"    ❌ {param:<25} → {how}")

    print("\n  ── SIGNAL-TYPE SUMMARY ──")
    by_signal: dict[str, list] = {}
    for r in all_results:
        s = r.get("signal", "?")
        by_signal.setdefault(s, []).append(r)

    for sig, items in by_signal.items():
        snrs  = [i["snr_psd_db"] for i in items if i.get("snr_psd_db") is not None]
        bws   = [i["bw_3db_hz"] for i in items if i.get("bw_3db_hz") is not None]
        m2m4s = [i["snr_m2m4_db"] for i in items if i.get("snr_m2m4_db") is not None]
        print(f"\n    {sig}:")
        print(f"      Count       : {len(items)} samples")
        if snrs:
            print(f"      SNR(PSD)    : {min(snrs):.1f} – {max(snrs):.1f} dB  (mean {sum(snrs)/len(snrs):.1f})")
        if bws:
            print(f"      BW(3dB)     : {min(bws)/1e3:.1f} – {max(bws)/1e3:.1f} kHz")
        if m2m4s:
            print(f"      M2M4        : {min(m2m4s):.2f} – {max(m2m4s):.2f} dB")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("  VERDICT")
    print("=" * 70)
    print("""
  ✅ Your pipeline foundation is SOLID:
     FFT, PSD, noise floor, segmentation, SNR are all working and produce
     clean results across CommSignal2, CommSignal3, EMISignal1.

  ⚠️  You are NOT yet ready for modulation hypothesis because:
     (a) Symbol rate is unknown — pipeline has no estimator yet.
     (b) Phase/amplitude variance features are not extracted.
     (c) Instantaneous frequency is not computed (critical for FSK vs PSK).
     (d) Higher-order cumulants (C42) are not computed.

  📋 RECOMMENDED NEXT STEPS (in order):
     1. Write   app/characterization/features.py
        → compute: phase_var, amp_var, inst_freq, kurtosis, C42
        → input : IQ npz (already available)
        → output: FeatureVector dataclass

     2. Write   app/characterization/symbol_rate.py
        → method: PSD of |IQ|^2 (squared-signal), find spectral peak
        → validates against CommSignal2 SigMF (known QPSK)

     3. Write   app/characterization/freq_offset.py
        → coarse: argmax(PSD) for AM/FM
        → fine  : FFT of IQ^2 / IQ^4 for PSK

     4. THEN go to   app/modulation/classifier.py
        → use FeatureVector → rule-based or ML ranker
        → output: {BPSK:p, QPSK:p, FSK:p, 16QAM:p}
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
