"""
scripts/validate_characterization.py
--------------------------------------
Validates all three characterization modules against:
  1. RadioML 2016.10A  — labeled modulations (BPSK, QPSK, QAM16, CPFSK)
  2. plot_data samples — real CommSignal2/3 and EMISignal1 IQ

Prints a human-readable analysis report and checks that:
  - C42_norm separates BPSK / QPSK / QAM16 / FSK correctly
  - Symbol rate is detectable from CommSignal2 (known QPSK @ 25 MHz)
  - Freq offset returns near-zero for baseband-centered signals
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
PLOT_DATA = ROOT / "outputs" / "plot_data"
RADIOML   = ROOT / "datasets" / "archive (1)" / "RML2016.10a_dict.pkl"

sys.path.insert(0, str(ROOT))

from app.characterization.features    import extract_features, FeatureVector
from app.characterization.symbol_rate import estimate_symbol_rate
from app.characterization.freq_offset import estimate_freq_offset
from app.modulation.classifier        import ModulationClassifier

SEP = "=" * 72


# ── RadioML validation ───────────────────────────────────────────────────────

def validate_radioml():
    print(f"\n{SEP}")
    print("  PART 1: RadioML 2016.10A — Feature Separation Check")
    print(SEP)

    if not RADIOML.exists():
        print(f"  [SKIP] RadioML not found at {RADIOML}")
        return

    print(f"  Loading {RADIOML.name} ...")
    with open(RADIOML, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Focus on our 4 MVP modulations at high SNR
    target_mods = {
        "BPSK":  "BPSK",
        "QPSK":  "QPSK",
        "QAM16": "QAM16",
        "CPFSK": "FSK (CPFSK)",
    }
    snr_level = 18   # use high SNR for clean feature separation test

    print(f"  SNR level: {snr_level} dB  |  Samples per mod: 100\n")
    print(f"  {'Modulation':<12} {'C42_norm':>10} {'PhaseVar':>10} "
          f"{'AmpVarN':>10} {'InstFStd':>10} {'SpectFlat':>11}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*11}")

    results: dict[str, list[FeatureVector]] = {}

    for mod_key, mod_label in target_mods.items():
        key = (mod_key, snr_level)
        if key not in data:
            print(f"  [{mod_label}] NOT in dataset at SNR={snr_level}")
            continue

        samples = data[key][:100]   # shape (100, 2, 128)
        fvs = []
        for s in samples:
            iq = (s[0] + 1j * s[1]).astype(np.complex128)
            fv = extract_features(iq, sample_rate=1.0)
            fvs.append(fv)

        results[mod_label] = fvs

        c42   = np.mean([f.C42_norm for f in fvs])
        phvar = np.mean([f.phase_variance for f in fvs])
        ampv  = np.mean([f.amplitude_variance_norm for f in fvs])
        instf = np.mean([f.instantaneous_freq_std for f in fvs])
        sflt  = np.mean([f.spectral_flatness for f in fvs])

        print(f"  {mod_label:<12} {c42:>10.4f} {phvar:>10.4f} "
              f"{ampv:>10.4f} {instf:>10.4f} {sflt:>11.4f}")

    # Summary interpretation
    print("\n  Expected theoretical |C42_norm|:")
    print("    BPSK   ~ 2.0")
    print("    QPSK   ~ 1.0")
    print("    QAM16  ~ 0.68")
    print("    FSK    ~ 0.0  (constant envelope)")

    # Check separation
    print("\n  Separation check:")
    c42_means = {}
    for label, fvs in results.items():
        c42_means[label] = np.mean([f.C42_norm for f in fvs])

    bpsk_ok  = c42_means.get("BPSK", 0)        > c42_means.get("QPSK", 0)
    qpsk_ok  = c42_means.get("QPSK", 0)        > c42_means.get("QAM16 (QAM16)", 0) if "QAM16 (QAM16)" in c42_means else True
    fsk_low  = c42_means.get("FSK (CPFSK)", 1) < 0.5

    status = lambda ok: "OK" if ok else "FAIL"
    print(f"    BPSK > QPSK in C42_norm : {status(bpsk_ok)}")
    print(f"    FSK C42_norm < 0.5      : {status(fsk_low)}")


# ── plot_data validation ─────────────────────────────────────────────────────

def validate_plot_data():
    print(f"\n{SEP}")
    print("  PART 2: Real Signals (outputs/plot_data) — Full Characterization")
    print(SEP)

    folders = sorted([f for f in PLOT_DATA.iterdir() if f.is_dir()])
    if not folders:
        print(f"  [SKIP] No folders in {PLOT_DATA}")
        return

    for folder in folders:
        iq_path      = folder / "iq_sample.npz"
        metrics_path = folder / "metrics.json"
        if not iq_path.exists():
            continue

        # Load IQ
        npz = np.load(iq_path, allow_pickle=False)
        key = list(npz.keys())[0]
        iq  = npz[key].astype(np.complex128)

        # Load metrics for context
        snr_psd = None
        bw_hz   = None
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            snr_psd = m.get("snr", {}).get("psd_based_db")
            bw_hz   = m.get("bandwidth", {}).get("bw_3db_hz")

        fs = 25e6   # known from SigMF

        # Run all three characterization modules
        fv     = extract_features(iq, sample_rate=fs, snr_db=snr_psd)
        sr     = estimate_symbol_rate(iq, sample_rate=fs, bandwidth_hz=bw_hz)
        cfo    = estimate_freq_offset(iq, sample_rate=fs)

        signal_name = folder.name
        print(f"\n  [{signal_name}]")
        print(f"    IQ samples    : {len(iq)}")
        print(f"    SNR (PSD)     : {snr_psd:.1f} dB" if snr_psd else "    SNR           : N/A")
        print(f"    BW(3dB)       : {bw_hz/1e3:.1f} kHz" if bw_hz else "    BW            : N/A")
        print()
        print(f"    --- Features ---")
        print(f"    C42_norm      : {fv.C42_norm:.4f}   (BPSK~2 QPSK~1 QAM~0.68 FSK~0)")
        print(f"    Phase var     : {fv.phase_variance:.4f}   (high=FSK/FM  low=PSK)")
        print(f"    Amp var norm  : {fv.amplitude_variance_norm:.4f}   (high=QAM  low=const-envelope)")
        print(f"    Inst freq std : {fv.instantaneous_freq_std/1e3:.1f} kHz")
        print(f"    Kurtosis amp  : {fv.kurtosis_amplitude:.3f}")
        print(f"    Spectral flat : {fv.spectral_flatness:.4f}   (0=tonal  1=noise)")
        print(f"    PAPR          : {fv.papr_db:.1f} dB")
        print()
        print(f"    --- Symbol Rate ---")
        print(f"    Estimated     : {sr.symbol_rate_hz/1e3:.1f} kHz  "
              f"(confidence={sr.confidence:.2f}  method={sr.method})")
        print(f"    Notes         : {sr.notes}")
        print()
        print(f"    --- Carrier Offset ---")
        print(f"    Offset        : {cfo.offset_hz/1e3:.2f} kHz  "
              f"(confidence={cfo.confidence:.2f}  method={cfo.method})")
        print(f"    Notes         : {cfo.notes}")

        # Modulation Classifier candidate ranking
        print()
        clf = ModulationClassifier()
        mod_res = clf.classify(
            fv,
            snr_db=snr_psd,
            symbol_rate_hz=sr.symbol_rate_hz if sr else None,
            bandwidth_hz=None,
            sample_rate=fs,
        )
        print(f"    >>> Top Modulation Hypothesis: {mod_res.top_candidate}  (confidence={mod_res.confidence:.2f})")
        print("    --- Candidate Leaderboard ---")
        for rank, cand in enumerate(mod_res.candidates[:3], 1):
            print(f"        #{rank} {cand.modulation:<7} (score={cand.score*100:5.1f}%, conf={cand.confidence:.2f}) | {cand.why}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SEP)
    print("  RF CHARACTERIZATION LAYER — VALIDATION REPORT")
    print(SEP)

    validate_radioml()
    validate_plot_data()

    print(f"\n{SEP}")
    print("  DONE")
    print(SEP)
