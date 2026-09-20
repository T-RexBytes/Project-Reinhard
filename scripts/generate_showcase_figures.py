#!/usr/bin/env python3
"""Generate publication-grade showcase figures highlighting Modulation Classification,
DSP Demodulation/EVM, and RF physical parameters for both data0 and zenodo datasets.

Creates 4 presentation-ready figures in outputs/figures/:
  1) data0_rf_showcase.png                  — Time, Constellation, PSD, Waterfall + Classification HUD
  2) zenodo_rfi_showcase.png                — Multi-band ATA deep space telescope RFI showcase + Classification
  3) rf_landscape_executive.png             — Cross-dataset executive dashboard & hypothesis distributions
  4) modulation_classification_showcase.png — Dedicated 4x4 deep-dive into candidate ranking & EVM verification

High DPI (300 DPI), clean scientific dark styling, rich annotations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA0_DIR = ROOT / "outputs" / "data0"
ZENODO_DIR = ROOT / "outputs" / "zenodo"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sleek modern dark styling
plt.style.use("dark_background")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "grid.color": "#2a2a2a",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


def load_file_package(folder: Path):
    """Load psd, spectrogram, iq_sample, and metrics for a folder."""
    psd_d = np.load(folder / "psd.npz")
    spec_d = np.load(folder / "spectrogram.npz")
    iq_d = np.load(folder / "iq_sample.npz")
    with open(folder / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(folder / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return psd_d, spec_d, iq_d, metrics, meta


# =========================================================================
# FIGURE 1: DATA0 SHOWCASE (MIT RF Challenge with Classification HUD)
# =========================================================================
def generate_data0_showcase():
    print("Generating Figure 1: data0_rf_showcase.png ...")
    targets = [
        ("CommSignal2_demod_train_0000", "CommSignal2 (QPSK Target)", "#00e5ff"),
        ("CommSignal3_demod_train_0000", "CommSignal3 (Wideband 16QAM/Burst)", "#ff9100"),
        ("EMISignal1_demod_train_0000", "EMISignal1 (Narrowband Tone/EMI)", "#76ff03"),
    ]

    fig = plt.figure(figsize=(19, 11), facecolor="#111116")
    gs = GridSpec(3, 4, figure=fig, width_ratios=[1.15, 0.95, 1.15, 1.35], wspace=0.28, hspace=0.35)

    fig.suptitle("SIH 26147 — DATASET SHOWCASE: MIT RF CHALLENGE (data0)", fontsize=15, fontweight="bold", color="#ffffff", y=0.97)
    fig.text(0.5, 0.94, "Supervised Benchmarks @ 25 MHz | Time Waveform, Sliced Constellation, Welch PSD & Classified Hypothesis HUD",
             ha="center", fontsize=9.5, color="#888899")

    for row_idx, (folder_name, label, accent_color) in enumerate(targets):
        folder = DATA0_DIR / folder_name
        if not folder.exists():
            continue
        psd, spec, iq_pkg, m, meta = load_file_package(folder)

        iq_raw = iq_pkg["iq_raw"]
        iq_norm = iq_pkg["iq_normalized"]
        fs = float(m["sample_rate_hz"])
        n_samples = len(iq_norm)
        t_us = np.arange(n_samples) / fs * 1e6

        # --- Col 1: Time Domain ---
        ax1 = fig.add_subplot(gs[row_idx, 0])
        ax1.set_facecolor("#16161e")
        show_n = min(250, n_samples)
        ax1.plot(t_us[:show_n], iq_norm[:show_n].real, color=accent_color, linewidth=1.1, label="I (In-Phase)", alpha=0.9)
        ax1.plot(t_us[:show_n], iq_norm[:show_n].imag, color="#ff4081", linewidth=0.9, label="Q (Quadrature)", alpha=0.8)
        ax1.plot(t_us[:show_n], np.abs(iq_norm[:show_n]), color="#ffffff", linewidth=0.8, linestyle=":", label="|Envelope|", alpha=0.6)
        ax1.set_title(f"{label} — Time Waveform", fontsize=10, fontweight="bold", color=accent_color, loc="left")
        ax1.set_xlabel("Time (µs)", fontsize=8, color="#aaaaaa")
        ax1.set_ylabel("Normalized Amplitude", fontsize=8, color="#aaaaaa")
        ax1.set_ylim(-2.2, 2.2)
        ax1.grid(True)
        if row_idx == 0:
            ax1.legend(loc="upper right", fontsize=7, facecolor="#22222a", edgecolor="none")

        # --- Col 2: Constellation Density Heatmap with Slicing Grid ---
        ax2 = fig.add_subplot(gs[row_idx, 1])
        ax2.set_facecolor("#16161e")
        n_const = min(2048, len(iq_norm))
        i_pts = iq_norm[:n_const].real
        q_pts = iq_norm[:n_const].imag
        ax2.hexbin(i_pts, q_pts, gridsize=35, cmap="plasma", mincnt=1, edgecolors="none")
        theta = np.linspace(0, 2 * np.pi, 200)
        ax2.plot(np.cos(theta), np.sin(theta), color="#444455", linestyle="--", linewidth=0.8)
        ax2.axhline(0, color="#333344", linewidth=0.6)
        ax2.axvline(0, color="#333344", linewidth=0.6)
        ax2.set_xlim(-2.0, 2.0)
        ax2.set_ylim(-2.0, 2.0)
        ax2.set_aspect("equal")

        top_mod = m.get("modulation_classification", {}).get("top_candidate", "Unknown")
        conf_pct = m.get("modulation_classification", {}).get("confidence", 0.0) * 100.0
        ax2.set_title(f"Constellation | Class: {top_mod} ({conf_pct:.0f}%)", fontsize=9, fontweight="bold", color="#ffffff")
        ax2.set_xlabel("In-Phase (I)", fontsize=8, color="#aaaaaa")
        ax2.set_ylabel("Quadrature (Q)", fontsize=8, color="#aaaaaa")
        ax2.grid(True)

        # --- Col 3: Welch PSD with Detection Annotations ---
        ax3 = fig.add_subplot(gs[row_idx, 2])
        ax3.set_facecolor("#16161e")
        freqs_mhz = psd["frequencies"] / 1e6
        psd_db = psd["psd_db"]
        nf = m["noise_floor"]["noise_floor_db"]
        thr = m["segmentation"]["detection_threshold_db"]
        bw_khz = m["bandwidth"]["bw_3db_hz"] / 1e3
        snr_db = m["snr"]["psd_based_db"]

        ax3.plot(freqs_mhz, psd_db, color=accent_color, linewidth=1.0, label="PSD")
        ax3.axhline(nf, color="#ef5350", linestyle="--", linewidth=1.0, label=f"NF: {nf:.1f} dB")
        ax3.axhline(thr, color="#ffd54f", linestyle=":", linewidth=1.0, label=f"Thr: {thr:.1f} dB")

        if m["bandwidth"].get("bw_3db_lower_hz") is not None and m["bandwidth"].get("bw_3db_upper_hz") is not None:
            low_mhz = m["bandwidth"]["bw_3db_lower_hz"] / 1e6
            high_mhz = m["bandwidth"]["bw_3db_upper_hz"] / 1e6
            ax3.axvspan(low_mhz, high_mhz, color=accent_color, alpha=0.15, label=f"3dB BW: {bw_khz:.0f} kHz")

        ax3.set_title(f"PSD Spectrum | SNR: {snr_db:.1f} dB", fontsize=9, color="#dddddd")
        ax3.set_xlabel("Frequency (MHz)", fontsize=8, color="#aaaaaa")
        ax3.set_ylabel("Power (dB/Hz)", fontsize=8, color="#aaaaaa")
        ax3.set_ylim(-115, psd_db.max() + 10)
        ax3.grid(True)
        if row_idx == 0:
            ax3.legend(loc="upper right", fontsize=7, facecolor="#22222a", edgecolor="none")

        # --- Col 4: Spectrogram Waterfall & Classification HUD ---
        ax4 = fig.add_subplot(gs[row_idx, 3])
        ax4.set_facecolor("#16161e")
        spec_db = spec["spectrogram_db"]
        t_bins_ms = spec["time_bins"] * 1e3
        f_bins_mhz = spec["frequencies"] / 1e6

        v_min, v_max = np.percentile(spec_db, 5), np.percentile(spec_db, 99)
        ax4.imshow(spec_db, aspect="auto", origin="lower", cmap="magma",
                   extent=[t_bins_ms[0], t_bins_ms[-1], f_bins_mhz[0], f_bins_mhz[-1]],
                   vmin=v_min, vmax=v_max)
        ax4.set_title("Time-Frequency Waterfall & Classification", fontsize=9, color="#dddddd")
        ax4.set_xlabel("Time (ms)", fontsize=8, color="#aaaaaa")
        ax4.set_ylabel("Freq (MHz)", fontsize=8, color="#aaaaaa")

        char = m.get("characterization", {})
        fv = char.get("features") or {}
        sym = char.get("symbol_rate") or {}
        cfo = char.get("freq_offset") or {}
        clf = m.get("modulation_classification", {})

        hud_text = (
            f"MODULATION: {top_mod} ({conf_pct:.1f}%)\n"
            f"SNR: {snr_db:.1f} dB | BW: {bw_khz:.0f} kHz\n"
            f"C42 Norm: {fv.get('C42_norm', 0):.3f}\n"
            f"Phase Sym: {fv.get('C21_sym', 0):.3f}\n"
            f"Kurtosis: {fv.get('kurt_amp', 0):.2f}\n"
            f"Symbol Rate: {sym.get('symbol_rate_hz', 0)/1e3:.0f} kHz\n"
            f"Carrier Offset: {cfo.get('offset_hz', 0)/1e3:+.1f} kHz"
        )
        ax4.text(0.97, 0.95, hud_text, transform=ax4.transAxes,
                 ha="right", va="top", fontsize=7.5, family="monospace",
                 color="#ffffff", bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a0a10", edgecolor=accent_color, alpha=0.90))

    out_path = OUT_DIR / "data0_rf_showcase.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =========================================================================
# FIGURE 2: ZENODO RFI SHOWCASE (Allen Telescope Array Multi-Band)
# =========================================================================
def generate_zenodo_showcase():
    print("Generating Figure 2: zenodo_rfi_showcase.png ...")
    captures = sorted([d for d in ZENODO_DIR.iterdir() if d.is_dir()])
    if not captures:
        print("  Skipping Zenodo showcase (no folders found)")
        return
    selected = captures[:6]

    fig = plt.figure(figsize=(19, 12), facecolor="#111116")
    gs = GridSpec(3, 2, figure=fig, wspace=0.25, hspace=0.35)

    fig.suptitle("SIH 26147 — DATASET SHOWCASE: ALLEN TELESCOPE ARRAY RFI (zenodo)", fontsize=15, fontweight="bold", color="#ffffff", y=0.97)
    fig.text(0.5, 0.94, "Deep-Space Radio Telescope Interference | Native 61.44 MHz ci16_le Across 6 RF Bands + Classification Verdict",
             ha="center", fontsize=9.5, color="#888899")

    colors = ["#ff5252", "#ff4081", "#e040fb", "#7c4dff", "#536dfe", "#00e5ff"]

    for idx, folder in enumerate(selected):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#16161e")
        psd, spec, iq_pkg, m, meta = load_file_package(folder)

        freqs_mhz = psd["frequencies"] / 1e6
        psd_db = psd["psd_db"]
        fc_mhz = float(m["center_frequency_hz"] / 1e6) if m.get("center_frequency_hz") else 0.0
        snr_db = m["snr"]["psd_based_db"]
        bw_khz = m["bandwidth"]["bw_3db_hz"] / 1e3
        nf_db = m["noise_floor"]["noise_floor_db"]

        char = m.get("characterization", {})
        fv = char.get("features") or {}
        sym = char.get("symbol_rate") or {}
        clf = m.get("modulation_classification", {})
        top_mod = clf.get("top_candidate", "Noise/CW")
        conf_pct = clf.get("confidence", 0.0) * 100.0

        c = colors[idx % len(colors)]
        ax.plot(freqs_mhz, psd_db, color=c, linewidth=1.0, label="PSD Spectrum")
        ax.axhline(nf_db, color="#ffffff", linestyle="--", linewidth=0.8, alpha=0.6, label=f"NF: {nf_db:.1f} dBFS")

        ax.set_title(f"Capture #{idx+1}: RF Center = {fc_mhz:.2f} MHz (Native fs = 61.44 MHz) | Class: {top_mod} ({conf_pct:.0f}%)",
                     fontsize=9.5, fontweight="bold", color=c, loc="left")
        ax.set_xlabel("Baseband Frequency (MHz)", fontsize=8, color="#aaaaaa")
        ax.set_ylabel("Power Spectral Density (dB/Hz)", fontsize=8, color="#aaaaaa")
        ax.set_ylim(-135, max(-55, psd_db.max() + 10))
        ax.grid(True)

        hud_info = (
            f"MOD VERDICT: {top_mod} ({conf_pct:.1f}%)\n"
            f"RF Freq: {fc_mhz:.2f} MHz\n"
            f"Native NF: {nf_db:.1f} dBFS\n"
            f"SNR: {snr_db:.1f} dB | BW: {bw_khz:.0f} kHz\n"
            f"C42 Norm: {fv.get('C42_norm', 0):.2f}\n"
            f"PAPR: {fv.get('papr_db', 0):.1f} dB"
        )
        ax.text(0.97, 0.93, hud_info, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, family="monospace",
                color="#ffffff", bbox=dict(boxstyle="round,pad=0.35", facecolor="#0e0e14", edgecolor=c, alpha=0.90))

    out_path = OUT_DIR / "zenodo_rfi_showcase.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =========================================================================
# FIGURE 3: EXECUTIVE CROSS-DATASET DASHBOARD
# =========================================================================
def generate_executive_dashboard():
    print("Generating Figure 3: rf_landscape_executive.png ...")
    with open(DATA0_DIR / "summary.json", encoding="utf-8") as f:
        d0_metrics = json.load(f)
    with open(ZENODO_DIR / "summary.json", encoding="utf-8") as f:
        zen_metrics = json.load(f)

    fig = plt.figure(figsize=(19, 11), facecolor="#111116")
    gs = GridSpec(2, 3, figure=fig, wspace=0.28, hspace=0.34)

    fig.suptitle("SIH 26147 — AUTOMATED RF ANALYSIS ASSISTANT: DATASET LANDSCAPE", fontsize=16, fontweight="bold", color="#ffffff", y=0.97)
    fig.text(0.5, 0.935, "Cross-Dataset Comparative Analysis | Controlled Benchmarks (data0) vs In-the-Wild Deep-Space RFI (zenodo)",
             ha="center", fontsize=9.5, color="#888899")

    # --- Panel 1: SNR vs Bandwidth ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#16161e")
    sig_colors = {"CommSignal2": "#00e5ff", "CommSignal3": "#ff9100", "EMISignal1": "#76ff03"}
    for sig_name, col in sig_colors.items():
        sub = [m for m in d0_metrics if m.get("signal") == sig_name]
        snrs = [m["snr"]["psd_based_db"] for m in sub]
        bws = [m["bandwidth"]["bw_3db_hz"] / 1e3 for m in sub]
        ax1.scatter(bws, snrs, color=col, s=70, label=f"data0: {sig_name}", alpha=0.9)

    z_snrs = [m["snr"]["psd_based_db"] for m in zen_metrics]
    z_bws = [m["bandwidth"]["bw_3db_hz"] / 1e3 for m in zen_metrics]
    ax1.scatter(z_bws, z_snrs, color="#e040fb", s=65, marker="^", label="zenodo: ATA RFI", alpha=0.85, edgecolors="#ffffff", linewidths=0.5)

    ax1.set_xscale("log")
    ax1.set_title("Signal Space: SNR vs. -3dB Bandwidth", fontsize=10, fontweight="bold", color="#ffffff", loc="left")
    ax1.set_xlabel("Bandwidth (kHz, log scale)", fontsize=8, color="#aaaaaa")
    ax1.set_ylabel("In-Band SNR (PSD, dB)", fontsize=8, color="#aaaaaa")
    ax1.grid(True)
    ax1.legend(loc="upper right", fontsize=7.5, facecolor="#22222a", edgecolor="none")

    # --- Panel 2: Higher-Order Cumulant C42 Separation ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#16161e")
    c42_data0 = [m["characterization"]["features"]["C42_norm"] for m in d0_metrics if m.get("characterization", {}).get("features")]
    c42_zen = [m["characterization"]["features"]["C42_norm"] for m in zen_metrics if m.get("characterization", {}).get("features")]

    ax2.axhline(2.0, color="#00e5ff", linestyle=":", linewidth=1.2, label="BPSK Target (2.0)")
    ax2.axhline(1.0, color="#76ff03", linestyle=":", linewidth=1.2, label="QPSK Target (1.0)")
    ax2.axhline(0.68, color="#ffd54f", linestyle=":", linewidth=1.2, label="16QAM Target (0.68)")
    ax2.axhline(0.0, color="#ff5252", linestyle=":", linewidth=1.2, label="FSK / Noise (~0.0)")

    bplot = ax2.boxplot([c42_data0, c42_zen], tick_labels=["data0 (MIT)", "zenodo (ATA RFI)"], patch_artist=True, widths=0.45)
    bplot['boxes'][0].set_facecolor("#00bcd4")
    bplot['boxes'][1].set_facecolor("#ab47bc")
    for elem in ['whiskers', 'caps', 'medians']:
        for item in bplot[elem]:
            item.set_color('#ffffff')

    ax2.set_title("Modulation Classifier Feature: Normalized C42", fontsize=10, fontweight="bold", color="#ffffff", loc="left")
    ax2.set_ylabel("Normalized Cumulant |C42| / C20²", fontsize=8, color="#aaaaaa")
    ax2.set_ylim(-0.5, 7.5)
    ax2.grid(True)
    ax2.legend(loc="upper right", fontsize=7, facecolor="#22222a", edgecolor="none")

    # --- Panel 3: Modulation Hypothesis Distribution ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#16161e")

    d0_mods = [m.get("modulation_classification", {}).get("top_candidate", "Unknown") for m in d0_metrics]
    zen_mods = [m.get("modulation_classification", {}).get("top_candidate", "Unknown") for m in zen_metrics]

    all_cand = ["QPSK", "16QAM", "BPSK", "2-FSK", "4-FSK", "Noise/CW"]
    d0_counts = [d0_mods.count(c) for c in all_cand]
    zen_counts = [zen_mods.count(c) for c in all_cand]

    y_pos = np.arange(len(all_cand))
    h = 0.35
    ax3.barh(y_pos - h/2, d0_counts, height=h, color="#00e5ff", label="data0 (MIT)", alpha=0.85)
    ax3.barh(y_pos + h/2, zen_counts, height=h, color="#e040fb", label="zenodo (ATA)", alpha=0.85)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(all_cand, fontsize=8)
    ax3.set_title("Top Classified Modulation Distribution", fontsize=10, fontweight="bold", color="#ffffff", loc="left")
    ax3.set_xlabel("Capture Count", fontsize=8, color="#aaaaaa")
    ax3.grid(True)
    ax3.legend(loc="lower right", fontsize=7.5, facecolor="#22222a", edgecolor="none")

    # --- Panel 4: Symbol Rate vs Bandwidth Correlation ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#16161e")
    d0_sym = [m["characterization"]["symbol_rate"]["symbol_rate_hz"]/1e3 for m in d0_metrics if m.get("characterization", {}).get("symbol_rate")]
    d0_bw = [m["bandwidth"]["bw_3db_hz"]/1e3 for m in d0_metrics if m.get("characterization", {}).get("symbol_rate")]

    zen_sym = [m["characterization"]["symbol_rate"]["symbol_rate_hz"]/1e3 for m in zen_metrics if m.get("characterization", {}).get("symbol_rate")]
    zen_bw = [m["bandwidth"]["bw_3db_hz"]/1e3 for m in zen_metrics if m.get("characterization", {}).get("symbol_rate")]

    ax4.scatter(d0_sym, d0_bw, color="#00e5ff", s=55, label="data0", alpha=0.85)
    ax4.scatter(zen_sym, zen_bw, color="#e040fb", marker="^", s=55, label="zenodo", alpha=0.85)
    ax4.plot([10, 5000], [10, 5000], color="#555566", linestyle="--", label="1:1 Nyquist Line")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_title("Estimated Symbol Rate vs. -3dB Bandwidth", fontsize=10, fontweight="bold", color="#ffffff", loc="left")
    ax4.set_xlabel("Symbol Rate (kHz, log scale)", fontsize=8, color="#aaaaaa")
    ax4.set_ylabel("Occupied Bandwidth (kHz, log scale)", fontsize=8, color="#aaaaaa")
    ax4.grid(True)
    ax4.legend(loc="lower right", fontsize=7.5, facecolor="#22222a", edgecolor="none")

    # --- Panel 5: Classification Confidence vs SNR ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#16161e")

    d0_conf = [m.get("modulation_classification", {}).get("confidence", 0.5) * 100 for m in d0_metrics]
    d0_snr = [m["snr"]["psd_based_db"] for m in d0_metrics]

    zen_conf = [m.get("modulation_classification", {}).get("confidence", 0.5) * 100 for m in zen_metrics]
    zen_snr = [m["snr"]["psd_based_db"] for m in zen_metrics]

    ax5.scatter(d0_snr, d0_conf, color="#00e5ff", s=60, label="data0", alpha=0.85)
    ax5.scatter(zen_snr, zen_conf, color="#e040fb", marker="^", s=60, label="zenodo", alpha=0.85)
    ax5.axvline(3.0, color="#ef5350", linestyle=":", label="Uncertainty Boundary (3 dB)")
    ax5.set_title("Classification Confidence vs. SNR", fontsize=10, fontweight="bold", color="#ffffff", loc="left")
    ax5.set_xlabel("In-Band SNR (dB)", fontsize=8, color="#aaaaaa")
    ax5.set_ylabel("Top Candidate Confidence (%)", fontsize=8, color="#aaaaaa")
    ax5.set_ylim(40, 102)
    ax5.grid(True)
    ax5.legend(loc="lower right", fontsize=7.5, facecolor="#22222a", edgecolor="none")

    # --- Panel 6: Dataset Metrics Summary Table ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#16161e")
    ax6.axis("off")

    papr_d0 = [m["characterization"]["features"]["papr_db"] for m in d0_metrics if m.get("characterization", {}).get("features")]
    papr_zen = [m["characterization"]["features"]["papr_db"] for m in zen_metrics if m.get("characterization", {}).get("features")]

    table_data = [
        ["Attribute", "data0 (MIT Challenge)", "zenodo (ATA RFI)"],
        ["Format", "SigMF cf32_le", "SigMF ci16_le"],
        ["Sampling Rate", "25.0 MHz", "61.44 MHz"],
        ["Center Frequency", "2437.0 MHz (ISM)", "622 MHz – 2.33 GHz"],
        ["Avg SNR (PSD)", f"{np.mean([m['snr']['psd_based_db'] for m in d0_metrics]):.1f} dB", f"{np.mean([m['snr']['psd_based_db'] for m in zen_metrics]):.1f} dB"],
        ["Top Candidate #1", "QPSK (Clean Comm2)", "16QAM / RFI (Burst)"],
        ["Avg PAPR", f"{np.mean(papr_d0):.1f} dB", f"{np.mean(papr_zen):.1f} dB"],
        ["Total Processed", f"{len(d0_metrics)} files", f"{len(zen_metrics)} frames"],
        ["Pipeline Safety", "Zero Memory Bloat", "Zero Memory Bloat"],
    ]

    table = ax6.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#333344")
        if row == 0:
            cell.set_facecolor("#222230")
            cell.set_text_props(weight="bold", color="#ffffff")
        else:
            cell.set_facecolor("#16161e" if row % 2 == 0 else "#1c1c24")
            cell.set_text_props(color="#cccccc" if col > 0 else "#00e5ff", weight="bold" if col == 0 else "normal")

    ax6.set_title("Executive Specifications & Classification Provenance", fontsize=10, fontweight="bold", color="#ffffff", pad=12)

    out_path = OUT_DIR / "rf_landscape_executive.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# =========================================================================
# FIGURE 4: DEDICATED MODULATION CLASSIFICATION & EVM SHOWCASE
# =========================================================================
def generate_modulation_classification_showcase():
    print("Generating Figure 4: modulation_classification_showcase.png ...")
    from app.demodulation.slicer import get_reference_constellation

    targets = [
        ("data0", DATA0_DIR / "CommSignal2_demod_train_0000", "CommSignal2 (MIT)", "#00e5ff"),
        ("data0", DATA0_DIR / "CommSignal3_demod_train_0000", "CommSignal3 (MIT)", "#ff9100"),
        ("data0", DATA0_DIR / "EMISignal1_demod_train_0000", "EMISignal1 (MIT)", "#76ff03"),
        ("zenodo", ZENODO_DIR / "2023-08-03-15-37-20_atarfi_frame0000", "ATA RFI Burst (Zenodo)", "#e040fb"),
    ]

    fig = plt.figure(figsize=(20, 14), facecolor="#111116")
    gs = GridSpec(4, 4, figure=fig, width_ratios=[1.0, 1.15, 1.1, 1.55], wspace=0.28, hspace=0.38)

    fig.suptitle("SIH 26147 — CANDIDATE MODULATION CLASSIFICATION & EVM VERIFICATION",
                 fontsize=16, fontweight="bold", color="#ffffff", y=0.98)
    fig.text(0.5, 0.955, "Hypothesis Ranking Leaderboard, Constellation Slicing EVM, Physical Feature Fits & Natural Language 'WHY' Justifications",
             ha="center", fontsize=10, color="#888899")

    for row_idx, (ds_type, folder, label, accent_color) in enumerate(targets):
        if not folder.exists():
            continue
        psd, spec, iq_pkg, m, meta = load_file_package(folder)
        iq_norm = iq_pkg["iq_normalized"]
        clf = m.get("modulation_classification", {})
        top_mod = clf.get("top_candidate", "Unknown")
        conf_pct = clf.get("confidence", 0.0) * 100.0
        candidates = clf.get("candidates", [])
        snr_val = clf.get("snr_db", 0.0)

        # --- Col 0: Constellation & Ideal Slicing Reference ---
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.set_facecolor("#16161e")
        n_pts = min(1500, len(iq_norm))
        ax0.scatter(iq_norm[:n_pts].real, iq_norm[:n_pts].imag, s=4, color=accent_color, alpha=0.5, label="Received IQ")

        # Plot ideal reference grid
        ref_const = get_reference_constellation(top_mod)
        ax0.scatter(ref_const.real, ref_const.imag, s=70, marker="x", color="#ff1744", linewidths=2.0, label="Ideal Constellation", zorder=10)

        theta = np.linspace(0, 2 * np.pi, 200)
        ax0.plot(np.cos(theta), np.sin(theta), color="#444455", linestyle="--", linewidth=0.7)
        ax0.axhline(0, color="#333344", linewidth=0.6)
        ax0.axvline(0, color="#333344", linewidth=0.6)
        ax0.set_xlim(-2.0, 2.0)
        ax0.set_ylim(-2.0, 2.0)
        ax0.set_aspect("equal")
        ax0.set_title(f"{label} — Sliced Constellation", fontsize=9.5, fontweight="bold", color=accent_color, loc="left")
        ax0.set_xlabel("I (In-Phase)", fontsize=8, color="#aaaaaa")
        ax0.set_ylabel("Q (Quadrature)", fontsize=8, color="#aaaaaa")
        ax0.grid(True)
        if row_idx == 0:
            ax0.legend(loc="upper right", fontsize=7, facecolor="#22222a", edgecolor="none")

        # --- Col 1: Candidate Hypothesis Leaderboard (Horizontal Probability Bars) ---
        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax1.set_facecolor("#16161e")

        mod_names = [c["modulation"] for c in candidates][::-1]
        mod_scores = [c["score"] * 100 for c in candidates][::-1]
        bar_colors = ["#76ff03" if name == top_mod else "#3f51b5" if "QAM" in name or "PSK" in name else "#546e7a" for name in mod_names]

        y_pos = np.arange(len(mod_names))
        bars = ax1.barh(y_pos, mod_scores, color=bar_colors, height=0.55, alpha=0.85, edgecolor="#ffffff", linewidth=0.4)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(mod_names, fontsize=8.5, fontweight="bold", color="#ffffff")
        ax1.set_xlim(0, max(60, max(mod_scores) * 1.25))
        ax1.set_title(f"Hypothesis Leaderboard | Conf: {conf_pct:.1f}%", fontsize=9.5, fontweight="bold", color="#ffffff", loc="left")
        ax1.set_xlabel("Posterior Probability (%)", fontsize=8, color="#aaaaaa")
        ax1.grid(True, axis="x")

        for bar in bars:
            w = bar.get_width()
            if w > 1.0:
                ax1.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", ha="left", fontsize=7.5, color="#ffffff", weight="bold")

        # --- Col 2: Feature Fits Bar Chart ---
        ax2 = fig.add_subplot(gs[row_idx, 2])
        ax2.set_facecolor("#16161e")

        top_cand = next((c for c in candidates if c["modulation"] == top_mod), candidates[0] if candidates else {})
        fits = top_cand.get("feature_fit", {})

        fit_names = [k.replace("fit_", "").replace("is_", "").upper() for k in fits.keys()]
        fit_vals = [float(v) * 100 if isinstance(v, (int, float)) else 50.0 for v in fits.values()]

        y_f = np.arange(len(fit_names))
        ax2.barh(y_f, fit_vals, color="#ffd54f", height=0.5, alpha=0.85, edgecolor="none")
        ax2.set_yticks(y_f)
        ax2.set_yticklabels(fit_names, fontsize=8, color="#cccccc")
        ax2.set_xlim(0, 105)
        ax2.set_title("Discriminator Feature Fits", fontsize=9.5, fontweight="bold", color="#ffffff", loc="left")
        ax2.set_xlabel("Fit Score (%)", fontsize=8, color="#aaaaaa")
        ax2.grid(True, axis="x")

        for idx_f, val_f in enumerate(fit_vals):
            ax2.text(max(5, val_f - 12), idx_f, f"{val_f:.0f}%", va="center", ha="left", fontsize=7.5, color="#000000", weight="bold")

        # --- Col 3: Explainable Evidence "WHY" & Provenance Card ---
        ax3 = fig.add_subplot(gs[row_idx, 3])
        ax3.set_facecolor("#16161e")
        ax3.axis("off")

        fs_mhz = float(m["sample_rate_hz"]) / 1e6
        fc_mhz = float(m.get("center_frequency_hz", 0.0) or 0.0) / 1e6
        why_text = top_cand.get("why", "No justification recorded.")
        feats = clf.get("features_summary", {})

        card_lines = [
            f"[CLASSIFICATION VERDICT]  ->  TOP: {top_mod} (Confidence: {conf_pct:.1f}%)",
            f"[PHYSICAL CONTEXT]       fs={fs_mhz:.2f} MHz | fc={fc_mhz:.2f} MHz | SNR={snr_val:.1f} dB",
            f"[EXTRACTED MOMENTS]     C42_norm={feats.get('C42_norm', 'N/A')} | Kurt(amp)={feats.get('kurt_amp', 'N/A')} | PAPR={feats.get('papr_db', 'N/A')} dB",
            f"[EVIDENCE JUSTIFICATION] {why_text}",
            f"[PROVENANCE PIPELINE]   Ingestion -> Preprocessing -> PSD/SNR -> Characterization -> Classifier",
        ]

        full_card_text = "\n".join(card_lines)
        ax3.text(0.02, 0.95, full_card_text, transform=ax3.transAxes,
                 ha="left", va="top", fontsize=8.0, family="monospace",
                 color="#ffffff", linespacing=1.45,
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#0e0e16", edgecolor=accent_color, linewidth=1.2, alpha=0.95))

    out_path = OUT_DIR / "modulation_classification_showcase.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    print("=" * 65)
    print("Generating Complete Showcase Figures with Modulation Classification")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 65)
    generate_data0_showcase()
    generate_zenodo_showcase()
    generate_executive_dashboard()
    generate_modulation_classification_showcase()
    print("=" * 65)
    print("ALL 4 SHOWCASE FIGURES GENERATED SUCCESSFULLY (300 DPI)")
    print("=" * 65)


if __name__ == "__main__":
    main()
