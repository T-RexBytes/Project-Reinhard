#!/usr/bin/env python3
"""Generate astonishing, work-ready detailed figures from plot_data.

Creates 5 publication-quality figures:
  1) psd_detailed.png       — PSD with noise floor, segments, peaks, BW, SNR annotations
  2) waterfall_detailed.png — Enhanced spectrograms with proper contrast + colorbars
  3) iq_detailed.png        — Constellation + time domain + histograms
  4) dashboard_summary.png  — Cross-signal comparison (SNR/BW/segments)
  5) report_per_signal.png  — Per-signal deep dive (3 pages, one per signal type)

DPI=300, tight layout, professional styling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

PLOT_DATA = Path("outputs/plot_data")
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Professional style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

COLORS = {
    "psd": "#1f77b4",
    "noise": "#d62728",
    "threshold": "#ff7f0e",
    "segment": ["#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d"],
    "bw": "#2ca02c",
    "peak": "#e377c2",
}

SIGNALS_ORDER = ["CommSignal2", "CommSignal3", "EMISignal1"]
SIG_COLOR = {"CommSignal2": "#1f77b4", "CommSignal3": "#ff7f0e", "EMISignal1": "#2ca02c"}

def load_entry(name):
    base = PLOT_DATA / name
    d = np.load(base / "psd.npz")
    s = np.load(base / "spectrogram.npz")
    iq = np.load(base / "iq_sample.npz")
    with open(base / "metrics.json") as f:
        m = json.load(f)
    return d, s, iq, m

def get_three_representatives():
    # One per signal type, index 0000 for consistency
    return [
        "CommSignal2_demod_train_0000",
        "CommSignal3_demod_train_0000",
        "EMISignal1_demod_train_0000",
    ]

# -------------------------------------------------------------------------
# 1) PSD Detailed — 3 rows, rich annotations
# -------------------------------------------------------------------------
def fig_psd_detailed():
    reps = get_three_representatives()
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    fig.suptitle("Power Spectral Density — Detailed Analysis (Welch, 1024-pt FFT)", fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94, "Red dashed = noise floor (MAD)  |  Orange shaded = detected segments  |  Green span = -3 dB bandwidth  |  Pink ★ = segment peaks", ha="center", fontsize=8, style="italic", color="#555555")

    for ax, name in zip(axes, reps):
        d, _, _, m = load_entry(name)
        freq_khz = d["frequencies"] / 1e3
        psd = d["psd_db"]
        nf = m["noise_floor"]["noise_floor_db"]
        thr = m["segmentation"]["detection_threshold_db"]
        segs = m["segmentation"]["segments"]
        bw = m["bandwidth"]
        snr = m["snr"]["psd_based_db"]

        # PSD curve
        ax.plot(freq_khz, psd, color=COLORS["psd"], linewidth=0.9, label="PSD", zorder=3)
        # Noise floor + threshold
        ax.axhline(nf, color=COLORS["noise"], linestyle="--", linewidth=1.2, label=f"Noise floor {nf:.1f} dB", zorder=4)
        ax.axhline(thr, color=COLORS["threshold"], linestyle=":", linewidth=1.1, label=f"Threshold {thr:.1f} dB", alpha=0.9, zorder=4)

        # Segments shaded
        for idx, seg in enumerate(segs):
            c = COLORS["segment"][idx % len(COLORS["segment"])]
            ax.axvspan(seg["start_freq_hz"]/1e3, seg["end_freq_hz"]/1e3, alpha=0.22, color=c, zorder=1)
            # Peak marker
            ax.plot(seg["peak_freq_hz"]/1e3, seg["peak_power_db"], marker="*", color=COLORS["peak"], markersize=8, markeredgecolor="black", markeredgewidth=0.4, zorder=5)

        # -3dB bandwidth span (centered around PSD max)
        if bw["bw_3db_lower_hz"] is not None and bw["bw_3db_upper_hz"] is not None:
            ax.axvspan(bw["bw_3db_lower_hz"]/1e3, bw["bw_3db_upper_hz"]/1e3, alpha=0.12, color=COLORS["bw"], hatch="///", edgecolor=COLORS["bw"], zorder=2, label=f"-3dB BW {bw['bw_3db_hz']/1e3:.0f} kHz")

        # Annotations box
        textstr = f"SNR: {snr:.1f} dB\nBW (-3dB): {bw['bw_3db_hz']/1e3:.0f} kHz\nBW (frac): {bw['bw_frac_hz']/1e3:.0f} kHz ({bw['occupancy_percent']:.0f}%)\nSegs: {len(segs)}  |  PSD max: {psd.max():.1f} dB"
        ax.text(0.02, 0.96, textstr, transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92))

        # Segment count labels inside shaded regions (if wide enough)
        for seg in segs:
            bw_khz = seg["bandwidth_hz"]/1e3
            if bw_khz > 150:  # only label wide segments to avoid clutter
                cx = seg["center_freq_hz"]/1e3
                ax.text(cx, thr+1.5, f"{bw_khz:.0f}k", ha="center", va="bottom", fontsize=6, color="#7a3a00",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))

        ax.set_title(f"{name} — {m['signal']} | {m['task']} | {m['duration_ms']:.2f} ms, {m['num_samples']:,} samples", fontsize=10, loc="left", pad=8)
        ax.set_ylabel("PSD (dB)")
        ax.set_xlim(freq_khz.min(), freq_khz.max())
        ax.set_ylim(psd.min()-2, psd.max()+4)
        ax.legend(loc="upper right", fontsize=6.5, framealpha=0.9, ncol=2)

    axes[-1].set_xlabel("Frequency offset (kHz) — centered at carrier")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT_DIR / "01_psd_detailed.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "01_psd_detailed.pdf", bbox_inches="tight")
    plt.close()
    print("Saved 01_psd_detailed.png/pdf")

# -------------------------------------------------------------------------
# 2) Waterfall Detailed — enhanced contrast, colorbars, segment overlays
# -------------------------------------------------------------------------
def fig_waterfall_detailed():
    reps = get_three_representatives()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Spectrogram / Waterfall — Time-Frequency Analysis (512-pt FFT, 128 time bins)", fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94, "Yellow = high power  |  Viridis scale  |  Horizontal orange = segment frequency extents", ha="center", fontsize=8, style="italic", color="#555555")

    for ax, name in zip(axes, reps):
        _, s, _, m = load_entry(name)
        freq_khz = s["frequencies"] / 1e3
        t_ms = s["time_bins"] * 1e3
        spec = s["spectrogram_db"]

        # Robust contrast: 2nd to 98th percentile
        vmin, vmax = np.percentile(spec, [2, 98])
        # Clamp to avoid outliers compressing scale
        vmin = max(vmin, spec.min()+5)
        vmax = min(vmax, spec.max()-1)

        im = ax.imshow(spec, aspect="auto", origin="lower",
                       extent=[t_ms[0], t_ms[-1], freq_khz[0], freq_khz[-1]],
                       cmap="viridis", vmin=vmin, vmax=vmax, interpolation="bilinear")

        # Overlay segment frequency spans as horizontal translucent bands
        for seg in m["segmentation"]["segments"]:
            ax.axhspan(seg["start_freq_hz"]/1e3, seg["end_freq_hz"]/1e3, alpha=0.12, color="orange", edgecolor="orange", linewidth=0.6)

        ax.set_title(f"{name} — {m['signal']} | SNR {m['snr']['psd_based_db']:.1f} dB, {m['segmentation']['num_segments']} segments", fontsize=10, loc="left")
        ax.set_ylabel("Freq (kHz)")
        # colorbar per axes
        cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
        cbar.set_label("Power (dB)", fontsize=7)
        cbar.ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT_DIR / "02_waterfall_detailed.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "02_waterfall_detailed.pdf", bbox_inches="tight")
    plt.close()
    print("Saved 02_waterfall_detailed.png/pdf")

# -------------------------------------------------------------------------
# 3) IQ Detailed — constellation + time domain + hist
# -------------------------------------------------------------------------
def fig_iq_detailed():
    reps = get_three_representatives()
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle("IQ Domain — Constellation, Time Domain & Amplitude Analysis (first 4096 samples)", fontsize=13, fontweight="bold", y=0.98)
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38, top=0.93, bottom=0.06, left=0.06, right=0.98)

    for row, name in enumerate(reps):
        _, _, iq, m = load_entry(name)
        raw = iq["iq_raw"]
        norm = iq["iq_normalized"]
        # Use normalized for constellation
        # Downsample for constellation scatter to avoid overplotting: plot ~2000 points
        idx = np.linspace(0, len(norm)-1, 2200, dtype=int)
        # Time domain: first 400 samples
        t = np.arange(400)

        # Col 0: I/Q vs time (raw)
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.plot(t, raw[:400].real, color="#1f77b4", linewidth=0.8, label="I")
        ax0.plot(t, raw[:400].imag, color="#ff7f0e", linewidth=0.8, label="Q")
        ax0.set_title(f"{name} — I/Q vs Time (raw)", fontsize=8)
        ax0.set_xlabel("Sample"); ax0.set_ylabel("Amplitude")
        ax0.legend(fontsize=6, loc="upper right")
        ax0.set_xlim(0, 400)

        # Col 1: Constellation (normalized)
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.scatter(norm[idx].real, norm[idx].imag, s=3, alpha=0.45, c="#1f77b4", edgecolors="none")
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 200)
        ax1.plot(np.cos(theta), np.sin(theta), color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="unit")
        ax1.set_title("Constellation (normalized)", fontsize=8)
        ax1.set_xlabel("I"); ax1.set_ylabel("Q")
        ax1.set_aspect("equal", adjustable="box")
        ax1.grid(True, alpha=0.3)
        # Annotate SNR/BW
        ax1.text(0.02, 0.98, f"SNR {m['snr']['psd_based_db']:.1f} dB\nBW {m['bandwidth']['bw_3db_hz']/1e3:.0f} kHz", transform=ax1.transAxes, va="top", ha="left", fontsize=6.5,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#cccccc"))

        # Col 2: Amplitude histogram
        ax2 = fig.add_subplot(gs[row, 2])
        amp = np.abs(norm)
        ax2.hist(amp, bins=60, color="#2ca02c", edgecolor="white", linewidth=0.4, alpha=0.85, density=True)
        ax2.axvline(np.mean(amp), color="red", linestyle="--", linewidth=1, label=f"mean {np.mean(amp):.2f}")
        ax2.set_title("|A| Histogram", fontsize=8)
        ax2.set_xlabel("|A|"); ax2.set_ylabel("Density")
        ax2.legend(fontsize=6)

        # Col 3: Power vs time + phase histogram inset
        ax3 = fig.add_subplot(gs[row, 3])
        # Power envelope for first 1500 samples
        pwr = np.abs(norm[:1500])**2
        ax3.plot(pwr, color="#9467bd", linewidth=0.7)
        ax3.fill_between(np.arange(len(pwr)), pwr, alpha=0.18, color="#9467bd")
        ax3.set_title("Instant Power (first 1500)", fontsize=8)
        ax3.set_xlabel("Sample"); ax3.set_ylabel("Power")
        ax3.set_xlim(0, 1500)

    plt.savefig(OUT_DIR / "03_iq_detailed.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "03_iq_detailed.pdf", bbox_inches="tight")
    plt.close()
    print("Saved 03_iq_detailed.png/pdf")

# -------------------------------------------------------------------------
# 4) Dashboard Summary — cross-signal comparison, publication ready
# -------------------------------------------------------------------------
def fig_dashboard():
    # Load all summary
    with open(PLOT_DATA / "summary.json") as f:
        all_m = json.load(f)

    # Sort by signal then SNR
    sig_order = {"CommSignal2": 0, "CommSignal3": 1, "EMISignal1": 2}
    all_m.sort(key=lambda x: (sig_order.get(x["signal"], 99), x["file"]))

    labels = [f"{m['signal'][:4]}..{m['index']:04d}\n{m['task'][:4]}" for m in all_m]
    snr = [m["snr"]["psd_based_db"] for m in all_m]
    nf = [m["noise_floor"]["noise_floor_db"] for m in all_m]
    bw = [m["bandwidth"]["bw_3db_hz"]/1e3 for m in all_m]
    segs = [m["segmentation"]["num_segments"] for m in all_m]
    psdmax = [m["psd_stats"]["max_db"] for m in all_m]
    colors = [SIG_COLOR[m["signal"]] for m in all_m]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("RF Analysis Pipeline — Cross-Signal Summary Dashboard (14 representative files)", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94, "Curated subset (not full 33k) — lean export, ~4.7 MB  |  PSD-based SNR, MAD noise floor, -3 dB BW, segmentation count", ha="center", fontsize=8, style="italic", color="#555555")

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35, top=0.90, bottom=0.10, left=0.06, right=0.98)

    # SNR
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(np.arange(len(labels)), snr, color=colors, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=6.5, rotation=0)
    ax.set_title("SNR (PSD-based, dB)", fontweight="bold", fontsize=10)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("dB"); ax.set_ylim(min(snr)-3, max(snr)+4)
    # legend for signal colors
    for sig, c in SIG_COLOR.items():
        ax.bar([], [], color=c, label=sig)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    # Bandwidth
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(np.arange(len(labels)), bw, color=colors, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=6.5)
    ax.set_title("-3 dB Bandwidth (kHz)", fontweight="bold", fontsize=10)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("kHz"); ax.set_ylim(0, max(bw)*1.15)
    # Highlight EMISignal1 low BW
    ax.text(0.98, 0.96, "EMISignal1 → narrowband", transform=ax.transAxes, ha="right", va="top", fontsize=7, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6ffe6", edgecolor="#2ca02c", alpha=0.9))

    # Segments
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar(np.arange(len(labels)), segs, color=colors, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.bar_label(bars, padding=2, fontsize=7)
    ax.set_title("Detected Segments (count)", fontweight="bold", fontsize=10)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("Count"); ax.set_ylim(0, max(segs)+1.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Noise floor
    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(np.arange(len(labels)), nf, color=colors, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=6.5)
    ax.set_title("Noise Floor (MAD, dB)", fontweight="bold", fontsize=10)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("dB"); ax.set_ylim(min(nf)-2, max(nf)+2)

    # PSD max
    ax = fig.add_subplot(gs[1, 1])
    bars = ax.bar(np.arange(len(labels)), psdmax, color=colors, edgecolor="black", linewidth=0.4, alpha=0.9)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=6.5)
    ax.set_title("PSD Peak (dB)", fontweight="bold", fontsize=10)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylabel("dB")

    # Table + insights
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    # Build markdown-like table as text
    header = f"{'File':<18} {'SNR':>6} {'BW':>7} {'NF':>7} {'Segs':>4}\n" + "-"*48 + "\n"
    rows = ""
    for m in all_m:
        fname = m["signal"][:4] + f"_{m['index']:04d}"
        rows += f"{fname:<18} {m['snr']['psd_based_db']:6.1f} {m['bandwidth']['bw_3db_hz']/1e3:7.0f} {m['noise_floor']['noise_floor_db']:7.1f} {m['segmentation']['num_segments']:4d}\n"
    ax.text(0.02, 0.98, header+rows, transform=ax.transAxes, va="top", ha="left", fontsize=6.2, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#cccccc"))
    ax.set_title("Metrics Table", fontweight="bold", fontsize=10, pad=10)
    # Insights box below table
    fig.text(0.68, 0.18, "Insights:  • EMISignal1 = high SNR, low BW, very low NF\n           • CommSignal3 = widest BW (fragile)\n           • Frame captures = longer, similar stats", fontsize=7, style="italic",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8e1", edgecolor="#ffcc00", alpha=0.95))

    plt.savefig(OUT_DIR / "04_dashboard_summary.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "04_dashboard_summary.pdf", bbox_inches="tight")
    plt.close()
    print("Saved 04_dashboard_summary.png/pdf")

# -------------------------------------------------------------------------
# 5) Per-signal deep dive — one comprehensive page per signal type
# -------------------------------------------------------------------------
def fig_per_signal_reports():
    for sig_name in SIGNALS_ORDER:
        name = sig_name + "_demod_train_0000"
        d, s, iq, m = load_entry(name)
        raw = iq["iq_raw"]; norm = iq["iq_normalized"]
        freq_khz = d["frequencies"]/1e3
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(sig_name + " — Deep Dive Report  |  " + m["file"] + " | " + str(m["num_samples"]) + " samples @ 25 MHz (" + str(round(m["duration_ms"],2)) + " ms)", fontsize=12, fontweight="bold", y=0.98)
        gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32, top=0.92, bottom=0.07, left=0.06, right=0.98)

        # Top-left: PSD detailed
        ax = fig.add_subplot(gs[0, 0])
        psd = d["psd_db"]; nf = m["noise_floor"]["noise_floor_db"]; thr = m["segmentation"]["detection_threshold_db"]
        ax.plot(freq_khz, psd, color=COLORS["psd"], lw=0.9)
        ax.axhline(nf, color=COLORS["noise"], ls="--", lw=1)
        ax.axhline(thr, color=COLORS["threshold"], ls=":", lw=1)
        for idx, seg in enumerate(m["segmentation"]["segments"]):
            ax.axvspan(seg["start_freq_hz"]/1e3, seg["end_freq_hz"]/1e3, alpha=0.18, color=COLORS["segment"][idx % len(COLORS["segment"])])
            ax.plot(seg["peak_freq_hz"]/1e3, seg["peak_power_db"], "*", color=COLORS["peak"], ms=7, mec="black", mew=0.4)
        ax.set_title("PSD — " + str(m["segmentation"]["num_segments"]) + " segments", fontsize=9, fontweight="bold")
        ax.set_xlabel("Freq (kHz)"); ax.set_ylabel("PSD (dB)")
        ax.set_xlim(freq_khz.min(), freq_khz.max())

        # Top-middle: Spectrogram
        ax = fig.add_subplot(gs[0, 1])
        freq2 = s["frequencies"]/1e3; t_ms = s["time_bins"]*1e3; spec = s["spectrogram_db"]
        vmin, vmax = np.percentile(spec, [2, 98])
        im = ax.imshow(spec, aspect="auto", origin="lower", extent=[t_ms[0], t_ms[-1], freq2[0], freq2[-1]], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="bilinear")
        for seg in m["segmentation"]["segments"]:
            ax.axhspan(seg["start_freq_hz"]/1e3, seg["end_freq_hz"]/1e3, alpha=0.10, color="orange")
        ax.set_title("Spectrogram", fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Freq (kHz)")
        plt.colorbar(im, ax=ax, pad=0.01, aspect=30, label="dB")

        # Top-right: Metrics + provenance
        ax = fig.add_subplot(gs[0, 2])
        ax.axis("off")
        # Metrics box
        bw = m["bandwidth"]; snr = m["snr"]
        txt = (
            "PIPELINE METRICS\n"
            + "="*28 + "\n"
            + "Noise floor (MAD):  " + f"{nf:.2f} dB\n"
            + "Threshold:          " + f"{thr:.2f} dB\n"
            + "SNR (PSD):          " + f"{snr['psd_based_db']:.2f} dB\n"
            + "  signal: " + f"{snr['psd_signal_power_db']:.1f} dB  noise: {snr['psd_noise_power_db']:.1f} dB\n"
            + "SNR (M2M4):         " + (f"{snr['m2m4_db']:.2f} dB" if snr["m2m4_db"] else "n/a") + "\n"
            + "BW -3dB:            " + f"{bw['bw_3db_hz']/1e3:.1f} kHz\n"
            + "  [" + f"{bw['bw_3db_lower_hz']/1e3:.0f}" + " → " + f"{bw['bw_3db_upper_hz']/1e3:.0f}" + " kHz]\n"
            + "BW frac:            " + f"{bw['bw_frac_hz']/1e3:.1f} kHz ({bw['occupancy_percent']:.1f}%)\n"
            + "Segments:           " + f"{m['segmentation']['num_segments']}\n"
            + "PSD max/min:        " + f"{m['psd_stats']['max_db']:.1f} / {m['psd_stats']['min_db']:.1f} dB\n"
            + "\nSEGMENTS\n" + "="*28 + "\n"
        )
        for i, seg in enumerate(m["segmentation"]["segments"]):
            txt += f"#{i} {seg['center_freq_hz']/1e3:+.0f}k  BW {seg['bandwidth_hz']/1e3:.0f}k  SNR {seg['snr_db']:.1f}dB\n"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f8ff", edgecolor="#1f77b4", alpha=0.96))
        ax.set_title("Metrics & Segments", fontsize=9, fontweight="bold")

        # Bottom-left: Constellation
        ax = fig.add_subplot(gs[1, 0])
        idx = np.linspace(0, len(norm)-1, 2500, dtype=int)
        ax.scatter(norm[idx].real, norm[idx].imag, s=4, alpha=0.5, c="#1f77b4", edgecolors="none")
        th = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), "r--", lw=0.8, alpha=0.6)
        ax.set_title("Constellation (normalized, 2500 pts)", fontsize=9, fontweight="bold")
        ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal", adjustable="box")

        # Bottom-middle: I/Q time
        ax = fig.add_subplot(gs[1, 1])
        t = np.arange(500)
        ax.plot(t, raw[:500].real, color="#1f77b4", lw=0.8, label="I")
        ax.plot(t, raw[:500].imag, color="#ff7f0e", lw=0.8, label="Q")
        ax.set_title("I/Q vs Time — first 500 samples (raw)", fontsize=9, fontweight="bold")
        ax.set_xlabel("Sample"); ax.set_ylabel("Amplitude"); ax.legend(fontsize=7)

        # Bottom-right: Power + amplitude hist
        ax = fig.add_subplot(gs[1, 2])
        # Two axes: hist and power
        ax_hist = ax
        amp = np.abs(norm)
        ax_hist.hist(amp, bins=50, color="#2ca02c", edgecolor="white", lw=0.4, alpha=0.85, density=True)
        ax_hist.axvline(np.mean(amp), color="red", ls="--", lw=1, label=f"mean {np.mean(amp):.2f}")
        ax_hist.set_title("|A| Histogram (normalized)", fontsize=9, fontweight="bold")
        ax_hist.set_xlabel("|A|"); ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=7)

        plt.savefig(OUT_DIR / ("05_report_" + sig_name + ".png"), dpi=300, bbox_inches="tight")
        plt.savefig(OUT_DIR / ("05_report_" + sig_name + ".pdf"), bbox_inches="tight")
        plt.close()
        print("Saved 05_report_" + sig_name + ".png/pdf")

if __name__ == "__main__":
    print("Generating detailed figures ->", OUT_DIR)
    fig_psd_detailed()
    fig_waterfall_detailed()
    fig_iq_detailed()
    fig_dashboard()
    fig_per_signal_reports()
    total = sum(f.stat().st_size for f in OUT_DIR.glob("*.png"))/1024/1024
    print(f"Done. {len(list(OUT_DIR.glob('*.png')))} PNGs, {total:.1f} MB in {OUT_DIR}")
