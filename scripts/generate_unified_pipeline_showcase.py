#!/usr/bin/env python3
"""
scripts/generate_unified_pipeline_showcase.py
---------------------------------------------
Generates a master, publication-grade "Three-in-One" visualization figure:
  outputs/figures/01_unified_tri_dataset_pipeline_steps.png

Layout: 6 Rows (Pipeline Stages) x 3 Columns (Dataset Categories)
  Columns:
    1. Category A: data0 (MIT RF Challenge — Hardware OTA, 2.437 GHz, 25 MHz)
    2. Category B: zenodo (ATA Radio Astronomy — Deep-Space RFI, 622.96 MHz / 1.4 GHz, 61.44 MHz)
    3. Category C: rml (RadioML 2016.10a — Synthetic Channel Impairment, Baseband, 1.0 MHz)

  Rows (Pipeline Stages):
    Row 1: Stage 1 — Signal Ingestion & Raw Time-Domain Waveforms
    Row 2: Stage 2 — Power Spectral Density (PSD) & Noise Floor Detection
    Row 3: Stage 3 — Time-Frequency Waterfall Spectrograms
    Row 4: Stage 4 — Physical Feature Extraction & Higher-Order Cumulants (HOC)
    Row 5: Stage 5 — Candidate Modulation Classification & Hypothesis Leaderboards
    Row 6: Stage 6 — Constellation Slicing, Synchronization & Physical State
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from app.ingestion.parser import load_signal
from app.ingestion.rml_parser import RMLDatasetReader
from app.preprocessing.normalize import full_normalize
from app.detection.fft import compute_psd
from app.detection.spectrogram import compute_spectrogram
from app.detection.noise_floor import estimate_noise_floor
from app.detection.bandwidth import estimate_bandwidth
from app.detection.snr import estimate_snr_from_psd
from app.characterization.features import extract_features
from app.characterization.symbol_rate import estimate_symbol_rate
from app.modulation.classifier import ModulationClassifier


def set_dark_theme():
    plt.style.use("dark_background")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#2d3748"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#1e293b"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.5


def build_unified_figure(out_path: Path):
    print("=" * 80)
    print("SIH 26147 - Generating Master Unified Tri-Dataset Multi-Step Pipeline Figure")
    print("=" * 80)

    set_dark_theme()
    t0 = time.time()

    # 1. Load the 3 dataset categories
    print("[1/7] Ingesting Category Signals...")
    # Category A: data0 (CommSignal2 QPSK)
    path_data0 = Path("D:/dataset/data0/demod_train/CommSignal2/CommSignal2_demod_train_0000.sigmf-data")
    sig_data0 = load_signal(str(path_data0))

    # Category B: zenodo (ATA RFI Frame)
    path_zenodo = Path("D:/dataset/zenodo_frames/2023-08-03-15-37-20_atarfi/frame_0000.sigmf-data")
    sig_zenodo = load_signal(str(path_zenodo))

    # Category C: rml (RadioML 2016.10a QPSK @ +18dB)
    rml_reader = RMLDatasetReader()
    sig_rml = rml_reader.get_frame("QPSK", 18, frame_idx=0)

    categories = [
        ("data0 (MIT RF Challenge)", sig_data0, "#38bdf8", "Hardware Over-the-Air (25 MHz @ 2.437 GHz)"),
        ("zenodo (ATA Radio Astronomy)", sig_zenodo, "#c084fc", "Terrestrial RFI Archive (61.44 MHz @ 623 MHz)"),
        ("rml (RadioML 2016.10a)", sig_rml, "#4ade80", "Synthetic Impaired Channel (1.0 MHz @ Baseband)"),
    ]

    # Preprocessing
    norm_signals = [full_normalize(s) for _, s, _, _ in categories]

    # Initialize Figure
    fig = plt.figure(figsize=(24, 26), facecolor="#070a12")
    gs = gridspec.GridSpec(6, 3, height_ratios=[1.0, 1.0, 1.1, 0.95, 1.05, 1.0], hspace=0.32, wspace=0.22)

    stage_titles = [
        "STAGE 1: SIGNAL INGESTION & RAW TIME-DOMAIN WAVEFORMS",
        "STAGE 2: POWER SPECTRAL DENSITY (PSD) & NOISE FLOOR BASELINE",
        "STAGE 3: TIME-FREQUENCY WATERFALL SPECTROGRAMS",
        "STAGE 4: PHYSICAL MOMENTS & CUMULANT FINGERPRINTS",
        "STAGE 5: MODULATION CLASSIFICATION & HYPOTHESIS LEADERBOARD",
        "STAGE 6: CONSTELLATION SLICING & SYNCHRONIZATION STATE",
    ]

    clf = ModulationClassifier()

    for col_idx, (cat_name, sig, color, subtext) in enumerate(categories):
        norm_sig = norm_signals[col_idx]
        sr = sig.metadata.sample_rate or 1e6
        fc = sig.metadata.center_frequency or 0.0

        # DSP Computations
        fft_len = 1024 if sig.num_samples >= 1024 else 128
        spec_fft = 512 if sig.num_samples >= 512 else 64
        psd = compute_psd(sig, fft_size=fft_len)
        spec = compute_spectrogram(norm_sig, fft_size=spec_fft, max_time_bins=128)
        nf = estimate_noise_floor(psd, method="mad")
        snr_res = estimate_snr_from_psd(psd, nf)
        bw_res = estimate_bandwidth(psd, method="3db")
        feats = extract_features(sig.data, sample_rate=sr, snr_db=snr_res.snr_db)
        class_res = clf.classify_signal(norm_sig)

        # ---------------------------------------------------------------------
        # ROW 0: STAGE 1 — Ingestion & Raw Waveforms
        # ---------------------------------------------------------------------
        ax0 = fig.add_subplot(gs[0, col_idx])
        ax0.set_facecolor("#0b111e")
        
        # Plot first N samples
        n_plot = min(250, len(sig.data))
        t_us = np.arange(n_plot) / sr * 1e6
        ax0.plot(t_us, sig.data.real[:n_plot], color=color, alpha=0.9, linewidth=1.2, label="I (In-Phase)")
        ax0.plot(t_us, sig.data.imag[:n_plot], color="#fbbf24", alpha=0.75, linewidth=1.0, linestyle="--", label="Q (Quadrature)")
        ax0.plot(t_us, np.abs(sig.data[:n_plot]), color="#f43f5e", alpha=0.6, linewidth=1.0, label="|Envelope|")
        
        header_text = f"{cat_name}\n{subtext}"
        ax0.set_title(header_text, fontsize=11, fontweight="bold", color=color, pad=8)
        ax0.set_xlabel("Time (µs)", fontsize=9, color="#94a3b8")
        ax0.set_ylabel("Amplitude", fontsize=9, color="#94a3b8")
        ax0.grid(True, linestyle=":", alpha=0.4, color="#334155")
        ax0.legend(loc="upper right", fontsize=7.5, framealpha=0.4, facecolor="#1e293b")
        ax0.tick_params(labelsize=8, colors="#94a3b8")
        ax0.text(0.03, 0.08, f"N={sig.num_samples:,} samples | {sig.metadata.data_type.value}",
                 transform=ax0.transAxes, fontsize=8, color="#cbd5e1",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e293b", alpha=0.7, edgecolor="#334155"))

        # ---------------------------------------------------------------------
        # ROW 1: STAGE 2 — PSD & Noise Floor Detection
        # ---------------------------------------------------------------------
        ax1 = fig.add_subplot(gs[1, col_idx])
        ax1.set_facecolor("#0b111e")

        freq_axis = psd.frequencies / 1e6 if sr > 2e6 else psd.frequencies / 1e3
        freq_unit = "MHz" if sr > 2e6 else "kHz"
        ax1.plot(freq_axis, psd.psd_db, color=color, linewidth=1.2, label="Welch PSD")
        ax1.axhline(nf.noise_floor_db, color="#94a3b8", linestyle=":", linewidth=1.1, label=f"MAD Noise Floor ({nf.noise_floor_db:.1f} dB)")
        ax1.axhline(nf.noise_floor_db + 10, color="#f59e0b", linestyle="--", linewidth=1.0, alpha=0.8, label="Detection (+10dB)")

        # Shading occupied bandwidth
        if bw_res.lower_3db_hz is not None and bw_res.upper_3db_hz is not None:
            f_low = (bw_res.lower_3db_hz / 1e6) if sr > 2e6 else (bw_res.lower_3db_hz / 1e3)
            f_high = (bw_res.upper_3db_hz / 1e6) if sr > 2e6 else (bw_res.upper_3db_hz / 1e3)
            ax1.axvspan(f_low, f_high, color=color, alpha=0.15, label=f"-3dB BW ({bw_res.bandwidth_hz/1e3:.1f} kHz)")

        ax1.set_title(f"Spectral PSD & Noise Threshold (SNR={snr_res.snr_db:.1f} dB)", fontsize=10.5, fontweight="bold", color="#e2e8f0", pad=6)
        ax1.set_xlabel(f"Baseband Frequency ({freq_unit})", fontsize=9, color="#94a3b8")
        ax1.set_ylabel("Power Spectral Density (dB)", fontsize=9, color="#94a3b8")
        ax1.grid(True, linestyle=":", alpha=0.4, color="#334155")
        ax1.legend(loc="lower center", fontsize=7.5, framealpha=0.4, facecolor="#1e293b", ncol=2)
        ax1.tick_params(labelsize=8, colors="#94a3b8")

        # ---------------------------------------------------------------------
        # ROW 2: STAGE 3 — Waterfall Spectrogram
        # ---------------------------------------------------------------------
        ax2 = fig.add_subplot(gs[2, col_idx])
        ax2.set_facecolor("#0b111e")

        t_mesh = spec.time_bins * 1e3  # ms
        f_mesh = spec.frequencies / 1e6 if sr > 2e6 else spec.frequencies / 1e3
        
        vmin = np.percentile(spec.spectrogram_db, 5)
        vmax = np.percentile(spec.spectrogram_db, 99.5)
        
        im = ax2.pcolormesh(t_mesh, f_mesh, spec.spectrogram_db, cmap="turbo", shading="auto", vmin=vmin, vmax=vmax)
        ax2.set_title(f"Time-Frequency Waterfall ({spec.num_time_bins} x {spec.num_freq_bins})", fontsize=10.5, fontweight="bold", color="#e2e8f0", pad=6)
        ax2.set_xlabel("Time (ms)", fontsize=9, color="#94a3b8")
        ax2.set_ylabel(f"Frequency ({freq_unit})", fontsize=9, color="#94a3b8")
        ax2.tick_params(labelsize=8, colors="#94a3b8")

        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=7.5, colors="#94a3b8")
        cbar.set_label("dBFS", fontsize=8, color="#94a3b8")

        # ---------------------------------------------------------------------
        # ROW 3: STAGE 4 — Feature Fingerprints
        # ---------------------------------------------------------------------
        ax3 = fig.add_subplot(gs[3, col_idx])
        ax3.set_facecolor("#0b111e")

        feat_keys = ["C42_norm", "Phase Sym", "PAPR / 10", "Flatness x2", "Zero Cross"]
        c21_sym = feats.C21_abs / max(feats.C20, 1e-12)
        feat_vals = [
            min(2.5, feats.C42_norm),
            min(1.0, c21_sym),
            min(2.0, feats.papr_db / 10.0),
            min(2.0, feats.spectral_flatness * 2.0),
            min(1.0, feats.zero_crossing_rate * 2.0),
        ]

        bar_colors = [color, "#fbbf24", "#f43f5e", "#a855f7", "#06b6d4"]
        y_pos = np.arange(len(feat_keys))
        bars = ax3.barh(y_pos, feat_vals, color=bar_colors, alpha=0.85, height=0.55)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(feat_keys, fontsize=8.5, color="#cbd5e1")
        ax3.set_xlim(0, 2.6)
        ax3.set_title("Physical Modulation Fingerprint", fontsize=10.5, fontweight="bold", color="#e2e8f0", pad=6)
        ax3.set_xlabel("Normalized Feature Magnitude", fontsize=9, color="#94a3b8")
        ax3.grid(True, linestyle=":", alpha=0.4, color="#334155")
        ax3.tick_params(labelsize=8, colors="#94a3b8")

        # Annotate exact numeric values
        for b, v, raw_v in zip(bars, feat_vals, [feats.C42_norm, c21_sym, feats.papr_db, feats.spectral_flatness, feats.zero_crossing_rate]):
            suffix = " dB" if "PAPR" in str(raw_v) else ""
            ax3.text(v + 0.05, b.get_y() + b.get_height()/2, f"{raw_v:.2f}{suffix}",
                     va="center", fontsize=8, color="#f8fafc", fontweight="bold")

        # ---------------------------------------------------------------------
        # ROW 4: STAGE 5 — Modulation Classification Leaderboard
        # ---------------------------------------------------------------------
        ax4 = fig.add_subplot(gs[4, col_idx])
        ax4.set_facecolor("#0b111e")

        cands = class_res.candidates[:4]
        cand_names = [c.modulation for c in cands][::-1]
        cand_probs = [c.score * 100.0 for c in cands][::-1]

        y_p = np.arange(len(cand_names))
        c_bar_colors = ["#3b82f6" if n == class_res.top_candidate else "#475569" for n in cand_names]
        bars4 = ax4.barh(y_p, cand_probs, color=c_bar_colors, alpha=0.9, height=0.55)
        ax4.set_yticks(y_p)
        ax4.set_yticklabels(cand_names, fontsize=9, fontweight="bold", color="#f8fafc")
        ax4.set_xlim(0, 105)
        ax4.set_title(f"Ranked Hypotheses -> Top: {class_res.top_candidate} ({class_res.confidence*100:.1f}%)",
                      fontsize=10.5, fontweight="bold", color=color, pad=6)
        ax4.set_xlabel("Posterior Probability (%)", fontsize=9, color="#94a3b8")
        ax4.grid(True, linestyle=":", alpha=0.4, color="#334155")
        ax4.tick_params(labelsize=8, colors="#94a3b8")

        for b, p in zip(bars4, cand_probs):
            ax4.text(p + 1.5, b.get_y() + b.get_height()/2, f"{p:.1f}%",
                     va="center", fontsize=8, color="#f8fafc")

        # WHY callout box
        why_text = f"EVIDENCE WHY:\n{cands[0].why[:110]}..."
        ax4.text(0.04, 0.12, why_text, transform=ax4.transAxes, fontsize=7.6, color="#cbd5e1",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", alpha=0.85, edgecolor="#334155"))

        # ---------------------------------------------------------------------
        # ROW 5: STAGE 6 — Constellation Slicing & Synchronization State
        # ---------------------------------------------------------------------
        ax5 = fig.add_subplot(gs[5, col_idx])
        ax5.set_facecolor("#0b111e")

        n_const = min(1000, len(norm_sig.data))
        iq_pts = norm_sig.data[:n_const]
        ax5.scatter(iq_pts.real, iq_pts.imag, color=color, alpha=0.65, s=20, edgecolors="none", label="Recovered Symbols")
        
        # Unit circle & reference cross
        theta = np.linspace(0, 2*np.pi, 200)
        ax5.plot(np.cos(theta), np.sin(theta), color="#475569", linestyle=":", linewidth=1.0)
        ax5.axhline(0, color="#334155", linewidth=0.8)
        ax5.axvline(0, color="#334155", linewidth=0.8)

        # Ideal QPSK points if applicable
        ref_qpsk = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        ax5.scatter(ref_qpsk.real, ref_qpsk.imag, color="#f43f5e", marker="+", s=70, linewidths=2.0, label="Ideal Lattice")

        ax5.set_title("Constellation Plane & Decision Lattices", fontsize=10.5, fontweight="bold", color="#e2e8f0", pad=6)
        ax5.set_xlabel("In-Phase (I)", fontsize=9, color="#94a3b8")
        ax5.set_ylabel("Quadrature (Q)", fontsize=9, color="#94a3b8")
        ax5.set_xlim(-2.2, 2.2)
        ax5.set_ylim(-2.2, 2.2)
        ax5.grid(True, linestyle=":", alpha=0.4, color="#334155")
        ax5.legend(loc="upper right", fontsize=7.5, framealpha=0.4, facecolor="#1e293b")
        ax5.tick_params(labelsize=8, colors="#94a3b8")

        # Invariant note box
        inv_text = f"Origin: {sig.metadata.dataset_origin}\nDtype: {sig.metadata.data_type.value}"
        ax5.text(0.04, 0.08, inv_text, transform=ax5.transAxes, fontsize=8, color="#cbd5e1",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e293b", alpha=0.8, edgecolor="#334155"))

    # Stage Labels along left margin
    for row_idx, st_title in enumerate(stage_titles):
        fig.text(0.08, 0.985 - (row_idx * 0.163), st_title,
                 fontsize=11.5, fontweight="bold", color="#f8fafc",
                 bbox=dict(boxstyle="square,pad=0.35", facecolor="#1e293b", edgecolor="#38bdf8", linewidth=1.2))

    plt.suptitle("SIH 26147 — UNIFIED RF ANALYSIS PIPELINE: THREE-IN-ONE MULTI-STEP SHOWCASE\n"
                 "Simultaneous Step-by-Step Categorical Evaluation across MIT RF Challenge, ATA Radio Astronomy, and DeepSig RadioML",
                 fontsize=15, fontweight="bold", color="#ffffff", y=0.998)

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SUCCESS] Saved Master Unified Figure to {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Total generation time: {time.time() - t0:.2f} s")


def main():
    figures_dir = ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_file = figures_dir / "01_unified_tri_dataset_pipeline_steps.png"
    build_unified_figure(out_file)


if __name__ == "__main__":
    main()
