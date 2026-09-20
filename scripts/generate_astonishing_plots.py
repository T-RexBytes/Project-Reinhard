#!/usr/bin/env python3
"""
scripts/generate_astonishing_plots.py
-------------------------------------
Generates ultra-high-impact, publication-grade RF visualizations:
  1) 01_3d_spectral_terrain_showcase.png:
     3D Elevation Waterfall & Spectral Mountain Ridges (Time x Freq x Power).
  2) 02_eye_diagram_and_phase_trajectories.png:
     Digital Signal Synchronization (I/Q Eye Diagrams, Phase Trellis Trajectories, Cyclic Squaring Spectrum).
  3) 03_rf_intelligence_constellation_radar.png:
     Polar Spider Feature Fingerprint, 3D Clustering Space, Sliced Constellation, and AI Reasoning Provenance.

High-DPI (300 DPI), modern dark sci-fi aesthetic, luminous vibrant color palettes.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA0_DIR = ROOT / "outputs" / "data0"
ZENODO_DIR = ROOT / "outputs" / "zenodo"
OUT_FIG_DIR = ROOT / "outputs" / "figures"
OUT_CLASS_DIR = ROOT / "outputs" / "classification"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_CLASS_DIR.mkdir(parents=True, exist_ok=True)

# Luminous modern dark aesthetic
plt.style.use("dark_background")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.edgecolor": "#333344",
    "axes.linewidth": 0.8,
    "grid.color": "#20202a",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})


def load_pkg(folder: Path):
    psd_d = np.load(folder / "psd.npz")
    spec_d = np.load(folder / "spectrogram.npz")
    iq_d = np.load(folder / "iq_sample.npz")
    with open(folder / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    return psd_d, spec_d, iq_d, metrics


# =========================================================================
# PLOT 1: 3D SPECTRAL TERRAIN SHOWCASE
# =========================================================================
def generate_3d_spectral_terrain():
    print("Generating 3D Spectral Terrain: 01_3d_spectral_terrain_showcase.png ...")
    targets = [
        (DATA0_DIR / "CommSignal2_demod_train_0000", "CommSignal2: QPSK Carrier Mesa (+18.9 dB SNR)", cm.turbo, (28, -55)),
        (DATA0_DIR / "CommSignal3_demod_train_0000", "CommSignal3: Wideband 16QAM / Multi-Carrier Canyon", cm.plasma, (32, -45)),
        (DATA0_DIR / "EMISignal1_demod_train_0000", "EMISignal1: Needle-Sharp Narrowband EMI Tone (+24 dB)", cm.viridis, (25, -60)),
        (ZENODO_DIR / "2023-08-03-15-37-20_atarfi_frame0000", "ATA Telescope RFI: Deep Space Terrestrial Spike (622.96 MHz)", cm.inferno, (30, -50)),
    ]

    fig = plt.figure(figsize=(22, 14), facecolor="#0a0a10")
    fig.suptitle("SIH 26147 — 3D SPECTRAL TERRAIN & ELEVATION WATERFALL SHOWCASE",
                 fontsize=18, fontweight="bold", color="#ffffff", y=0.98)
    fig.text(0.5, 0.955, "3D Surface Topography (Time x Frequency x Power) | Visualizing Signal Power Rising Above Noise Floor Plateau",
             ha="center", fontsize=11, color="#8888aa")

    gs = GridSpec(2, 2, figure=fig, wspace=0.15, hspace=0.22)

    for idx, (folder, title, colormap, (elev, azim)) in enumerate(targets):
        if not folder.exists():
            continue
        psd, spec, iq_pkg, m = load_pkg(folder)
        spec_db = spec["spectrogram_db"]
        time_bins_ms = spec["time_bins"] * 1e3
        freqs_mhz = spec["frequencies"] / 1e6

        # Downsample for crisp, fast 3D mesh rendering
        t_step = max(1, len(time_bins_ms) // 48)
        f_step = max(1, len(freqs_mhz) // 48)
        T = time_bins_ms[::t_step]
        F = freqs_mhz[::f_step]
        Z = spec_db[::f_step, ::t_step]

        # Meshgrid: X = Time (ms), Y = Frequency (MHz)
        X_mesh, Y_mesh = np.meshgrid(T, F)

        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection="3d")
        ax.set_facecolor("#0a0a10")
        ax.xaxis.pane.set_facecolor("#0e0e16")
        ax.yaxis.pane.set_facecolor("#0e0e16")
        ax.zaxis.pane.set_facecolor("#0a0a10")
        ax.xaxis.pane.set_edgecolor("#222233")
        ax.yaxis.pane.set_edgecolor("#222233")
        ax.zaxis.pane.set_edgecolor("#222233")

        # Floor contour offset
        z_min = float(np.min(Z))
        z_max = float(np.max(Z))

        surf = ax.plot_surface(X_mesh, Y_mesh, Z, cmap=colormap,
                               linewidth=0.2, antialiased=True, edgecolors="#ffffff22",
                               rstride=1, cstride=1, alpha=0.92)

        # Projected floor contour
        ax.contour(X_mesh, Y_mesh, Z, zdir="z", offset=z_min, cmap=colormap, alpha=0.5, levels=12)

        # Detection threshold translucent plane
        thr = m["segmentation"].get("detection_threshold_db", z_min + 10.0)
        if z_min < thr < z_max:
            ax.plot_surface(X_mesh, Y_mesh, np.full_like(Z, thr),
                            color="#ffd54f", alpha=0.15, linewidth=0)

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#ffffff", pad=10)
        ax.set_xlabel("Time (ms)", fontsize=8.5, color="#aaaaaa", labelpad=8)
        ax.set_ylabel("Frequency (MHz)", fontsize=8.5, color="#aaaaaa", labelpad=8)
        ax.set_zlabel("PSD (dB/Hz)", fontsize=8.5, color="#aaaaaa", labelpad=8)
        ax.set_zlim(z_min, z_max + 4)

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=14, pad=0.08)
        cbar.ax.tick_params(labelsize=7, colors="#aaaaaa")
        cbar.set_label("dB/Hz", fontsize=7.5, color="#aaaaaa")

    out_path = OUT_FIG_DIR / "01_3d_spectral_terrain_showcase.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    shutil.copy(out_path, OUT_CLASS_DIR / out_path.name)


# =========================================================================
# PLOT 2: EYE DIAGRAMS & VECTOR PHASE TRAJECTORIES
# =========================================================================
def generate_eye_and_phase_trajectories():
    print("Generating Eye Diagrams & Phase Trellises: 02_eye_diagram_and_phase_trajectories.png ...")
    from app.demodulation.synchronizer import costas_loop, derotate_carrier
    from app.demodulation.slicer import get_reference_constellation

    targets = [
        ("CommSignal2_demod_train_0000", "CommSignal2 (QPSK @ 25 MHz)", "QPSK", "#00e5ff"),
        ("CommSignal3_demod_train_0000", "CommSignal3 (16QAM / Multi-Carrier)", "16QAM", "#ff9100"),
    ]

    fig = plt.figure(figsize=(22, 12), facecolor="#0c0c12")
    fig.suptitle("SIH 26147 — RECEIVER DSP DYNAMICS: EYE DIAGRAMS & CONTINUOUS PHASE TRELLIS",
                 fontsize=17, fontweight="bold", color="#ffffff", y=0.98)
    fig.text(0.5, 0.955, "Receiver Demodulation Physics | In-Phase & Quadrature Eye Openings, Phase Transition Vectors, and Baud Squaring Spectrum",
             ha="center", fontsize=10.5, color="#8888aa")

    gs = GridSpec(2, 4, figure=fig, width_ratios=[1.1, 1.1, 1.15, 1.25], wspace=0.25, hspace=0.32)

    for row_idx, (folder_name, title, mod_name, accent) in enumerate(targets):
        folder = DATA0_DIR / folder_name
        if not folder.exists():
            continue
        psd, spec, iq_pkg, m = load_pkg(folder)
        iq_norm = iq_pkg["iq_normalized"]
        fs = float(m["sample_rate_hz"])
        char = m.get("characterization", {})
        sym = char.get("symbol_rate", {})
        sym_rate = sym.get("symbol_rate_hz", 500e3)
        if sym_rate <= 0:
            sym_rate = 500e3

        sps = max(4, int(round(fs / sym_rate)))
        eye_span = 2 * sps  # 2-symbol window for eye diagram

        # Synchronize carrier phase via Costas loop
        iq_synced, _ = costas_loop(iq_norm, modulation=mod_name, loop_bw=0.02)
        n_trace = min(3000, len(iq_synced))
        iq_use = iq_synced[:n_trace]

        # --- Col 0: In-Phase (I) Eye Diagram ---
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.set_facecolor("#12121a")
        t_eye = np.linspace(-1.0, 1.0, eye_span)

        num_eyes = len(iq_use) // eye_span
        for k in range(min(num_eyes, 120)):
            seg = iq_use[k * eye_span : (k + 1) * eye_span].real
            ax0.plot(t_eye, seg, color=accent, alpha=0.18, linewidth=0.9)

        ax0.axvline(0, color="#ffffff44", linestyle=":", label="Strobe Optimum")
        ax0.axhline(0, color="#333344", linewidth=0.6)
        ax0.set_title(f"{title} — In-Phase Eye Diagram", fontsize=9.5, fontweight="bold", color=accent, loc="left")
        ax0.set_xlabel("Time (Symbol Intervals 2T)", fontsize=8, color="#aaaaaa")
        ax0.set_ylabel("In-Phase Amplitude", fontsize=8, color="#aaaaaa")
        ax0.set_ylim(-2.0, 2.0)
        ax0.grid(True)
        if row_idx == 0:
            ax0.legend(loc="upper right", fontsize=7, facecolor="#1e1e28", edgecolor="none")

        # --- Col 1: Quadrature (Q) Eye Diagram ---
        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax1.set_facecolor("#12121a")
        for k in range(min(num_eyes, 120)):
            seg = iq_use[k * eye_span : (k + 1) * eye_span].imag
            ax1.plot(t_eye, seg, color="#ff4081", alpha=0.18, linewidth=0.9)

        ax1.axvline(0, color="#ffffff44", linestyle=":")
        ax1.axhline(0, color="#333344", linewidth=0.6)
        ax1.set_title(f"{title} — Quadrature Eye Diagram", fontsize=9.5, fontweight="bold", color="#ff4081", loc="left")
        ax1.set_xlabel("Time (Symbol Intervals 2T)", fontsize=8, color="#aaaaaa")
        ax1.set_ylabel("Quadrature Amplitude", fontsize=8, color="#aaaaaa")
        ax1.set_ylim(-2.0, 2.0)
        ax1.grid(True)

        # --- Col 2: Continuous Phase Trellis / Vector Trajectory ---
        ax2 = fig.add_subplot(gs[row_idx, 2])
        ax2.set_facecolor("#12121a")

        # Plot line trajectory connecting consecutive samples
        n_traj = min(400, len(iq_use))
        i_tr = iq_use[:n_traj].real
        q_tr = iq_use[:n_traj].imag
        ax2.plot(i_tr, q_tr, color="#ffffff", alpha=0.35, linewidth=0.8, zorder=1)

        # Gradient color dots on top
        sc = ax2.scatter(i_tr, q_tr, c=np.arange(n_traj), cmap="cool", s=14, alpha=0.85, zorder=2)

        # Overlay ideal points
        ref = get_reference_constellation(mod_name)
        ax2.scatter(ref.real, ref.imag, s=85, marker="x", color="#ffd54f", linewidths=2.2, label="Ideal Points", zorder=3)

        theta = np.linspace(0, 2 * np.pi, 150)
        ax2.plot(np.cos(theta), np.sin(theta), color="#444455", linestyle="--", linewidth=0.7)
        ax2.axhline(0, color="#333344", linewidth=0.6)
        ax2.axvline(0, color="#333344", linewidth=0.6)
        ax2.set_xlim(-2.1, 2.1)
        ax2.set_ylim(-2.1, 2.1)
        ax2.set_aspect("equal")
        ax2.set_title("Vector Phase Trajectory in Complex Plane", fontsize=9.5, fontweight="bold", color="#ffffff", loc="left")
        ax2.set_xlabel("In-Phase (I)", fontsize=8, color="#aaaaaa")
        ax2.set_ylabel("Quadrature (Q)", fontsize=8, color="#aaaaaa")
        ax2.grid(True)
        if row_idx == 0:
            ax2.legend(loc="upper right", fontsize=7, facecolor="#1e1e28", edgecolor="none")

        # --- Col 3: Cyclic Baud-Rate Squaring Spectrum ---
        ax3 = fig.add_subplot(gs[row_idx, 3])
        ax3.set_facecolor("#12121a")

        # |IQ|^2 spectrum reveals sharp clock spectral line at baud rate
        x_sq = np.abs(iq_norm[:8192]) ** 2
        x_sq -= np.mean(x_sq)
        win = np.hanning(len(x_sq))
        sq_fft = np.abs(np.fft.rfft(x_sq * win)) ** 2
        sq_freqs = np.fft.rfftfreq(len(x_sq), d=1.0 / fs) / 1e3  # kHz

        # Plot only up to 2.5 MHz
        valid = sq_freqs <= 2500.0
        f_sub = sq_freqs[valid]
        p_sub = 10.0 * np.log10(np.maximum(sq_fft[valid], 1e-12))
        p_sub -= np.max(p_sub)  # Normalize peak to 0 dB

        ax3.plot(f_sub, p_sub, color="#76ff03", linewidth=1.1, label="|x(t)|² Clock Spectrum")
        ax3.axvline(sym_rate / 1e3, color="#ff1744", linestyle="--", linewidth=1.2, label=f"Baud Clock ({sym_rate/1e3:.0f} kHz)")
        ax3.set_title(f"Cyclostationary Clock Spectrum | Peak: {sym_rate/1e3:.0f} kHz", fontsize=9.5, fontweight="bold", color="#ffffff", loc="left")
        ax3.set_xlabel("Baud Modulation Frequency (kHz)", fontsize=8, color="#aaaaaa")
        ax3.set_ylabel("Normalized Power (dB)", fontsize=8, color="#aaaaaa")
        ax3.set_ylim(-35, 3)
        ax3.grid(True)
        ax3.legend(loc="upper right", fontsize=7.5, facecolor="#1e1e28", edgecolor="none")

    out_path = OUT_FIG_DIR / "02_eye_diagram_and_phase_trajectories.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    shutil.copy(out_path, OUT_CLASS_DIR / out_path.name)


# =========================================================================
# PLOT 3: POLAR SPIDER FINGERPRINTS & AI CLASSIFICATION TOPOLOGY
# =========================================================================
def generate_rf_intelligence_radar():
    print("Generating RF Intelligence Master Poster: 03_rf_intelligence_constellation_radar.png ...")
    with open(DATA0_DIR / "summary.json", encoding="utf-8") as f:
        d0_metrics = json.load(f)
    with open(ZENODO_DIR / "summary.json", encoding="utf-8") as f:
        zen_metrics = json.load(f)

    fig = plt.figure(figsize=(22, 13), facecolor="#09090e")
    fig.suptitle("SIH 26147 — RF INTELLIGENCE, POLAR FEATURE FINGERPRINT & CLUSTERING TOPOLOGY",
                 fontsize=17, fontweight="bold", color="#ffffff", y=0.98)
    fig.text(0.5, 0.955, "Explainable Multi-Dimensional Discriminators | Polar Radar Signatures, 3D Physical Clustering Space & Decision Boundaries",
             ha="center", fontsize=10.5, color="#8888aa")

    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.1, 1.1, 1.2], wspace=0.26, hspace=0.30)

    # --- Panel 1: Polar Spider Radar Fingerprint for CommSignal2 (QPSK) ---
    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    ax1.set_facecolor("#101018")

    categories = ["C42 Norm\n(target 1.0)", "Phase Sym\n(|C21|/C20)", "PAPR (dB)\n(scale /10)", "Amp Var\n(std/mean)", "Inst Freq Dev\n(x5)", "Flatness\n(Wiener)"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Prototype theoretical QPSK
    qpsk_proto = [1.0, 0.05, 0.35, 0.38, 0.15, 0.05]
    qpsk_proto += qpsk_proto[:1]

    # Measured CommSignal2
    m_c2 = next(m for m in d0_metrics if m.get("signal") == "CommSignal2")
    fv2 = m_c2["characterization"]["features"]
    c2_vals = [
        min(1.5, fv2["C42_norm"]),
        min(1.0, fv2.get("C21_sym", 0.01)),
        min(1.0, fv2["papr_db"] / 10.0),
        min(1.0, fv2["amplitude_variance_norm"]),
        min(1.0, fv2.get("instantaneous_freq_std_norm", 0.05) * 5.0),
        min(1.0, fv2.get("spectral_flatness", 0.05)),
    ]
    c2_vals += c2_vals[:1]

    ax1.plot(angles, qpsk_proto, color="#ffd54f", linewidth=1.5, linestyle="--", label="Ideal QPSK Prototype")
    ax1.fill(angles, qpsk_proto, color="#ffd54f", alpha=0.10)

    ax1.plot(angles, c2_vals, color="#00e5ff", linewidth=2.0, label="Measured CommSignal2")
    ax1.fill(angles, c2_vals, color="#00e5ff", alpha=0.25)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=7.5, color="#dddddd")
    ax1.set_ylim(0, 1.2)
    ax1.set_title("Polar Feature Radar: CommSignal2 vs. QPSK Prototype", fontsize=10, fontweight="bold", color="#00e5ff", pad=15)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=7.5, facecolor="#161622", edgecolor="none")

    # --- Panel 2: Polar Spider Radar Fingerprint for Zenodo ATA RFI ---
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    ax2.set_facecolor("#101018")

    # Prototype Noise / CW
    noise_proto = [0.05, 0.05, 0.85, 0.90, 0.80, 0.85]
    noise_proto += noise_proto[:1]

    m_zen = zen_metrics[0]
    fvz = m_zen["characterization"]["features"]
    zen_vals = [
        min(1.5, fvz["C42_norm"] / 5.0),
        min(1.0, fvz.get("C21_sym", 0.02)),
        min(1.0, fvz["papr_db"] / 20.0),
        min(1.0, fvz["amplitude_variance_norm"]),
        min(1.0, fvz.get("instantaneous_freq_std_norm", 0.15) * 3.0),
        min(1.0, fvz.get("spectral_flatness", 0.5)),
    ]
    zen_vals += zen_vals[:1]

    ax2.plot(angles, noise_proto, color="#ff5252", linewidth=1.5, linestyle="--", label="Thermal Noise Prototype")
    ax2.fill(angles, noise_proto, color="#ff5252", alpha=0.10)

    ax2.plot(angles, zen_vals, color="#e040fb", linewidth=2.0, label="Measured ATA RFI")
    ax2.fill(angles, zen_vals, color="#e040fb", alpha=0.25)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=7.5, color="#dddddd")
    ax2.set_ylim(0, 1.2)
    ax2.set_title("Polar Feature Radar: ATA RFI vs. Noise Prototype", fontsize=10, fontweight="bold", color="#e040fb", pad=15)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=7.5, facecolor="#161622", edgecolor="none")

    # --- Panel 3: 3D Physical Clustering Space (C42 x Kurtosis x PAPR) ---
    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    ax3.set_facecolor("#09090e")
    ax3.xaxis.pane.set_facecolor("#0e0e16")
    ax3.yaxis.pane.set_facecolor("#0e0e16")
    ax3.zaxis.pane.set_facecolor("#09090e")
    ax3.xaxis.pane.set_edgecolor("#222233")
    ax3.yaxis.pane.set_edgecolor("#222233")
    ax3.zaxis.pane.set_edgecolor("#222233")

    # Group data0 by signal
    colors_map = {"CommSignal2": "#00e5ff", "CommSignal3": "#ff9100", "EMISignal1": "#76ff03"}
    for sig_name, col in colors_map.items():
        sub = [m for m in d0_metrics if m.get("signal") == sig_name]
        c42s = [m["characterization"]["features"]["C42_norm"] for m in sub]
        kurts = [m["characterization"]["features"].get("kurt_amp", -0.8) for m in sub]
        paprs = [m["characterization"]["features"]["papr_db"] for m in sub]
        ax3.scatter(c42s, kurts, paprs, color=col, s=75, label=f"data0: {sig_name}", alpha=0.9)

    # Zenodo
    z_c42 = [min(10, m["characterization"]["features"]["C42_norm"]) for m in zen_metrics]
    z_kurt = [min(15, m["characterization"]["features"].get("kurt_amp", 5)) for m in zen_metrics]
    z_papr = [m["characterization"]["features"]["papr_db"] for m in zen_metrics]
    ax3.scatter(z_c42, z_kurt, z_papr, color="#e040fb", s=70, marker="^", label="zenodo: ATA RFI", alpha=0.9, edgecolors="#ffffff", linewidths=0.5)

    ax3.view_init(elev=24, azim=-50)
    ax3.set_title("3D Feature Clustering Space", fontsize=10.5, fontweight="bold", color="#ffffff", pad=10)
    ax3.set_xlabel("C42 Norm", fontsize=8, color="#aaaaaa", labelpad=6)
    ax3.set_ylabel("Kurtosis |IQ|", fontsize=8, color="#aaaaaa", labelpad=6)
    ax3.set_zlabel("PAPR (dB)", fontsize=8, color="#aaaaaa", labelpad=6)
    ax3.legend(loc="upper left", fontsize=7.5, facecolor="#161622", edgecolor="none")

    # --- Panel 4: Amplitude Distribution (Single-Shell vs. Multi-Ring Separation) ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#101018")

    # Load raw normalized amplitudes
    _, _, iq2, _ = load_pkg(DATA0_DIR / "CommSignal2_demod_train_0000")
    _, _, iq3, _ = load_pkg(DATA0_DIR / "CommSignal3_demod_train_0000")

    amp2 = np.abs(iq2["iq_normalized"][:5000])
    amp3 = np.abs(iq3["iq_normalized"][:5000])

    bins_a = np.linspace(0, 2.5, 45)
    ax4.hist(amp2, bins=bins_a, density=True, color="#00e5ff", alpha=0.6, label="CommSignal2 (QPSK: Single Shell)", edgecolor="none")
    ax4.hist(amp3, bins=bins_a, density=True, color="#ff9100", alpha=0.5, label="CommSignal3 (16QAM: Multi-Ring)", edgecolor="none")

    ax4.set_title("Amplitude Density: Single-Shell vs Multi-Ring", fontsize=10, fontweight="bold", color="#ffffff", loc="left")
    ax4.set_xlabel("Normalized Envelope |IQ|", fontsize=8.5, color="#aaaaaa")
    ax4.set_ylabel("Probability Density", fontsize=8.5, color="#aaaaaa")
    ax4.grid(True)
    ax4.legend(loc="upper right", fontsize=7.5, facecolor="#161622", edgecolor="none")

    # --- Panel 5: Phase Variance vs. Normalized Instantaneous Frequency Deviation ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#101018")

    for sig_name, col in colors_map.items():
        sub = [m for m in d0_metrics if m.get("signal") == sig_name]
        p_var = [m["characterization"]["features"].get("phase_variance", 1.0) for m in sub]
        f_dev = [m["characterization"]["features"].get("instantaneous_freq_std_norm", 0.05) for m in sub]
        ax5.scatter(f_dev, p_var, color=col, s=65, label=sig_name, alpha=0.9)

    zen_p_var = [m["characterization"]["features"].get("phase_variance", 50.0) for m in zen_metrics]
    zen_f_dev = [m["characterization"]["features"].get("instantaneous_freq_std_norm", 0.2) for m in zen_metrics]
    ax5.scatter(zen_f_dev, zen_p_var, color="#e040fb", s=65, marker="^", label="ATA RFI", alpha=0.9)

    ax5.set_yscale("log")
    ax5.set_title("Phase Variance vs. Dimensionless Freq Deviation", fontsize=10, fontweight="bold", color="#ffffff", loc="left")
    ax5.set_xlabel("Normalized Freq Dev (cycles/sample)", fontsize=8.5, color="#aaaaaa")
    ax5.set_ylabel("Phase Variance (rad², log scale)", fontsize=8.5, color="#aaaaaa")
    ax5.grid(True)
    ax5.legend(loc="upper left", fontsize=7.5, facecolor="#161622", edgecolor="none")

    # --- Panel 6: AI Explainable Reasoning Architecture Diagram ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#101018")
    ax6.axis("off")

    exec_summary_text = (
        "┌─────────────────────────────────────────────────────────────┐\n"
        "│       AI MODULATION REASONING ENGINE ARCHITECTURE           │\n"
        "├─────────────────────────────────────────────────────────────┤\n"
        "│ 1. Dimensionless Invariance:                                │\n"
        "│    Normalized frequency deviation and cumulants match       │\n"
        "│    identically across 25 MHz (MIT) and 61.44 MHz (Zenodo).  │\n"
        "│                                                             │\n"
        "│ 2. Sub-Gaussian vs. Multi-Ring Separation:                  │\n"
        "│    Kurt(|IQ|) < -0.4 uniquely isolates QPSK single-shell;   │\n"
        "│    Kurt(|IQ|) >= -0.2 separates 16QAM multi-tier grid.      │\n"
        "│                                                             │\n"
        "│ 3. Multiplicative Cumulant Gating:                          │\n"
        "│    FSK hypothesis score is strictly zeroed when C42 > 0.45. │\n"
        "│                                                             │\n"
        "│ 4. Honest Uncertainty & Noise Floor Gating:                 │\n"
        "│    When SNR < 3 dB, Noise/CW is boosted to eliminate false  │\n"
        "│    modulations on telescope background sky scans.           │\n"
        "│                                                             │\n"
        "│ 5. Physical Provenance Guarantee:                           │\n"
        "│    Every DSP transform appends a verifiable ProvenanceStep. │\n"
        "└─────────────────────────────────────────────────────────────┘"
    )

    ax6.text(0.02, 0.95, exec_summary_text, transform=ax6.transAxes,
             ha="left", va="top", fontsize=8.5, family="monospace",
             color="#00e5ff", linespacing=1.35,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0b0b14", edgecolor="#76ff03", linewidth=1.2))

    out_path = OUT_FIG_DIR / "03_rf_intelligence_constellation_radar.png"
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    shutil.copy(out_path, OUT_CLASS_DIR / out_path.name)


def main():
    print("=" * 65)
    print("Generating Next-Gen Astonishing RF Visualization Figures")
    print(f"Output directory: {OUT_FIG_DIR}")
    print("=" * 65)
    generate_3d_spectral_terrain()
    generate_eye_and_phase_trajectories()
    generate_rf_intelligence_radar()
    print("=" * 65)
    print("ALL 3 ASTONISHING FIGURES GENERATED SUCCESSFULLY (300 DPI)")
    print("=" * 65)


if __name__ == "__main__":
    main()
