#!/usr/bin/env python3
"""
scripts/generate_progress_graph.py
-----------------------------------
Renders a pipeline-stage progress chart showing which SIH 26147 stages are
implemented + validated vs. still missing:

  outputs/figures/pipeline_progress.png

Each stage bar is colored by status:
    DONE+validated   (green)   —— ingestion .. decoding
    BUILDING (cyan)            —— (none currently)
    MISSING   (red)            —— hypothesis / evidence / API decode / GUI
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "outputs" / "figures" / "pipeline_progress.png"

STAGES = [
    ("1 Ingestion (IQ/WAV/SigMF/RML)", "done"),
    ("2 Preprocessing", "done"),
    ("3 Detection (PSD/NF/seg/SNR/BW)", "done"),
    ("4 Characterization (HOC/Rsym/CFO)", "done"),
    ("5 Modulation classification", "done"),
    ("6 Demodulation DSP (sync/EVM/LLR)", "done"),
    ("7 Decoding (interleaver/FEC/CRC)", "done"),
    ("8 Hypothesis engine + evidence", "missing"),
    ("9 API decode exposure", "missing"),
    ("10 GUI / report exporter", "missing"),
]

COLORS = {"done": "#2ecc71", "building": "#1abc9c", "missing": "#e74c3c"}
LABELS = {"done": "BUILT + VALIDATED", "building": "IN PROGRESS", "missing": "MISSING"}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=200)
    fig.patch.set_facecolor("#0b1021")
    ax.set_facecolor("#0b1021")

    names = [s[0] for s in STAGES]
    statuses = [s[1] for s in STAGES]
    y = list(range(len(STAGES)))[::-1]
    colors = [COLORS[s[1]] for s in STAGES]

    ax.barh(y, [1.0] * len(STAGES), color=colors, height=0.62, alpha=0.92, edgecolor="none")
    for yi, (name, st) in zip(y, STAGES):
        ax.text(0.5, yi, name, ha="center", va="center", color="#0b1021",
                fontsize=10.5, fontweight="bold")
        x = 0.5 + 0.36 + 0.001
        ax.text(x, yi, LABELS[st], ha="left", va="center", color=COLORS[st],
                fontsize=9.5, fontweight="bold", family="monospace")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(STAGES) - 0.4)
    ax.axis("off")
    ax.text(0.5, len(STAGES) + 0.18,
            "SIH 26147 — Automated RF Analysis Assistant · Pipeline Progress",
            ha="center", va="bottom", color="#eaf0ff", fontsize=14, fontweight="bold")
    ax.text(0.5, len(STAGES) + 1.0,
            "validate_characterization (PASS) · validate_demodulation 14/14 · "
            "validate_decoding 11/11 · pytest 86 passed",
            ha="center", va="bottom", color="#9fb4e6", fontsize=9.5, family="monospace")
    ax.text(0.5, -0.62, STAGES[-1][0].split(" ")[0] + " stages current — data feed (HEAD 7c0b068)",
            ha="center", va="bottom", color="#5f7399", fontsize=8.5, family="monospace")

    fig.savefig(OUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())