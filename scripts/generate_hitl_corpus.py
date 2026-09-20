#!/usr/bin/env python3
"""Build (or extend) the synthetic HITL benchmark corpus.

Build order §9.2: the generator produces standardized IQ bursts with full
hidden ground truth (modulation × interleaver × FEC × frame × SNR) through the
exact encoder chain the decoder consumes, then characterizes them with the SAME
estimates the live pipeline uses at advisory query time (PSD SNR, -3dB
bandwidth, 4-method symbol rate, 22-feature extraction). Every corpus entry is
a verified case with ``truth_source="synthetic"``.

Usage:
    python scripts/generate_hitl_corpus.py -o corpus.json
    python scripts/generate_hitl_corpus.py -o corpus.json --samples-per-config 2 --seed 1234
    python scripts/generate_hitl_corpus.py -o corpus.json --dry-run        # print the grid, build nothing

Prints the corpus size and source mix on completion.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.benchmark.generator import (  # noqa: E402
    MODULATIONS,
    INTERLEAVERS,
    FECS,
    FRAMES,
    SNR_POINTS,
    generate_corpus,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", default="corpus.json")
    p.add_argument("--samples-per-config", type=int, default=1)
    p.add_argument("-p", "--modulations", nargs="*", default=list(MODULATIONS))
    p.add_argument("--interleavers", nargs="*", default=list(INTERLEAVERS))
    p.add_argument("--fec", nargs="*", default=list(FECS))
    p.add_argument("--frames", nargs="*", default=list(FRAMES))
    p.add_argument("--snr", nargs="*", type=float, default=list(SNR_POINTS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    grid_size = (len(args.modulations) * len(args.interleavers)
                 * len(args.fec) * len(args.frames) * len(args.snr))
    print(f"grid: {len(args.modulations)} mods × {len(args.interleavers)} ilv × "
          f"{len(args.fec)} fec × {len(args.frames)} frame × {len(args.snr)} snr "
          f"= {grid_size} configs × {args.samples_per_config} sample(s) "
          f"= {grid_size * args.samples_per_config} entries")
    if args.dry_run:
        return 0

    t0 = time.time()
    corpus = generate_corpus(
        args.output,
        modulations=tuple(args.modulations),
        interleavers=tuple(args.interleavers),
        fes=tuple(args.fec),
        frames=tuple(args.frames),
        snrs=tuple(args.snr),
        samples_per_config=args.samples_per_config,
        seed=args.seed,
    )
    dest = corpus.save(args.output)
    print(f"corpus: {corpus.n} entries -> {dest} "
          f"({time.time() - t0:.1f}s), source mix: {corpus.source_mix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())