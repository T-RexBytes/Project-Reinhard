"""
app/benchmark/advisor.py
------------------------
The Benchmark Advisor (Build order §9.3).

Read-only retrieval over a verified corpus. It NEVER writes into the evidence
graph and NEVER contributes to any composite score — it narrates what verified
prior cases say about the current signal and stops when the corpus is too thin
to be trustworthy (§ risk table: advisory suppressed < ~5 neighbours or beyond
a distance threshold).

`AdvisoryRecord` is the machine-readable output; `format_advisory_text` is the
templated natural-language fallback the LLM narration would otherwise produce
(verbatim-number post-check → numbers always appear in the record verbatim).
"""

from __future__ import annotations

from dataclasses import field
from typing import Optional

import numpy as np
from pydantic import BaseModel

from app.benchmark.corpus import (
    BenchmarkCorpus,
    TRUTH_SOURCE_WEIGHT,
)

MIN_NEIGHBOURS = 5
MAX_DISTANCE = 1.5          # beyond this the neighbours aren't really "similar"
TOP_MODULATIONS = 3
DEFAULT_K = 12


class AdvisoryRecord(BaseModel):
    """What the corpus says about one signal at a gate."""

    neighbours: int = 0
    feature_distance_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    verified_outcomes: dict[str, int] = field(default_factory=dict)
    chains_that_validated: list[dict] = field(default_factory=list)
    caveat: str = ""
    source_mix: dict[str, int] = field(default_factory=dict)
    suppressed: bool = True      # True when the corpus is too thin to advise
    suppression_reason: str = ""

    def to_dict(self) -> dict:
        return self.model_dump()

    @property
    def top_modulation(self) -> Optional[str]:
        if not self.verified_outcomes:
            return None
        return max(self.verified_outcomes, key=lambda m: self.verified_outcomes[m])


class BenchmarkAdvisor:
    """k-NN advisory lookup over a verified BenchmarkCorpus."""

    def __init__(self, corpus: BenchmarkCorpus | None = None,
                 k: int = DEFAULT_K, min_neighbours: int = MIN_NEIGHBOURS,
                 max_distance: float = MAX_DISTANCE):
        self.corpus = corpus or BenchmarkCorpus()
        self.k = int(k)
        self.min_neighbours = int(min_neighbours)
        self.max_distance = float(max_distance)

    # -- core consult --------------------------------------------------------
    def consult(self, query_vector, gate: str = "") -> AdvisoryRecord:
        """Return the advisory for a query vector, or a suppressed record."""
        if self.corpus.n == 0:
            return AdvisoryRecord(neighbours=0, suppressed=True,
                                  suppression_reason="empty corpus")
        qv = np.asarray(query_vector, dtype=np.float32)
        distances, indices = self.corpus.query(qv, k=self.k)
        valid = [i for i, d in zip(indices, distances)
                 if d <= self.max_distance]
        if len(valid) < self.min_neighbours:
            return AdvisoryRecord(
                neighbours=len(valid),
                feature_distance_range=[float(distances.min()), float(distances.max())]
                if len(distances) else [0.0, 0.0],
                suppressed=True,
                suppression_reason=(
                    f"{len(valid)} neighbours within distance {self.max_distance} "
                    f"(need ≥ {self.min_neighbours})"),
            )

        rec = self._build_record([int(i) for i in valid], distances)
        rec.suppressed = False
        return rec

    def _build_record(self, idx: list[int], distances, gate: str = "") -> AdvisoryRecord:
        entries = [self.corpus.entries[i] for i in idx]
        # Truth-source weighting: human_confirmed counts less than verified entries.
        outcomes: dict[str, float] = {}
        for e in entries:
            outcomes[e.verified_modulation] = (
                outcomes.get(e.verified_modulation, 0.0)
                + TRUTH_SOURCE_WEIGHT.get(e.truth_source, 1.0)
            )
        ordered = sorted(outcomes.items(), key=lambda kv: -kv[1])[:TOP_MODULATIONS]
        verified_outcomes = {str(m): int(round(w)) for m, w in ordered}

        chains: dict[str, dict] = {}
        for e in entries:
            if not e.verified_chain:
                continue
            c = chains.setdefault(e.verified_chain, {"count": 0, "crc_pass": 0})
            c["count"] += 1
            if e.crc_validated:
                c["crc_pass"] += 1

        chains_validated = [
            {
                "chain": name,
                "count": info["count"],
                "crc_pass_rate": round(info["crc_pass"] / info["count"], 3),
            }
            for name, info in sorted(chains.items(), key=lambda kv: -kv[1]["count"])
        ]

        dists = np.asarray([float(d) for d in distances], dtype=float)
        caveat_bits = []
        snrs = [e.snr_db for e in entries]
        if snrs:
            mean_snr = float(np.mean(snrs))
            if mean_snr > 5.0 + 0.5:
                caveat_bits.append(
                    f"{len(snrs)} neighbours average SNR {mean_snr:.1f} dB")
        source_mix: dict[str, int] = {}
        for e in entries:
            source_mix[e.truth_source] = source_mix.get(e.truth_source, 0) + 1
        if source_mix.get("human_confirmed", 0) > len(entries) // 2:
            caveat_bits.append("majority of neighbours are human-confirmed, not ground truth")

        # The referenced distance array is the neighbour distances at `idx`.
        dist_array = dists
        return AdvisoryRecord(
            neighbours=len(entries),
            feature_distance_range=[round(float(dist_array.min()), 4),
                                    round(float(dist_array.max()), 4)],
            verified_outcomes=verified_outcomes,
            chains_that_validated=chains_validated,
            caveat="; ".join(caveat_bits),
            source_mix=source_mix,
        )


# ── Templated narration (LLM fallback — numbers verbatim from the record) ──

def format_advisory_text(rec: AdvisoryRecord) -> str:
    """One/two analyst-readable sentences from a record.

    Every number printed here appears verbatim in the record (the same rule the
    optional LLM narration must obey).
    """
    if rec.suppressed:
        return (f"Advisory suppressed: only {rec.neighbours} similar verified "
                f"runs within distance range {rec.feature_distance_range} "
                f"({rec.suppression_reason}).")
    top = rec.top_modulation or "unknown"
    counts = ", ".join(f"{m} {c}" for m, c in rec.verified_outcomes.items())
    chain = ""
    if rec.chains_that_validated:
        best = rec.chains_that_validated[0]
        chain = (f" Chain that most often validated: {best['chain']} "
                 f"({best['count']} runs, {best['crc_pass_rate']:.0%} CRC pass).")
    caveat = f" Caveat: {rec.caveat}." if rec.caveat else ""
    return (f"On {rec.neighbours} similar verified signals (feature distance "
            f"{rec.feature_distance_range[0]:.2f}–{rec.feature_distance_range[1]:.2f}), "
            f"the distribution was {counts}; most likely {top}.{chain}{caveat}")


def merge_consult_many(records: list[AdvisoryRecord]) -> AdvisoryRecord:
    """Aggregate per-gate advisories into a single run-level record."""
    active = [r for r in records if not r.suppressed]
    if not active:
        return records[0] if records else AdvisoryRecord(
            neighbours=0, suppressed=True, suppression_reason="no advisory produced")
    total = sum(r.neighbours for r in active)
    outcomes: dict[str, float] = {}
    for r in active:
        for m, c in r.verified_outcomes.items():
            outcomes[m] = outcomes.get(m, 0.0) + c
    chains: dict[str, dict] = {}
    for r in active:
        for c in r.chains_that_validated:
            agg = chains.setdefault(c["chain"], {"count": 0, "crc": 0})
            agg["count"] += c["count"]
            agg["crc"] += c["count"] * c["crc_pass_rate"]
    return AdvisoryRecord(
        neighbours=total,
        feature_distance_range=[
            min(r.feature_distance_range[0] for r in active),
            max(r.feature_distance_range[1] for r in active),
        ],
        verified_outcomes={m: int(round(c)) for m, c in
                           sorted(outcomes.items(), key=lambda kv: -kv[1])},
        chains_that_validated=[
            {"chain": name, "count": info["count"],
             "crc_pass_rate": round(info["crc"] / info["count"], 3)}
            for name, info in sorted(chains.items(), key=lambda kv: -kv[1]["count"])
        ],
        caveat=active[0].caveat,
        source_mix={},
        suppressed=False,
    )