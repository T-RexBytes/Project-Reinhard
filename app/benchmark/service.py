"""
app/benchmark/service.py
------------------------
HITL glue between the FastAPI layer and the benchmark package.

Owns the two stateful things the advisor needs across requests:
  1. The verified ``BenchmarkCorpus`` (lazy-loaded from ``RF_BENCHMARK_CORPUS``
     or ``corpus.json``) — grows on every human confirm/override (§6 feedback
     loop; ``human_confirmed`` entries weighted 0.35 at read time).
  2. A per-run audit store answering "was that the machine or the human?"
     (§7 run audit trail).

The advisory NEVER enters the evidence graph and NEVER touches a composite
score — it is returned read-only from ``consult_for_signal``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from app.benchmark.corpus import (
    QUERY_FEATURE_NAMES,
    BenchmarkCorpus,
    TRUTH_SOURCES,
)
from app.benchmark.advisor import (
    AdvisoryRecord,
    BenchmarkAdvisor,
)
from app.benchmark.generator import (
    corpus_query_vector,
    estimate_physical_features,
)

CORPUS_ENV = "RF_BENCHMARK_CORPUS"
DEFAULT_CORPUS = "corpus.json"

AUTO_STAGES = ["ingestion", "detection", "characterization", "classification",
               "demodulation", "decoding"]


# ── Corpus singleton ─────────────────────────────────────────────────────────

_corpus_lock = threading.Lock()
_corpus: Optional[BenchmarkCorpus] = None
_corpus_path: Optional[Path] = None


def corpus_path() -> Path:
    return Path(os.environ.get(CORPUS_ENV, DEFAULT_CORPUS))


def get_corpus(reload: bool = False) -> BenchmarkCorpus:
    """Process-wide verified corpus, lazily loaded and never retrained."""
    global _corpus, _corpus_path
    with _corpus_lock:
        path = corpus_path()
        if _corpus is None or reload or _corpus_path != path:
            _corpus = BenchmarkCorpus(path) if path.exists() else BenchmarkCorpus(path=path)
            _corpus_path = path
        return _corpus


# ── Consult ──────────────────────────────────────────────────────────────────

def consult_for_signal(signal, run_id: str = "",
                       k: int = 12) -> dict:
    """Compute the query vector and advisory for an uploaded signal.

    Uses the SAME characterization estimates as the live pipeline +
    generator (`estimate_physical_features`), then consults the
    `BenchmarkAdvisor`. Returns everything the frontend needs to render
    §3.3/§11.2/§11.4 without a second round trip.
    """
    sr = signal.metadata.sample_rate or 1.0
    feats = estimate_physical_features(signal.data[:65536], sample_rate=sr)
    qvec = corpus_query_vector(feats)
    advisory = BenchmarkAdvisor(get_corpus(), k=k).consult(qvec, gate="run")
    return {
        "run_id": run_id,
        "query_feature_names": QUERY_FEATURE_NAMES,
        "query_vector": [float(x) for x in qvec],
        "physical_estimates": {
            "snr_db": feats.get("snr_db"),
            "bandwidth_hz": feats.get("bandwidth_hz"),
            "symbol_rate_hz": feats.get("symbol_rate_hz"),
        },
        "advisory": advisory.to_dict(),
        "corpus": {
            "size": get_corpus().n,
            "source_mix": get_corpus().source_mix(),
            "schema_version": 1,
        },
    }


# ── Run audit store (§7) ─────────────────────────────────────────────────────

_runs_lock = threading.Lock()
_runs: dict[str, dict] = {}
_RUNS_FILE_ENV = "RF_HITL_RUNS_FILE"


def _runs_file() -> Optional[Path]:
    v = os.environ.get(_RUNS_FILE_ENV)
    return Path(v) if v else None


def _runs_dump() -> None:
    f = _runs_file()
    if f is None:
        return
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(_runs, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def _run_locked(run_id: str) -> dict:
    if not run_id:
        raise ValueError("run_id required for gate actions")
    run = _runs.setdefault(run_id, {
        "run_id": run_id,
        "gates_fired": [],
        "advisor_consulted": False,
        "advisor_neighbours": 0,
        "human_overrides": [],
        "stages_automatic": list(AUTO_STAGES),
        "llm_used_for": [],
        "corpus_added": [],
    })
    return run


def record_consult(run_id: str, advisory: AdvisoryRecord) -> dict:
    """Mark a run's advisory consult in its audit record."""
    with _runs_lock:
        run = _run_locked(run_id)
        run["advisor_consulted"] = True
        run["advisor_neighbours"] = advisory.neighbours
        _runs_dump()
        return run


def apply_gate(run_id: str, gate: str, action: str, to_value: str,
               feature_vector: list[float], snr_db: Optional[float] = None,
               chain: Optional[str] = None,
               from_value: Optional[str] = None,
               crc_validated: bool = False) -> dict:
    """Record a human gate decision and (on confirm/override) promote it to the
    verified corpus (§6 feedback loop). Returns {audit, corpus, overrides}.
    """
    if gate not in ("G1", "G2", "G3"):
        raise ValueError("gate must be one of G1, G2, G3")
    if action not in ("confirm", "override"):
        raise ValueError("action must be 'confirm' or 'override'")
    if not to_value:
        raise ValueError("to_value required")

    qv = np.asarray(list(feature_vector or []), dtype=np.float32)
    if qv.shape != (len(QUERY_FEATURE_NAMES),):
        raise ValueError(
            f"feature_vector must have {len(QUERY_FEATURE_NAMES)} values, "
            f"got {list(qv.shape)}")

    with _runs_lock:
        run = _run_locked(run_id)
        if gate not in run["gates_fired"]:
            run["gates_fired"].append(gate)
        run["human_overrides"].append({
            "gate": gate,
            "action": action,
            "from": from_value,
            "to": to_value,
            "chain": chain,
            "timestamp": time.time(),
        })
        if action in ("confirm", "override"):
            c = get_corpus()
            entry = c.add_from_run(
                feature_vector=qv,
                verified_modulation=to_value,
                verified_chain=chain,
                crc_validated=bool(crc_validated),
                snr_db=float(snr_db) if snr_db is not None else 0.0,
                truth_source="human_confirmed",
                run_id=run_id,
            )
            c.save()
            run["corpus_added"].append({
                "run_id": run_id, "gate": gate, "action": action,
                "entry": entry.model_dump(),
            })
        _runs_dump()

    return {
        "run_id": run_id,
        "audit": dict(run),
        "corpus": {
            "size": get_corpus().n,
            "source_mix": get_corpus().source_mix(),
        },
    }


def get_run(run_id: str) -> Optional[dict]:
    with _runs_lock:
        return dict(_runs.get(run_id, {})) or None


def all_runs() -> dict:
    with _runs_lock:
        return dict(_runs)


# ── Corpus summary (§11.4 CRC / chain pass-rate matrix) ─────────────────────

def corpus_summary() -> dict:
    """Per-chain CRC pass rate and truth-source mix for the whole corpus."""
    c = get_corpus()
    per_chain: dict[str, dict] = {}
    tr = [t for t in TRUTH_SOURCES]
    for e in c.entries:
        k = e.verified_chain or f"{e.verified_modulation}|(no chain)"
        row = per_chain.setdefault(k, {
            "chain": k, "count": 0, "crc_pass": 0, "sources": {t: 0 for t in tr},
        })
        row["count"] += 1
        row["crc_pass"] += 1 if e.crc_validated else 0
        row["sources"][e.truth_source] = row["sources"].get(e.truth_source, 0) + 1
    rows = sorted(per_chain.values(), key=lambda r: -r["crc_pass"] / max(r["count"], 1))
    return {
        "size": c.n,
        "source_mix": c.source_mix(),
        "per_chain": [
            {**r, "crc_pass_rate": round(r["crc_pass"] / max(r["count"], 1), 3)}
            for r in rows
        ],
    }