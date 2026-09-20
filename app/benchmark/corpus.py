"""
app/benchmark/corpus.py
-----------------------
Verified benchmark corpus for the Benchmark Advisor.

A `CorpusEntry` is one *verified* run: a feature vector (the SAME query key the
advisor will look up) plus the ground-truth modulation / decode chain that a
human or a synthetic generator proved valid. Entries are retrieved but never
trained on — k-NN lookup only.

This is the "no model, no weights" source of truth (§3 / §6 of the HITL design):
every entry is a verified case with a `truth_source` the advisor surfaces so a
human can discount a corpus full of their own past confirmations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


# ── Query feature protocol ──────────────────────────────────────────────────
# The corpus is queried with the characterization feature vector the main
# pipeline already computes (detection + 22-feature extraction). Only
# *normalized / scale-invariant* quantities enter the distance metric so the
# advisor is receiver-agnostic and needs NO fitted scaler (zero training).
#
# Columns are fixed by QUERY_FEATURE_NAMES — query and corpus MUST be built with
# the same function so distances are comparable.

QUERY_FEATURE_NAMES: list[str] = [
    "snr_db_std",            # snr_db / 50.0
    "bw_frac",               # bandwidth_hz / sample_rate
    "sym_rate_frac",         # symbol_rate / sample_rate
    "phase_std_norm",        # phase_std / pi
    "kurtosis_phase",
    "amplitude_variance_norm",
    "kurtosis_amplitude",
    "papr_db",
    "inst_freq_mean_norm",
    "inst_freq_std_norm",
    "c21_sym",
    "c42_norm",
    "spectral_flatness",
    "zero_crossing_rate",
]

# Features that are unbounded coding-theory quantities get soft-clamped so a
# single outlier value cannot dominate the Euclidean distance.
_CLAMP = {
    "kurtosis_phase": (-10.0, 10.0),
    "kurtosis_amplitude": (-3.0, 10.0),
    "papr_db": (0.0, 25.0),
    "c21_sym": (0.0, 2.0),
    "c42_norm": (0.0, 5.0),
    "inst_freq_mean_norm": (0.0, 0.5),
    "inst_freq_std_norm": (0.0, 0.25),
    "snr_db_std": (0.0, 1.2),
    "bw_frac": (0.0, 1.0),
    "sym_rate_frac": (0.0, 1.0),
}


def build_query_vector(
    snr_db: Optional[float],
    bandwidth_hz: Optional[float],
    symbol_rate_hz: Optional[float],
    sample_rate: Optional[float],
    fv: object | None = None,
    feature_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the canonical query vector for a signal.

    Args:
        snr_db / bandwidth_hz / symbol_rate_hz: physical estimates.
        sample_rate: signal sample rate (needed to normalize BW / symbol rate).
        fv: a `FeatureVector` instance. Takes precedence over ``feature_array``.
        feature_array: flat float array consistent with
            ``FeatureVector.to_array()`` order (used when the dataclass is not
            available). ``FeatureVector.feature_names()`` must match the order.

    Returns:
        float32 array in ``QUERY_FEATURE_NAMES`` ordering.
    """
    sr = float(sample_rate or 1.0)
    snr = float(snr_db) if snr_db is not None and np.isfinite(snr_db) else 0.0
    bw = float(bandwidth_hz) if bandwidth_hz is not None and bandwidth_hz > 0 else 0.0
    symr = float(symbol_rate_hz) if symbol_rate_hz is not None and symbol_rate_hz > 0 else 0.0

    if fv is not None:
        feats = {
            "phase_std": fv.phase_std,
            "kurtosis_phase": fv.kurtosis_phase,
            "amplitude_variance_norm": fv.amplitude_variance_norm,
            "kurtosis_amplitude": fv.kurtosis_amplitude,
            "papr_db": fv.papr_db,
            "inst_freq_mean_norm": fv.instantaneous_freq_mean_norm,
            "inst_freq_std_norm": fv.instantaneous_freq_std_norm,
            "c21_sym": fv.C21_sym,
            "c42_norm": fv.C42_norm,
            "spectral_flatness": fv.spectral_flatness,
            "zero_crossing_rate": fv.zero_crossing_rate,
        }
    elif feature_array is not None:
        from app.characterization.features import FeatureVector
        names = FeatureVector.feature_names()
        arr = np.asarray(feature_array, dtype=np.float64).ravel()
        if arr.size != len(names):
            raise ValueError(f"feature_array has {arr.size} values, expected {len(names)}")
        values = dict(zip(names, arr))
        _f = lambda name: float(values[name])
        feats = {
            "phase_std": float(np.sqrt(max(_f("phase_variance"), 0.0))),
            "kurtosis_phase": _f("kurtosis_phase"),
            "amplitude_variance_norm": _f("amplitude_variance_norm"),
            "kurtosis_amplitude": _f("kurtosis_amplitude"),
            "papr_db": _f("papr_db"),
            "inst_freq_mean_norm": _f("instantaneous_freq_mean_norm"),
            "inst_freq_std_norm": _f("instantaneous_freq_std_norm"),
            "c21_sym": _f("c21_sym"),
            "c42_norm": _f("c42_norm"),
            "spectral_flatness": _f("spectral_flatness"),
            "zero_crossing_rate": _f("zero_crossing_rate"),
        }
    else:
        feats = {k: 0.0 for k in
                 ("phase_std", "kurtosis_phase", "amplitude_variance_norm",
                  "kurtosis_amplitude", "papr_db", "inst_freq_mean_norm",
                  "inst_freq_std_norm", "c21_sym", "c42_norm",
                  "spectral_flatness", "zero_crossing_rate")}

    raw = {
        "snr_db_std": snr / 50.0,
        "bw_frac": bw / max(sr, 1e-12),
        "sym_rate_frac": symr / max(sr, 1e-12),
        "phase_std_norm": float(feats.get("phase_std", 0.0)) / np.pi,
        **feats,
    }
    vec = np.empty(len(QUERY_FEATURE_NAMES), dtype=np.float32)
    for i, name in enumerate(QUERY_FEATURE_NAMES):
        v = float(raw.get(name, 0.0))
        lo, hi = _CLAMP.get(name, (-np.inf, np.inf))
        v = float(np.clip(v, lo, hi))
        if not np.isfinite(v):
            v = 0.0
        vec[i] = v
    return vec


# ── Corpus record ───────────────────────────────────────────────────────────

TRUTH_SOURCES = ("synthetic", "known_standard", "human_confirmed")

# Weight of each truth source when the advisor displays (or prefers) entries.
# Human-confirmed entries are *shown* but discounted relative to verified
# synthetic / known-standard ground truth.
TRUTH_SOURCE_WEIGHT = {"synthetic": 1.0, "known_standard": 1.0, "human_confirmed": 0.35}


class CorpusEntry(BaseModel):
    """One verified run in the benchmark corpus."""

    feature_vector: list[float] = Field(..., description=QUERY_FEATURE_NAMES)
    verified_modulation: str
    verified_chain: Optional[str] = None
    crc_validated: bool = False
    snr_db: float = 0.0
    truth_source: str = "synthetic"
    run_id: str = ""


# ── Corpus store ────────────────────────────────────────────────────────────

class BenchmarkCorpus:
    """JSON-persisted collection of verified corpus entries with k-NN support.

    The tree is rebuilt lazily on first query and invalidated on add/load so a
    confirmed run is immediately usable for the next similar signal (§5 feedback
    loop).
    """

    def __init__(self, path: Optional[str | Path] = None):
        self.path: Optional[Path] = Path(path) if path else None
        self.entries: list[CorpusEntry] = []
        self._tree = None
        self._vectors: Optional[np.ndarray] = None
        if self.path is not None and self.path.exists():
            self.load(self.path)

    # -- persistence ---------------------------------------------------------
    def load(self, path: str | Path) -> "BenchmarkCorpus":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.entries = [CorpusEntry(**e) for e in data.get("entries", [])]
        self.path = Path(path)
        self._invalidate()
        return self

    def save(self, path: Optional[str | Path] = None) -> Path:
        dest = Path(path) if path else (self.path or Path("corpus.json"))
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "feature_names": QUERY_FEATURE_NAMES,
            "entry_count": len(self.entries),
            "entries": [e.model_dump() for e in self.entries],
        }
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.path = dest
        return dest

    # -- mutation ------------------------------------------------------------
    def add(self, entry: CorpusEntry) -> None:
        self.entries.append(entry)
        self._invalidate()

    def add_from_run(self, feature_vector, verified_modulation: str,
                     verified_chain: Optional[str], crc_validated: bool,
                     snr_db: float, truth_source: str, run_id: str) -> CorpusEntry:
        entry = CorpusEntry(
            feature_vector=[float(x) for x in np.asarray(feature_vector).ravel()],
            verified_modulation=verified_modulation,
            verified_chain=verified_chain,
            crc_validated=bool(crc_validated),
            snr_db=float(snr_db),
            truth_source=truth_source if truth_source in TRUTH_SOURCES else "human_confirmed",
            run_id=run_id,
        )
        self.add(entry)
        return entry

    # -- k-NN ----------------------------------------------------------------
    def _invalidate(self) -> None:
        self._tree = None
        self._vectors = None

    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = np.array(
                [e.feature_vector for e in self.entries], dtype=np.float32
            ).reshape(len(self.entries), -1)
        return self._vectors

    def tree(self):
        from scipy.spatial import cKDTree
        if self._tree is None or self._tree.n != len(self.entries):
            self._tree = cKDTree(self.vectors())
        return self._tree

    def query(self, query_vector, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) of the k nearest entries."""
        if not self.entries:
            return np.empty(0), np.empty(0, dtype=np.int64)
        k = max(1, min(int(k), len(self.entries)))
        return self.tree().query(np.asarray(query_vector, dtype=np.float32), k=k)

    @property
    def n(self) -> int:
        return len(self.entries)

    def source_mix(self) -> dict[str, int]:
        mix = {}
        for e in self.entries:
            mix[e.truth_source] = mix.get(e.truth_source, 0) + 1
        return mix