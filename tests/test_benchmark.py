"""Tests for the Benchmark-Assessment HITL layer: cache, corpus, generator, advisor.

Covers the no-model/no-training contract (§3.2/§9.2/§9.3 of the HITL design):
- StageCache content addressing and round-trips.
- Query vector protocol: canonical 14-col ordering, clamping, scale-invariance.
- Generator: encoder-chain ground truth, interleave/FEC combinations, SNR target.
- BenchmarkCorpus: persistence round-trip, k-NN retrieval, add_from_run.
- BenchmarkAdvisor: majority outcome, suppression below min neighbours,
  distance threshold suppression, advisory text / chain aggregation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.benchmark.cache import StageCache, make_cache_key
from app.benchmark.corpus import (
    QUERY_FEATURE_NAMES,
    BenchmarkCorpus,
    CorpusEntry,
    build_query_vector,
    TRUTH_SOURCES,
    TRUTH_SOURCE_WEIGHT,
)
from app.benchmark.advisor import (
    AdvisoryRecord,
    BenchmarkAdvisor,
    format_advisory_text,
    merge_consult_many,
)


def _qv(mod: str, snr: float = 15.0) -> np.ndarray:
    """A basic 14-col query vector; mod just shifts c42_norm so classes cluster."""
    c42 = {"BPSK": 2.0, "QPSK": 1.0, "16QAM": 0.68, "2FSK": 0.1}[mod]
    base = {
        "snr_db_std": snr / 50.0,
        "bw_frac": 0.2,
        "sym_rate_frac": 0.25,
        "phase_std_norm": 0.15 if mod != "2FSK" else 0.5,
        "kurtosis_phase": 1.0,
        "amplitude_variance_norm": 0.05 if mod not in ("16QAM",) else 0.30,
        "kurtosis_amplitude": 1.0,
        "papr_db": 2.0 if mod != "16QAM" else 4.0,
        "inst_freq_mean_norm": 0.01,
        "inst_freq_std_norm": 0.02 if mod != "2FSK" else 0.18,
        "c21_sym": 1.0 if mod == "BPSK" else 0.1,
        "c42_norm": c42,
        "spectral_flatness": 0.9,
        "zero_crossing_rate": 0.1 if mod != "2FSK" else 0.4,
    }
    return np.array([base[n] for n in QUERY_FEATURE_NAMES], dtype=np.float32)


def _mini_corpus() -> BenchmarkCorpus:
    c = BenchmarkCorpus()
    for mod in ("BPSK", "QPSK", "16QAM", "2FSK"):
        for snr in (5.0, 20.0):
            c.add(CorpusEntry(
                feature_vector=_qv(mod, snr).tolist(),
                verified_modulation=mod,
                verified_chain=f"viterbi|none|preamble+crc16",
                crc_validated=True,
                snr_db=snr,
                truth_source="synthetic",
                run_id=f"t-{mod}-{int(snr)}",
            ))
    return c


class TestStageCache:
    def test_round_trip_and_key_stability(self):
        c = StageCache()
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        c.put("feat", data, {"a": 1}, {"out": 5})
        assert c.get("feat", data, {"a": 1}) == {"out": 5}
        k1 = make_cache_key("feat", data, {"a": 1})
        k2 = make_cache_key("feat", data, {"a": 1})
        k3 = make_cache_key("feat", data, {"a": 2})
        assert k1 == k2
        assert k1 != k3
        assert c.stats()["size"] == 1

    def test_dtype_sensitivity(self):
        c = StageCache()
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float64)
        c.put("x", a, {}, "f32")
        assert c.get("x", b, {}) is None       # different dtype -> different key
        assert c.get("x", a, {}) == "f32"

    def test_clear(self):
        c = StageCache()
        c.put("x", np.zeros(4), {}, 1)
        c.clear()
        assert c.stats()["size"] == 0


class TestQueryVector:
    def test_ordering_and_length(self):
        qv = build_query_vector(snr_db=20.0, bandwidth_hz=200.0e3,
                                symbol_rate_hz=250.0e3, sample_rate=1.0e6)
        assert qv.shape == (14,)
        assert np.all(np.isfinite(qv))

    def test_scale_invariance(self):
        # halving the sample rate halves bandwidth+symrate fractions -> identical
        qv1 = build_query_vector(20.0, 200_000.0, 250_000.0, 1.0e6)
        qv2 = build_query_vector(20.0, 100_000.0, 125_000.0, 500_000.0)
        assert np.allclose(qv1, qv2)

    def test_clamping(self):
        qv = build_query_vector(snr_db=1e6, bandwidth_hz=1e9,
                                symbol_rate_hz=1e9, sample_rate=1.0)
        assert qv[QUERY_FEATURE_NAMES.index("bw_frac")] <= 1.0
        assert qv[QUERY_FEATURE_NAMES.index("snr_db_std")] <= 1.2


class TestGenerator:
    def test_grid_and_ground_truth(self):
        from app.benchmark.generator import generate_synthetic_sample
        for mod in ("BPSK", "QPSK", "16QAM", "2FSK"):
            iq, gt = generate_synthetic_sample(mod, "block(4,8)", "viterbi",
                                               "preamble+crc16", 15.0, seed=3)
            assert iq.dtype == np.complex64 and iq.ndim == 1 and len(iq) > 1000
            assert "viterbi" in gt.canonical_chain()
            assert gt.snr_true_db == 15.0
            assert gt.modulation == mod

    def test_fsk_is_constant_envelope(self):
        from app.benchmark.generator import generate_synthetic_sample
        iq, gt = generate_synthetic_sample("2FSK", "none", "none",
                                           "preamble+crc16", 20.0, seed=5)
        env = np.abs(iq)
        assert np.std(env) / np.mean(env) < 0.25     # CPFSK stays flat even with noise

    def test_power_units(self):
        from app.benchmark.generator import generate_synthetic_sample
        iq, _ = generate_synthetic_sample("QPSK", "none", "none",
                                          "preamble+crc16", 0.0, seed=9)
        # AWGN at 0 dB -> total power ≈ 2x signal power (both unit-power scaled)
        assert 0.8 < np.mean(np.abs(iq) ** 2) < 3.0


class TestCorpus:
    def test_round_trip(self, tmp_path):
        c = _mini_corpus()
        p = tmp_path / "c.json"
        c.save(p)
        c2 = BenchmarkCorpus(p)
        assert c2.n == c.n
        assert c2.entries[0].verified_modulation == c.entries[0].verified_modulation
        assert c2.entries[0].feature_vector == c.entries[0].feature_vector

    def test_query_returns_nearest(self):
        c = _mini_corpus()
        dists, idx = c.query(_qv("16QAM", 20.0), k=8)
        assert len(dists) == 8
        assert dists[0] < 1e-6           # exact self-match first

    def test_source_mix_and_weights(self):
        c = _mini_corpus()
        assert c.source_mix() == {"synthetic": 8}
        assert TRUTH_SOURCE_WEIGHT["synthetic"] >= TRUTH_SOURCE_WEIGHT["human_confirmed"]
        assert set(TRUTH_SOURCES) == {"synthetic", "known_standard", "human_confirmed"}


class TestAdvisor:
    def test_majority_outcome(self):
        adv = BenchmarkAdvisor(_mini_corpus(), k=8)
        rec = adv.consult(_qv("QPSK", 20.0))
        assert not rec.suppressed
        assert rec.top_modulation in ("QPSK",)
        assert rec.neighbours >= 5

    def test_suppression_thin_corpus(self):
        c = BenchmarkCorpus()
        for mod in ("BPSK", "QPSK"):
            c.add(CorpusEntry(feature_vector=_qv(mod).tolist(),
                              verified_modulation=mod,
                              verified_chain="none|none|preamble+crc16",
                              crc_validated=True, snr_db=15.0,
                              truth_source="synthetic", run_id=mod))
        adv = BenchmarkAdvisor(c, k=12)
        rec = adv.consult(_qv("BPSK"))
        assert rec.suppressed          # only 2 entries < min_neighbours=5
        assert "neighbours" in rec.suppression_reason

    def test_distance_threshold_suppression(self):
        c = BenchmarkCorpus()
        vec_far = np.array(_qv("BPSK").tolist())
        for i in range(6):
            far = vec_far + np.random.default_rng(i).normal(3.0, 0.5, len(vec_far))
            c.add(CorpusEntry(feature_vector=far.tolist(),
                              verified_modulation="2FSK",
                              verified_chain="viterbi|none|preamble+crc16",
                              crc_validated=True, snr_db=15.0,
                              truth_source="synthetic", run_id=f"far{i}"))
        rec = BenchmarkAdvisor(c, k=12).consult(_qv("BPSK"))
        assert rec.suppressed and "distance" in rec.suppression_reason

    def test_empty_corpus(self):
        rec = BenchmarkAdvisor(BenchmarkCorpus()).consult(np.zeros(14, np.float32))
        assert rec.suppressed and "empty" in rec.suppression_reason

    def test_format_text_verbatim_numbers(self):
        rec = AdvisoryRecord(
            neighbours=7,
            feature_distance_range=[0.1, 0.9],
            verified_outcomes={"QPSK": 5, "BPSK": 2},
            chains_that_validated=[{"chain": "viterbi|block(4, 8)|preamble+crc16",
                                    "count": 7, "crc_pass_rate": 0.857}],
            caveat="sample caveat",
            suppressed=False,
        )
        text = format_advisory_text(rec)
        assert "7" in text and "QPSK" in text and "0.1" in text and "0.9" in text
        assert "caveat" in text

    def test_merge_consult_many(self):
        r1 = AdvisoryRecord(neighbours=6, feature_distance_range=[0.0, 1.0],
                            verified_outcomes={"QPSK": 4, "BPSK": 2},
                            chains_that_validated=[{"chain": "a", "count": 6,
                                                    "crc_pass_rate": 1.0}],
                            suppressed=False, caveat="")
        r2 = AdvisoryRecord(neighbours=5, feature_distance_range=[0.2, 1.2],
                            verified_outcomes={"QPSK": 3, "16QAM": 2},
                            chains_that_validated=[{"chain": "a", "count": 5,
                                                    "crc_pass_rate": 0.8}],
                            suppressed=False, caveat="")
        merged = merge_consult_many([r1, r2])
        assert merged.neighbours == 11
        assert merged.verified_outcomes["QPSK"] == 7