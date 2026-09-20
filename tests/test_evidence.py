"""Tests for the Evidence Graph: models, per-stage aggregation, cross-stage
linkage (downstream feedback), scoring, and JSON / GraphML serialization."""

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.evidence.models import (
    EvidenceGraph,
    EvidenceItem,
    HypothesisNode,
    SupportEdge,
    clamp01,
    make_evidence_id,
    make_evidence_id_unique,
    quality_from_ratio,
)
from app.evidence.aggregate import (
    MODULATION_FAMILIES,
    build_evidence_graph,
    collect_characterization,
    collect_decoding,
    collect_demodulation,
    collect_detection,
    collect_modulation,
    link_features_to_families,
)
from app.evidence.serializer import (
    export_evidence_graph,
    graph_from_json,
    graph_to_graphml,
    graph_to_json,
    load_json,
    save_graphml,
    save_json,
)
from app.modulation.classifier import (
    ModulationCandidate,
    ModulationClassificationResult,
)
from app.decoding.decoder import DecodingResult


def _obj(**kw):
    return SimpleNamespace(**kw)


def _evm(rms=3.5, db=-30.0):
    return _obj(evm_rms_percent=rms, evm_db=db)


def _mod_result(top="QPSK", conf=0.85, scores=None, features_summary=None):
    scores = scores or {"QPSK": 0.62, "BPSK": 0.15, "16QAM": 0.12,
                        "2-FSK": 0.05, "4-FSK": 0.03, "Noise/CW": 0.03}
    cands = [
        ModulationCandidate(mod, score, conf * (0.9 if i == 0 else 0.4),
                            {"fit_c42": 1.0} if mod in ("BPSK", "QPSK", "16QAM") else {},
                            f"why {mod}")
        for i, (mod, score) in enumerate(scores.items())
    ]
    return ModulationClassificationResult(
        top_candidate=top, confidence=conf, candidates=cands,
        snr_db=15.0, sample_rate=2e6,
        features_summary=features_summary or {"C42_norm": 0.9},
    )


def _decode_result(spec, crc=True, ok=None, verr=0, nrs=2, corr=0.91,
                   fec="viterbi+rs255_223"):
    ok = crc if ok is None else ok
    return DecodingResult(
        spec_name=spec,
        chain=f"deint -> {fec} -> frame",
        ok=ok, crc_valid=crc,
        num_viterbi_errors=verr, num_rs_symbols_corrected=nrs,
        preamble_correlation=corr, frame_offset=16 if ok else None,
        used_soft=True, fec=fec, why="test chain",
    )


class TestQualityHelpers:
    def test_clamp01(self):
        assert clamp01(-1.0) == 0.0
        assert clamp01(2.0) == 1.0
        assert clamp01(0.5) == 0.5

    def test_quality_tiers(self):
        assert quality_from_ratio(0.9) == "good"
        assert quality_from_ratio(0.55) == "fair"
        assert quality_from_ratio(0.2) == "poor"

    def test_make_evidence_id(self):
        assert make_evidence_id("QPSK", "classifier_score") == "QPSK:classifier_score"

    def test_make_evidence_id_unique(self):
        a = make_evidence_id_unique("spec", "crc_valid")
        b = make_evidence_id_unique("spec", "crc_valid")
        assert a != b and a.startswith("spec:crc_valid:")


class TestEvidenceItem:
    def test_contribution_supports(self):
        item = EvidenceItem(id="x", stage="modulation", candidate="QPSK",
                            measurement="score", value=0.5, value_norm=0.5,
                            supports=True, weight=2.0)
        assert item.contribution() == pytest.approx(1.0)

    def test_contribution_contradicts(self):
        item = EvidenceItem(id="y", stage="demodulation", candidate="QPSK",
                            measurement="gated", value=1.0, value_norm=1.0,
                            supports=False, weight=0.7)
        assert item.contribution() == pytest.approx(-0.7)

    def test_to_dict_round_trip_fields(self):
        item = EvidenceItem(id="x", stage="modulation", candidate="QPSK",
                            measurement="classifier_score", value=0.62, source="c")
        d = item.to_dict()
        assert d["id"] == "x"
        assert d["supports"] is True
        assert d["provenance"] == "signal_inferred"


class TestEvidenceGraphModel:
    def test_add_item_and_hypothesis(self):
        g = EvidenceGraph()
        g.add_item(EvidenceItem(id="i1", stage="modulation", candidate="QPSK", measurement="s"))
        g.add_hypothesis("QPSK", kind="modulation")
        g.add_edge("i1", "QPSK")
        assert "i1" in g.items
        assert g.hypotheses["QPSK"].num_items == 1

    def test_add_edge_unknown_item_raises(self):
        g = EvidenceGraph()
        with pytest.raises(KeyError):
            g.add_edge("missing", "QPSK")

    def test_add_edge_autocreates_hypothesis(self):
        g = EvidenceGraph()
        g.add_item(EvidenceItem(id="i1", stage="modulation", candidate="QPSK", measurement="s"))
        g.add_edge("i1", "QPSK")
        assert g.hypotheses["QPSK"].kind == "modulation"

    def test_dependency_lineage(self):
        g = EvidenceGraph()
        g.add_hypothesis("QPSK", kind="modulation")
        g.add_hypothesis("spec", kind="decode_chain")
        g.add_dependency("spec", "QPSK")
        assert "QPSK" in g.hypotheses["spec"].parents

    def test_neutral_score_when_no_evidence(self):
        g = EvidenceGraph()
        g.add_hypothesis("QPSK", kind="modulation")
        g.compute_scores()
        assert g.hypotheses["QPSK"].score == pytest.approx(0.5)

    def test_supported_score_above_neutral(self):
        g = EvidenceGraph()
        g.add_item(EvidenceItem(id="i1", stage="modulation", candidate="QPSK",
                                measurement="score", value=0.9, value_norm=0.9, weight=1.4))
        g.add_edge("i1", "QPSK")
        g.compute_scores()
        assert g.hypotheses["QPSK"].score > 0.5

    def test_contradiction_lowers_score(self):
        g = EvidenceGraph()
        g.add_item(EvidenceItem(id="i1", stage="demodulation", candidate="QPSK",
                                measurement="gated", value_norm=1.0, supports=False,
                                weight=0.7))
        g.add_edge("i1", "QPSK")
        g.compute_scores()
        assert g.hypotheses["QPSK"].score < 0.5

    def test_trace_edges_do_not_affect_score(self):
        g = EvidenceGraph()
        g.add_hypothesis("Noise/CW", kind="modulation")
        g.add_item(EvidenceItem(id="f1", stage="characterization", candidate="signal",
                                measurement="spectral_flatness", value_norm=1.0))
        g.add_edge("f1", "Noise/CW", kind="trace")
        g.compute_scores()
        assert g.hypotheses["Noise/CW"].score == pytest.approx(0.5)

    def test_top_hypotheses_ranking(self):
        g = EvidenceGraph()
        for name in ("A", "B"):
            g.add_item(EvidenceItem(id=f"i{name}", stage="modulation", candidate=name,
                                    measurement="score",
                                    value=0.9 if name == "A" else 0.1,
                                    value_norm=0.9 if name == "A" else 0.1, weight=1.4))
            g.add_edge(f"i{name}", name)
        g.compute_scores()
        top = g.top_hypotheses(k=2)
        assert top[0].name == "A"
        assert top[0].score > top[1].score


class TestAggregation:
    def test_collect_detection_adds_signal_items(self):
        g = EvidenceGraph()
        collect_detection(g, snr=_obj(snr_db=15.2, method="psd_ratio"),
                          bandwidth=_obj(bandwidth_hz=1.2e6),
                          noise_floor=_obj(noise_floor_db=-80.0, method="mad"),
                          segmentation=_obj(num_segments=1, detection_threshold_db=-70))
        names = {i.measurement for i in g.items.values()}
        assert {"snr_db", "bandwidth_hz", "noise_floor_db", "num_segments"} <= names
        assert all(i.candidate == "signal" for i in g.items.values())

    def test_collect_characterization_adds_features(self):
        features = _obj(C42_norm=0.9, C21_sym=0.05, papr_db=4.5,
                        spectral_flatness=0.35, amplitude_variance_norm=0.2,
                        kurtosis_amplitude=-0.5)
        g = EvidenceGraph()
        collect_characterization(g, features=features,
                                 symbol_rate=_obj(symbol_rate_hz=1e6, confidence=0.8, method="env"),
                                 freq_offset=_obj(offset_hz=12000.0, confidence=0.6, method="psd"))
        names = {i.measurement for i in g.items.values()}
        assert "c42_norm" in names and "symbol_rate_hz" in names
        assert "freq_offset_hz" in names

    def test_collect_modulation_creates_family_nodes(self):
        g = EvidenceGraph()
        collect_modulation(g, _mod_result())
        assert "QPSK" in g.hypotheses
        qpsk_items = [i for i in g.items.values()
                      if i.candidate == "QPSK" and i.stage == "modulation"]
        measurements = {i.measurement for i in qpsk_items}
        assert "classifier_score" in measurements
        assert "classifier_confidence" in measurements

    def test_collect_modulation_wires_edges(self):
        g = EvidenceGraph()
        collect_modulation(g, _mod_result())
        g.compute_scores()
        assert g.hypotheses["QPSK"].score > g.hypotheses["2-FSK"].score

    def test_collect_demodulation_clean_raises_family(self):
        base = EvidenceGraph()
        collect_modulation(base, _mod_result())
        base.compute_scores()
        score_bare = base.hypotheses["QPSK"].score

        g = EvidenceGraph()
        collect_modulation(g, _mod_result())
        collect_demodulation(g, _obj(modulation="QPSK", gated=False,
                                     evm=_evm(rms=3.5, db=-30.0), num_symbols=512,
                                     residual_phase_rad=0.02), family="QPSK")
        g.compute_scores()
        assert g.hypotheses["QPSK"].score > score_bare

    def test_collect_demodulation_gated_lowers_family(self):
        base = EvidenceGraph()
        collect_modulation(base, _mod_result())
        base.compute_scores()
        score_bare = base.hypotheses["QPSK"].score

        g = EvidenceGraph()
        collect_modulation(g, _mod_result())
        collect_demodulation(g, _obj(modulation="QPSK", gated=True, evm=None,
                                     num_symbols=4, why="too few symbols"), family="QPSK")
        g.compute_scores()
        assert g.hypotheses["QPSK"].score < score_bare

    def test_collect_decoding_wires_chain_and_parents(self):
        g = EvidenceGraph()
        collect_decoding(g, [_decode_result("viterbi|none|preamble+crc16", crc=True)],
                         family="QPSK")
        assert "viterbi|none|preamble+crc16" in g.hypotheses
        chain = g.hypotheses["viterbi|none|preamble+crc16"]
        assert chain.parents == ["QPSK"]
        crc_items = [i for i in g.items.values() if i.measurement == "crc_valid"]
        assert len(crc_items) == 1 and crc_items[0].family == "QPSK"

    def test_collect_decoding_feedback_to_family(self):
        g = EvidenceGraph()
        g.add_hypothesis("QPSK", kind="modulation")
        collect_decoding(g, [_decode_result("A", crc=True),
                             _decode_result("B", crc=False, verr=12)],
                         family="QPSK")
        feedback = [i for i in g.items.values()
                    if i.stage == "decoding" and i.candidate == "QPSK"
                    and i.measurement == "chain_crc_valid"]
        assert len(feedback) == 1 and feedback[0].supports is True

    def test_link_features_traceability(self):
        g = EvidenceGraph()
        features = _obj(C42_norm=0.9, C21_sym=0.05, papr_db=4.5,
                        spectral_flatness=0.35, amplitude_variance_norm=0.2,
                        kurtosis_amplitude=-0.5)
        collect_characterization(g, features=features)
        link_features_to_families(g)
        trace_edges = [e for e in g.edges if e.kind == "trace"]
        assert len(trace_edges) > 0
        # trace edges should not feed scores
        g.compute_scores()
        assert g.hypotheses["BPSK"].score == pytest.approx(0.5)


class TestDownstreamFeedbackProperty:
    """The project's critical property: downstream success must raise the
    upstream hypothesis's confidence."""

    def _base_graph(self):
        g = EvidenceGraph()
        collect_modulation(g, _mod_result())
        return g

    def _modulation_only_score(self):
        g = EvidenceGraph()
        collect_modulation(g, _mod_result())
        g.compute_scores()
        return g.hypotheses["QPSK"].score

    def test_modulation_only(self):
        assert self._modulation_only_score() > 0.5

    def test_add_clean_demodulation_raises_family(self):
        base = self._modulation_only_score()
        g = EvidenceGraph()
        collect_demodulation(g, _obj(modulation="QPSK", gated=False,
                                     evm=_evm(), num_symbols=1024,
                                     residual_phase_rad=0.01), family="QPSK")
        g.compute_scores()
        assert g.hypotheses["QPSK"].score > base

    def test_add_crc_valid_chain_raises_family_further(self):
        g = self._base_graph()
        collect_demodulation(g, _obj(modulation="QPSK", gated=False,
                                     evm=_evm(), num_symbols=1024,
                                     residual_phase_rad=0.01), family="QPSK")
        collect_decoding(g, [_decode_result("viterbi+rs255_223|block(4,8)|preamble+crc32",
                                            crc=True)], family="QPSK")
        g.compute_scores()
        with_demod = EvidenceGraph()
        collect_modulation(with_demod, _mod_result())
        collect_demodulation(with_demod, _obj(modulation="QPSK", gated=False,
                                              evm=_evm(), num_symbols=1024,
                                              residual_phase_rad=0.01), family="QPSK")
        with_demod.compute_scores()
        assert g.hypotheses["QPSK"].score > with_demod.hypotheses["QPSK"].score

    def test_crc_chain_outranks_crc_fail_chain(self):
        g = EvidenceGraph()
        collect_decoding(g, [_decode_result("good", crc=True),
                             _decode_result("bad", crc=False, ok=False, verr=20, corr=0.1)],
                         family="QPSK")
        g.compute_scores()
        good = g.hypotheses["good"].score
        bad = g.hypotheses["bad"].score
        assert good > bad


class TestBuildGraphEndToEnd:
    def test_full_build_ranks_qpsk_first(self):
        g = build_evidence_graph(
            snr=_obj(snr_db=15.2, method="psd_ratio"),
            bandwidth=_obj(bandwidth_hz=1.2e6),
            noise_floor=_obj(noise_floor_db=-80.0, method="mad"),
            segmentation=_obj(num_segments=1, detection_threshold_db=-70),
            features=_obj(C42_norm=0.9, C21_sym=0.05, papr_db=4.5,
                          spectral_flatness=0.35, amplitude_variance_norm=0.2,
                          kurtosis_amplitude=-0.5),
            symbol_rate=_obj(symbol_rate_hz=1e6, confidence=0.85, method="env"),
            freq_offset=_obj(offset_hz=12000.0, confidence=0.6, method="psd"),
            mod_classification=_mod_result(),
            demodulation=_obj(modulation="QPSK", gated=False, evm=_evm(),
                              num_symbols=512, residual_phase_rad=0.02),
            decoding=[_decode_result("viterbi+rs255_223|block(4,8)|preamble+crc32", crc=True),
                      _decode_result("viterbi|none|preamble+crc16", crc=False, verr=10)],
            metadata={"filename": "test.iq"},
        )
        mods = [h.name for h in g.top_hypotheses(k=6) if h.kind == "modulation"]
        assert mods[0] == "QPSK"
        assert g.hypotheses["QPSK"].score > 0.5
        # decode chain depends on family
        family_of_chain = g.hypotheses["viterbi|none|preamble+crc16"].parents
        assert family_of_chain

    def test_build_without_any_stages_is_neutral(self):
        g = build_evidence_graph(metadata={})
        for fam in MODULATION_FAMILIES:
            assert g.hypotheses[fam].score == pytest.approx(0.5)


class TestSerialization:
    def _graph(self):
        return build_evidence_graph(
            snr=_obj(snr_db=15.2, method="psd_ratio"),
            mod_classification=_mod_result(),
            demodulation=_obj(modulation="QPSK", gated=False, evm=_evm(),
                              num_symbols=512, residual_phase_rad=0.02),
            decoding=[_decode_result("viterbi|none|preamble+crc16", crc=True)],
            metadata={"filename": "test.iq"},
        )

    def test_json_round_trip(self):
        g = self._graph()
        g2 = graph_from_json(graph_to_json(g))
        assert set(g2.items) == set(g.items)
        assert len(g2.edges) == len(g.edges)
        assert set(g2.hypotheses) == set(g.hypotheses)
        assert g2.hypotheses["QPSK"].score == pytest.approx(g.hypotheses["QPSK"].score, abs=5e-3)
        assert g2.metadata["filename"] == "test.iq"

    def test_graphml_is_valid_xml(self):
        g = self._graph()
        text = graph_to_graphml(g)
        root = ET.fromstring(text)
        nodes = root.findall(".//{http://graphml.graphdrawing.org/xmlns}node")
        assert len(nodes) >= len(g.hypotheses) + len(g.items)

    def test_save_and_load_files(self):
        g = self._graph()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ev.json"
            save_json(g, p)
            assert p.exists()
            g2 = load_json(p)
            assert len(g2.items) == len(g.items)

    def test_export_writes_both_formats(self):
        g = self._graph()
        with tempfile.TemporaryDirectory() as d:
            out = export_evidence_graph(g, d, basename="ev")
            assert out["json"].exists()
            assert out["graphml"].exists()

    def test_graphml_contains_dependency_edges(self):
        g = self._graph()
        text = graph_to_graphml(g)
        assert "dependency" in text
        assert "hyp:viterbi|none|preamble+crc16" in text


class TestPublicApi:
    def test_init_exports(self):
        import app.evidence as ev
        for name in ("EvidenceGraph", "EvidenceItem", "HypothesisNode",
                     "SupportEdge", "build_evidence_graph", "graph_to_json",
                     "graph_to_graphml", "export_evidence_graph"):
            assert hasattr(ev, name)