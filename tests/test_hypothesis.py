"""Tests for the Hypothesis Engine (app/hypothesis/)."""

import numpy as np
import pytest

from app.hypothesis import (
    HypothesisChain,
    HypothesisEngineConfig,
    ChainScoringWeights,
    DemodParams,
    DeinterleaverParams,
    FECParams,
    FrameParams,
    ScoreUncertainty,
    make_chain_id,
    make_demod_params_grid,
    make_deinterleaver_params_grid,
    make_fec_params_grid,
    make_frame_params_grid,
    DEFAULT_DEMOD_PARAMS,
    enumerate_chain_candidates,
    evaluate_chain_live,
    run_hypothesis_search,
    hypothesize,
    score_modulation_from_evidence,
    score_demodulation_from_evidence,
    score_decoding_from_evidence,
    score_demodulation_from_result,
    score_decoding_from_results,
    score_snr_margin,
    compute_composite_score,
    quantify_uncertainty,
    rank_chains,
    compare_chains,
    generate_why,
    generate_short_why,
    generate_comparison_why,
    generate_hypothesis_report,
    generate_chain_json,
)
from app.evidence import build_evidence_graph, EvidenceGraph, EvidenceItem, make_evidence_id
from app.models import ComplexSignal, SignalMetadata, DataType, IQOrdering
from app.demodulation.slicer import EVMResult, DemodulationResult
from app.decoding.decoder import DecodingResult


def _make_signal(n: int = 4096, mod: str = "QPSK", sps: int = 4, fs: float = 1e6):
    """Create a synthetic ComplexSignal (BPSK/QPSK/16QAM/FSK)."""
    np.random.seed(42)
    num_symbols = max(1, n // sps)
    if mod == "BPSK":
        bits = np.random.randint(0, 2, num_symbols) * 2 - 1
        syms = bits.astype(np.complex64)
    elif mod == "16QAM":
        grid = np.array([-3, -1, 1, 3])
        i_ch = np.random.choice(grid, num_symbols)
        q_ch = np.random.choice(grid, num_symbols)
        syms = (i_ch + 1j * q_ch).astype(np.complex64) / np.sqrt(10.0)
    elif "FSK" in mod:
        num_tones = 4 if "4" in mod else 2
        tones = np.random.randint(0, num_tones, num_symbols)
        dev = 0.25
        k = tones - (num_tones - 1) / 2
        base = np.zeros(n, dtype=np.complex64)
        for i in range(num_symbols):
            a = i * sps
            b = min(a + sps, n)
            freqs = k[i] * dev * fs
            base[a:b] = np.exp(1j * 2 * np.pi * freqs * np.arange(a, b) / fs)
        syms = base
    else:
        bits = np.random.randint(0, 4, num_symbols)
        syms = np.exp(1j * (bits * np.pi / 2.0 + np.pi / 4.0)).astype(np.complex64)

    if "FSK" in mod:
        iq = syms + (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64) * 0.04
    else:
        oversampled = np.repeat(syms, sps)[:n]
        noise = (np.random.randn(len(oversampled)) + 1j * np.random.randn(len(oversampled))).astype(np.complex64) * 0.04
        iq = oversampled + noise

    meta = SignalMetadata(
        filename="test.cf32",
        file_size_bytes=len(iq) * 4,
        data_type=DataType.CF32,
        sample_rate=fs,
        center_frequency=100.0e6,
        iq_ordering=IQOrdering.INTERLEAVED,
        num_samples=len(iq),
    )
    return ComplexSignal(data=iq.astype(np.complex64), metadata=meta)


def _make_evidence_graph(mod: str = "QPSK", snr_db: float = 20.0):
    """Build a synthetic evidence graph for a QPSK-like signal."""
    from app.evidence import build_evidence_graph, MODULATION_FAMILIES

    class _SNR:
        def __init__(self, value):
            self.snr_db = value
            self.method = "psd"

    snr_obj = _SNR(snr_db)

    class _NF:
        noise_floor_db = -90.0
        method = "mad"

    class _BW:
        bandwidth_hz = 250000.0

    class _Seg:
        num_segments = 3

    demod = DemodulationResult(
        modulation=mod,
        evm=EVMResult(evm_rms_percent=8.0, evm_db=-22.0, evm_peak_percent=18.0),
        num_symbols=1024,
        bits=[0, 1, 0, 1] * 128,
        residual_phase_rad=0.05,
        gated=False,
    )
    dec1 = DecodingResult(
        spec_name=f"viterbi|none|preamble+crc16",
        chain="viterbi -> frame/crc16",
        ok=True, crc_valid=True,
        num_viterbi_errors=0, num_rs_symbols_corrected=0,
        preamble_correlation=0.9, fec="viterbi",
    )
    dec2 = DecodingResult(
        spec_name=f"none|none|preamble+crc16",
        chain="none -> frame/crc16",
        ok=False, crc_valid=False,
        num_viterbi_errors=123, num_rs_symbols_corrected=0,
        preamble_correlation=0.1, fec="none",
    )

    graph = build_evidence_graph(
        snr=snr_obj,
        noise_floor=_NF(),
        bandwidth=_BW(),
        segmentation=_Seg(),
        mod_classification=None,
        demodulation=demod,
        decoding=[dec1, dec2],
        demod_modulation=mod,
    )
    return graph


class TestModels:
    def test_make_chain_id(self):
        a = make_chain_id()
        b = make_chain_id()
        assert a != b
        assert len(a) == 12

    def test_demod_params_grid(self):
        grid = make_demod_params_grid("QPSK", [1.0, 2.0, 4.0, 8.0])
        assert len(grid) == 4
        for p in grid:
            assert p.modulation == "QPSK"
            assert p.samples_per_symbol in (1.0, 2.0, 4.0, 8.0)

    def test_deinterleaver_params_grid(self):
        grid = make_deinterleaver_params_grid(["none", "block"], [(4, 8), (8, 16)])
        assert len(grid) == 3
        assert grid[0].type == "none"
        assert grid[1].type == "block" and grid[1].block_n == 4 and grid[1].block_m == 8

    def test_fec_params_grid(self):
        grid = make_fec_params_grid(["none", "viterbi", "rs255_223"])
        assert len(grid) == 3
        assert [f.mode for f in grid] == ["none", "viterbi", "rs255_223"]

    def test_frame_params_grid(self):
        grid = make_frame_params_grid(["preamble+crc16", "preamble+crc32"])
        assert len(grid) == 2
        assert grid[0].frame_name == "preamble+crc16"
        assert grid[1].frame_name == "preamble+crc32"
        assert grid[0].crc_variant == "crc16_ibm"

    def test_hypothesis_chain_to_dict(self):
        chain = HypothesisChain(
            id="abc123",
            modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK", samples_per_symbol=4.0),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="viterbi"),
            frame=FrameParams(
                frame_name="preamble+crc16",
                preamble=[0, 1],
                payload_lengths=[16],
                crc_variant="crc16_ibm",
            ),
        )
        d = chain.to_dict()
        assert d["modulation"] == "QPSK"
        assert d["demod_params"]["samples_per_symbol"] == 4.0
        assert d["fec"]["mode"] == "viterbi"
        assert d["composite_score"] == 0.0
        assert "why" in d

    def test_default_demod_params(self):
        assert "QPSK" in DEFAULT_DEMOD_PARAMS
        assert DEFAULT_DEMOD_PARAMS["QPSK"].samples_per_symbol == 2.0


class TestEnumeration:
    def test_enumerate_bounded(self):
        config = HypothesisEngineConfig(
            top_modulations=2,
            sps_candidates=[2.0, 4.0],
            deinterleaver_types=["none", "block"],
            block_sizes=[(4, 8)],
            fec_modes=["none", "viterbi"],
            frame_candidates=["preamble+crc16"],
            max_chains=100,
        )
        chains = enumerate_chain_candidates(config, ["QPSK", "BPSK"])
        assert len(chains) > 0
        assert len(chains) <= config.max_chains
        # Every chain must be a full path
        for c in chains:
            assert c.modulation in ("QPSK", "BPSK")
            assert c.demod_params.modulation == c.modulation

    def test_enumerate_skips_noise(self):
        config = HypothesisEngineConfig(
            top_modulations=5, max_chains=50,
            sps_candidates=[2.0],
            deinterleaver_types=["none"],
            fec_modes=["none"],
            frame_candidates=["preamble+crc16"],
        )
        chains = enumerate_chain_candidates(config, ["QPSK", "Noise/CW", "BPSK"])
        # Noise/CW produces only a single placeholder chain
        noise_chains = [c for c in chains if c.modulation == "Noise/CW"]
        assert len(noise_chains) == 1
        assert noise_chains[0].fec.mode == "none"
        assert noise_chains[0].frame.frame_name == "none"

    def test_enumerate_noise_first(self):
        config = HypothesisEngineConfig(top_modulations=5, max_chains=50)
        chains = enumerate_chain_candidates(config, ["Noise/CW", "QPSK"])
        noise_chains = [c for c in chains if c.modulation == "Noise/CW"]
        assert len(noise_chains) == 1

    def test_max_chains_enforced(self):
        config = HypothesisEngineConfig(
            top_modulations=3,
            sps_candidates=[1.0, 2.0, 4.0, 8.0, 16.0],
            deinterleaver_types=["none", "block"],
            block_sizes=[(4, 8), (8, 16), (16, 32)],
            fec_modes=["none", "viterbi", "rs255_223", "rs255_239",
                       "viterbi+rs255_223", "viterbi+rs255_239"],
            frame_candidates=["preamble+crc16", "preamble+crc32"],
            max_chains=40,
        )
        chains = enumerate_chain_candidates(config, ["QPSK", "BPSK", "16QAM"])
        assert len(chains) <= 40


class TestScoring:
    def test_score_modulation_from_evidence(self):
        graph = _make_evidence_graph("QPSK")
        score, evidence = score_modulation_from_evidence(graph, "QPSK")
        assert 0.0 <= score <= 1.0
        assert len(evidence) > 0

    def test_score_modulation_absent(self):
        graph = _make_evidence_graph("QPSK")
        score, evidence = score_modulation_from_evidence(graph, "GFSK")
        assert score == 0.0
        assert evidence == []

    def test_score_demodulation_from_evidence(self):
        graph = _make_evidence_graph("QPSK")
        score, evidence = score_demodulation_from_evidence(graph, "QPSK")
        assert 0.0 <= score <= 1.0
        assert any(e.stage == "demodulation" for e in evidence)

    def test_score_demodulation_from_result(self):
        demod = DemodulationResult(
            modulation="QPSK",
            evm=EVMResult(evm_rms_percent=5.0, evm_db=-26.0, evm_peak_percent=12.0),
            num_symbols=1024,
            bits=[0, 1] * 256,
            residual_phase_rad=0.01,
            gated=False,
        )
        score, evidence = score_demodulation_from_result(demod)
        assert score > 0.7

    def test_score_demodulation_gated(self):
        demod = DemodulationResult(
            modulation="QPSK", evm=None, num_symbols=0, bits=[],
            gated=True, why="too short",
        )
        score, evidence = score_demodulation_from_result(demod)
        assert score == 0.0

    def test_score_decoding_from_results_crc_valid(self):
        dec = DecodingResult(
            spec_name="viterbi|none|crc16", chain="viterbi",
            ok=True, crc_valid=True, num_viterbi_errors=0,
            num_rs_symbols_corrected=0, preamble_correlation=0.9,
        )
        score, evidence = score_decoding_from_results([dec])
        assert score > 0.7

    def test_score_decoding_from_results_crc_fail(self):
        dec = DecodingResult(
            spec_name="viterbi|none|crc16", chain="viterbi",
            ok=False, crc_valid=False, num_viterbi_errors=300,
            num_rs_symbols_corrected=0, preamble_correlation=0.05,
        )
        score, evidence = score_decoding_from_results([dec])
        assert score < 0.4

    def test_score_decoding_empty(self):
        assert score_decoding_from_results([]) == (0.0, [])

    def test_score_decoding_from_evidence(self):
        graph = _make_evidence_graph("QPSK")
        score, evidence = score_decoding_from_evidence(graph, "viterbi|none|preamble+crc16")
        assert 0.0 <= score <= 1.0

    def test_score_snr_margin(self):
        graph = _make_evidence_graph("QPSK", snr_db=20.0)
        margin = score_snr_margin(graph)
        assert margin > 0.7

    def test_compute_composite_score(self):
        chain = HypothesisChain(
            id="x", modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK"),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="viterbi"),
            frame=FrameParams(frame_name="preamble+crc16", preamble=[], payload_lengths=[16], crc_variant="crc16_ibm"),
            modulation_score=0.9, demodulation_score=0.8, decoding_score=0.9,
        )
        score = compute_composite_score(chain)
        assert 0.0 <= score <= 1.0
        assert score > 0.6

    def test_compute_composite_score_weights(self):
        chain = HypothesisChain(
            id="x", modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK"),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="viterbi"),
            frame=FrameParams(frame_name="preamble+crc16", preamble=[], payload_lengths=[16], crc_variant="crc16_ibm"),
            modulation_score=0.0, demodulation_score=0.0, decoding_score=1.0,
        )
        weights = ChainScoringWeights(w_modulation=0.25, w_demodulation=0.25, w_decoding=0.35, w_snr_margin=0.15)
        score = compute_composite_score(chain, weights=weights, snr_db=20.0)
        # decoding_weight_capped: w_decoding * 1.0 + w_snr * 1.0 = 0.5
        assert abs(score - (0.35 + 0.15)) < 1e-6


class TestUncertainty:
    def test_quantify_uncertainty(self):
        graph = _make_evidence_graph("QPSK")
        chains = []
        for i in range(3):
            c = HypothesisChain(
                id=f"c{i}", modulation="QPSK",
                demod_params=DemodParams(modulation="QPSK"),
                deinterleaver=DeinterleaverParams(type="none"),
                fec=FECParams(mode="viterbi"),
                frame=FrameParams(frame_name="preamble+crc16", preamble=[], payload_lengths=[16], crc_variant="crc16_ibm"),
                modulation_score=0.9, demodulation_score=0.8, decoding_score=0.9,
                composite_score=0.85,
                modulation_evidence=graph.scoring_items_for("QPSK"),
                demodulation_evidence=graph.scoring_items_for("QPSK"),
            )
            chains.append(c)
        u = quantify_uncertainty(chains[0], graph, chains)
        assert isinstance(u, ScoreUncertainty)
        assert 0.0 <= u.std_dev <= 0.5
        assert u.confidence_interval_95 is not None
        lo, hi = u.confidence_interval_95
        assert lo <= hi
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0

    def test_uncertainty_low_evidence_high(self):
        graph = _make_evidence_graph("QPSK")
        chain = HypothesisChain(
            id="x", modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK"),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="viterbi"),
            frame=FrameParams(frame_name="preamble+crc16", preamble=[], payload_lengths=[16], crc_variant="crc16_ibm"),
            composite_score=0.7,
        )
        # No evidence attached -> high uncertainty
        u = quantify_uncertainty(chain, graph, [chain])
        assert u.evidence_completeness < 0.5
        assert u.std_dev > 0.1


class TestRanking:
    def test_rank_chains(self):
        chains = [
            HypothesisChain(id=f"c{i}", modulation=m,
                            demod_params=DemodParams(modulation=m),
                            deinterleaver=DeinterleaverParams(type="none"),
                            fec=FECParams(mode="none"),
                            frame=FrameParams(frame_name="preamble+crc16", preamble=[], payload_lengths=[16], crc_variant="crc16_ibm"),
                            composite_score=s)
            for i, (m, s) in enumerate([("QPSK", 0.3), ("QPSK", 0.9), ("BPSK", 0.6)])
        ]
        ranked = rank_chains(chains, k=2)
        assert len(ranked) == 2
        assert ranked[0].id == "c1"
        assert ranked[1].id == "c2"

    def test_compare_chains(self):
        a = HypothesisChain(id="a", modulation="QPSK",
                            demod_params=DemodParams(modulation="QPSK"),
                            deinterleaver=DeinterleaverParams(type="none"),
                            fec=FECParams(mode="viterbi"),
                            frame=FrameParams(frame_name="preamble+crc16", preamble=[], payload_lengths=[16], crc_variant="crc16_ibm"),
                            modulation_score=0.9, demodulation_score=0.8, decoding_score=0.9,
                            composite_score=0.85)
        b = HypothesisChain(id="b", modulation="BPSK",
                            demod_params=DemodParams(modulation="BPSK"),
                            deinterleaver=DeinterleaverParams(type="none"),
                            fec=FECParams(mode="none"),
                            frame=FrameParams(frame_name="preamble+crc32", preamble=[], payload_lengths=[16], crc_variant="crc32_ieee"),
                            modulation_score=0.5, demodulation_score=0.4, decoding_score=0.5,
                            composite_score=0.5)
        diff = compare_chains(a, b)
        assert diff["score_diff"] > 0
        assert diff["modulation_same"] is False
        assert diff["fec_diff"] is True
        assert diff["frame_diff"] is True


class TestExplain:
    def test_generate_why(self):
        graph = _make_evidence_graph("QPSK")
        chain = HypothesisChain(
            id="x", modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK", samples_per_symbol=4.0),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="viterbi"),
            frame=FrameParams(frame_name="preamble+crc16", preamble=[0, 1], payload_lengths=[16], crc_variant="crc16_ibm"),
            modulation_score=0.85, demodulation_score=0.75, decoding_score=0.8,
            composite_score=0.8,
            modulation_evidence=graph.scoring_items_for("QPSK")[:3],
            demodulation_evidence=graph.scoring_items_for("QPSK")[:2],
        )
        text = generate_why(chain, graph)
        assert "Hypothesis:" in text
        assert "WHY this modulation" in text
        assert "QPSK" in text

    def test_generate_short_why(self):
        chain = HypothesisChain(
            id="x", modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK"),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="viterbi"),
            frame=FrameParams(frame_name="preamble+crc16", preamble=[], payload_lengths=[16], crc_variant="crc16_ibm"),
            modulation_score=0.9, demodulation_score=0.8, decoding_score=0.9,
            composite_score=0.85,
        )
        short = generate_short_why(chain)
        assert "QPSK" in short
        assert "CRC-valid frame" in short

    def test_generate_comparison_why(self):
        a = HypothesisChain(id="a", modulation="QPSK",
                            demod_params=DemodParams(modulation="QPSK"),
                            deinterleaver=DeinterleaverParams(type="none"),
                            fec=FECParams(mode="viterbi"),
                            frame=FrameParams(frame_name="preamble+crc16", preamble=[], payload_lengths=[16], crc_variant="crc16_ibm"),
                            composite_score=0.85)
        b = HypothesisChain(id="b", modulation="BPSK",
                            demod_params=DemodParams(modulation="BPSK"),
                            deinterleaver=DeinterleaverParams(type="none"),
                            fec=FECParams(mode="none"),
                            frame=FrameParams(frame_name="preamble+crc32", preamble=[], payload_lengths=[16], crc_variant="crc32_ieee"),
                            composite_score=0.5)
        text = generate_comparison_why(a, b)
        assert "Score difference" in text

    def test_generate_hypothesis_report(self):
        chains = [
            HypothesisChain(id="a", modulation="QPSK",
                            demod_params=DemodParams(modulation="QPSK"),
                            deinterleaver=DeinterleaverParams(type="none"),
                            fec=FECParams(mode="viterbi"),
                            frame=FrameParams(frame_name="preamble+crc16", preamble=[], payload_lengths=[16], crc_variant="crc16_ibm"),
                            composite_score=0.85),
            HypothesisChain(id="b", modulation="BPSK",
                            demod_params=DemodParams(modulation="BPSK"),
                            deinterleaver=DeinterleaverParams(type="none"),
                            fec=FECParams(mode="none"),
                            frame=FrameParams(frame_name="preamble+crc32", preamble=[], payload_lengths=[16], crc_variant="crc32_ieee"),
                            composite_score=0.5),
        ]
        report = generate_hypothesis_report(chains)
        assert "HYPOTHESIS ENGINE REPORT" in report
        assert "Rank 1" in report

    def test_generate_chain_json(self):
        chains = [HypothesisChain(id="a", modulation="QPSK",
                                  demod_params=DemodParams(modulation="QPSK"),
                                  deinterleaver=DeinterleaverParams(type="none"),
                                  fec=FECParams(mode="viterbi"),
                                  frame=FrameParams(frame_name="preamble+crc16", preamble=[], payload_lengths=[16], crc_variant="crc16_ibm"),
                                  composite_score=0.85)]
        data = generate_chain_json(chains)
        assert isinstance(data, list)
        assert data[0]["rank"] == 1
        assert "why" in data[0]
        assert "chain" in data[0]


class TestSearch:
    def test_chain_to_decoding_spec(self):
        # Ensure internal conversion works
        from app.hypothesis.search import _chain_to_decoding_spec
        chain = HypothesisChain(
            id="x", modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK", samples_per_symbol=4.0),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="viterbi"),
            frame=FrameParams(frame_name="preamble+crc16", preamble=[0, 1, 1, 1, 1, 1, 1, 0] * 2,
                              payload_lengths=[16, 32], crc_variant="crc16_ibm"),
        )
        spec = _chain_to_decoding_spec(chain)
        assert spec.name.startswith("viterbi|none")
        assert spec.apply_viterbi is True

    def test_evaluate_chain_live_qpsk(self):
        signal = _make_signal(4096, "QPSK", sps=4)
        graph = _make_evidence_graph("QPSK")
        config = HypothesisEngineConfig(run_demodulation=True, run_decoding=True)
        chain = HypothesisChain(
            id="x", modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK", samples_per_symbol=4.0),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="none"),
            frame=FrameParams(frame_name="preamble+crc16", preamble=[0, 1, 1, 1, 1, 1, 1, 0] * 2,
                              payload_lengths=[16, 100], crc_variant="crc16_ibm"),
        )
        results = evaluate_chain_live(chain, signal, config, graph)
        assert chain.composite_score >= 0.0
        assert chain.demodulation_score > 0.5  # clean QPSK demod
        assert 0.0 <= chain.modulation_score <= 1.0

    def test_run_hypothesis_search_top_qpsk(self):
        signal = _make_signal(4096, "QPSK", sps=4)
        graph = _make_evidence_graph("QPSK")
        config = HypothesisEngineConfig(
            top_modulations=3,
            sps_candidates=[2.0, 4.0],
            deinterleaver_types=["none"],
            fec_modes=["none", "viterbi"],
            frame_candidates=["preamble+crc16"],
            max_chains=30,
            top_k_output=5,
        )
        chains = run_hypothesis_search(signal, graph, config)
        assert len(chains) == 5
        # QPSK should be the top modulation
        assert chains[0].modulation == "QPSK"

    def test_hypothesize_convenience(self):
        signal = _make_signal(4096, "QPSK", sps=4)
        graph = _make_evidence_graph("QPSK")
        chains = hypothesize(signal, graph, top_k=3)
        assert len(chains) <= 3
        for c in chains:
            assert c.why or c.composite_score >= 0.0


class TestSpsSeeding:
    """Bug A: the hypothesis engine's demod must reuse the estimated sps,
    never fall back to treating every raw sample as a symbol (sps=1.0)."""

    def test_seed_sps_candidates_prefers_estimated_from_graph(self):
        from app.hypothesis.search import _seed_sps_candidates
        signal = _make_signal(4096, "QPSK", sps=16)
        graph = EvidenceGraph()
        graph.add_item(EvidenceItem(
            id="signal:symbol_rate_hz", stage="characterization",
            candidate="signal", measurement="symbol_rate_hz",
            value=signal.metadata.sample_rate / 16.0, value_norm=0.5,
            supports=True, weight=0.5, source="symbol_rate.estimate_symbol_rate",
        ))
        graph.add_item(EvidenceItem(
            id="signal:symbol_rate_confidence", stage="characterization",
            candidate="signal", measurement="symbol_rate_confidence",
            value=0.9, value_norm=0.9, supports=True, weight=0.3,
            source="symbol_rate.estimate_symbol_rate",
        ))
        seeded = _seed_sps_candidates(signal, graph, [1.0, 2.0, 4.0])
        assert seeded[0] == pytest.approx(16.0, abs=0.5)
        # The estimated point must always be in the grid, and deduped.
        assert any(abs(s - 16.0) < 0.5 for s in seeded)
        assert len(seeded) == len(set(seeded))

    def test_run_hypothesis_search_demods_at_estimated_sps(self):
        # sps=16 burst: the fixed default grid [1,2,4] previously never even
        # tried the estimated clock (~14-16), and the sps=1 entry demoded every
        # raw sample as a symbol. The estimate must now be seeded so a chain is
        # live-evaluated at the real symbol clock.
        from app.hypothesis.search import _seed_sps_candidates
        signal = _make_signal(4096, "QPSK", sps=16)
        graph = _make_evidence_graph("QPSK")
        config = HypothesisEngineConfig(
            top_modulations=1,
            run_demodulation=True,
            run_decoding=False,
            deinterleaver_types=["none"],
            fec_modes=["none"],
            frame_candidates=["preamble+crc16"],
            max_chains=50,
            top_k_output=50,
        )
        chains = run_hypothesis_search(signal, graph, config)
        est = _seed_sps_candidates(signal, graph, config.sps_candidates)[0]
        est_chains = [c for c in chains
                      if abs(c.demod_params.samples_per_symbol - est) < 0.75]

        assert est_chains, f"estimated sps {est:.2f} was never evaluated"
        # Chains demodulated at the seeded working point must use the real
        # symbol clock: num_symbols ~ n/sps, never the raw sample count.
        for c in est_chains:
            ev = [e for e in c.demodulation_evidence
                  if e.measurement == "evm_rms_percent"]
            assert ev, "estimated-sps chain must carry demod evidence"
            n_sym = ev[0].context.get("num_symbols", 0)
            assert n_sym < 3000, (
                f"estimated-sps demod produced {n_sym} symbols ~ raw sample count"
            )
            assert ev[0].context.get("samples_per_symbol", 0.0) > 3.0


class TestDecodeUnification:
    """Bug B: hypothesis-engine decode evidence and the standalone leaderboard
    (`search_decode` default grid) must share one candidate space and agree."""

    def _leaderboard(self, signal):
        from app.demodulation.slicer import demodulate_signal
        from app.decoding.decoder import search_decode
        demod = demodulate_signal(signal, modulation="QPSK",
                                  samples_per_symbol=4.0, max_symbols=4096)
        lb = search_decode(demod.bits, llrs=demod.llrs_sample or None,
                           use_soft_viterbi=True, top_k=500)
        return lb, demod

    def test_hypothesis_decode_uses_leaderboard_grid(self):
        from app.decoding.decoder import search_decode
        from app.demodulation.slicer import demodulate_signal
        signal = _make_signal(4096, "QPSK", sps=4)
        graph = _make_evidence_graph("QPSK")
        lb, demod = self._leaderboard(signal)
        lb_by_name = {r.spec_name: r for r in lb}

        config = HypothesisEngineConfig(
            top_modulations=1,
            sps_candidates=[4.0],
            deinterleaver_types=["none"],
            fec_modes=["none", "viterbi"],
            frame_candidates=["preamble+crc16"],
            max_chains=20,
            top_k_output=10,
        )
        chains = run_hypothesis_search(signal, graph, config)
        assert chains
        top = chains[0]
        assert top.decoding_results, "top chain missing decode evidence"
        for r in top.decoding_results:
            # Every engine decode result is a member of the leaderboard grid.
            assert r.spec_name in lb_by_name, r.spec_name
        # The chain's own spec maps to the SAME ranked result as the leaderboard.
        own = top.decoding_results[0]
        assert own.crc_valid == lb_by_name[own.spec_name].crc_valid
        assert own.spec_name == lb_by_name[own.spec_name].spec_name

    def test_config_can_pin_shared_grid(self):
        # A caller pins the decode search in ONE place (Manual/HITL mode);
        # the engine must search exactly that grid, nothing else.
        from app.decoding.decoder import DecodingSpec, none_spec, FEC_NONE
        pinned = [DecodingSpec(
            name="none|none|preamble+crc16",
            deinterleaver=none_spec(), fec=FEC_NONE,
            frame_descriptor=None,
        )]
        config = HypothesisEngineConfig(
            run_demodulation=True,
            run_decoding=True,
            decode_spec_grid=pinned,
        )
        chain = HypothesisChain(
            id="x", modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK", samples_per_symbol=4.0),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="none"),
            frame=FrameParams(frame_name="preamble+crc16", preamble=[0, 1],
                              payload_lengths=[16], crc_variant="crc16_ibm"),
        )
        signal = _make_signal(4096, "QPSK", sps=4)
        graph = _make_evidence_graph("QPSK")
        evaluate_chain_live(chain, signal, config, graph)
        # Only the pinned spec is searched, and the chain's own result maps to it.
        assert [r.spec_name for r in chain.decoding_results] == ["none|none|preamble+crc16"]
        # decode_spec_grid round-trips through to_dict
        d = config.to_dict()
        assert d["decode_spec_grid"] is not None


def test_downstream_success_raises_family_score():
    """A CRC-valid decode chain must raise the upstream modulation family score."""
    # Build graph with a strong family + CRC-valid chain
    graph = _make_evidence_graph("QPSK")
    # Evidence graph auto-scores; the QPSK family should be strong
    qpsk = graph.hypotheses["QPSK"]
    assert qpsk.score > 0.5
    # Top decode chain for the family should benefit too
    decode_nodes = [h for h in graph.hypotheses.values() if h.kind == "decode_chain"]
    assert len(decode_nodes) > 0
    # CRC-valid chain ranks above CRC-fail
    valid = graph.hypotheses.get("viterbi|none|preamble+crc16")
    assert valid is not None
    assert valid.score > 0.5


class TestCrcDifferentialScoring:
    """decode-chain scoring must prefer a real CRC-valid payload over ties."""

    @staticmethod
    def _cr(spec_name, crc_valid, preamble_corr=0.95, payload_bits=None, fec="none", viterbi=0, rs=0):
        return DecodingResult(
            spec_name=spec_name,
            chain=f"deint(none) -> frame/{spec_name}",
            ok=bool(crc_valid),
            crc_valid=crc_valid,
            payload_bits=payload_bits or [0, 1, 1, 0],
            payload_bytes=bytes([0x55] * 16),
            num_viterbi_errors=viterbi,
            num_rs_symbols_corrected=rs,
            preamble_correlation=preamble_corr,
            frame_offset=0,
            used_soft=True,
            fec=fec,
            why="test",
        )

    def test_crc_valid_beats_crc_invalid(self):
        s_good, _ = score_decoding_from_results([self._cr("preamble+crc16", crc_valid=True)])
        s_bad, _ = score_decoding_from_results([self._cr("preamble+crc32", crc_valid=False)])
        assert s_good > s_bad
        assert s_good > 0.7

    def test_identical_results_score_identically(self):
        a, _ = score_decoding_from_results([self._cr("x", True)])
        b, _ = score_decoding_from_results([self._cr("x", True)])
        assert a == b

    def test_identical_chains_score_identically(self):
        # Two chains carrying the same live DecodingResults must rank equally
        chain_a = HypothesisChain(
            id="a", modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK", samples_per_symbol=4.0),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="none"),
            frame=FrameParams(frame_name="preamble+crc16", preamble=[0, 1], payload_lengths=[16], crc_variant="crc16_ibm"),
        )
        chain_b = HypothesisChain(
            id="b", modulation="QPSK",
            demod_params=DemodParams(modulation="QPSK", samples_per_symbol=4.0),
            deinterleaver=DeinterleaverParams(type="none"),
            fec=FECParams(mode="none"),
            frame=FrameParams(frame_name="preamble+crc16", preamble=[0, 1], payload_lengths=[16], crc_variant="crc16_ibm"),
        )
        for c in (chain_a, chain_b):
            c.decoding_results = [self._cr("x", True)]
            c.modulation_evidence = []
            c.demodulation_evidence = []
        s_a = score_decoding_from_results(chain_a.decoding_results)[0]
        s_b = score_decoding_from_results(chain_b.decoding_results)[0]
        assert s_a == s_b

    def test_full_pipeline_graph_scores_crc_valid_chain(self):
        # CRC-valid chain in the evidence graph raises the family score
        graph = _make_evidence_graph("QPSK")
        valid = graph.hypotheses.get("viterbi|none|preamble+crc16")
        fail = graph.hypotheses.get("none|none|preamble+crc16")
        assert valid is not None and fail is not None
        assert valid.score >= fail.score