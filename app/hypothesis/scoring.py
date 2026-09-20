"""
app/hypothesis/scoring.py
-------------------------
Composite scoring for hypothesis chains.

Combines evidence from modulation classification, demodulation verification,
and decoding chain results into a single calibrated composite score with
uncertainty quantification.
"""

from __future__ import annotations

import math
from typing import Optional

from app.evidence.models import EvidenceGraph, EvidenceItem, clamp01, make_evidence_id
from app.decoding.decoder import DecodingResult
from app.hypothesis.models import (
    HypothesisChain,
    ChainScoringWeights,
    ScoreUncertainty,
)


# ---------------------------------------------------------------------------
# Modulation scoring from EvidenceGraph
# ---------------------------------------------------------------------------

def score_modulation_from_evidence(
    graph: EvidenceGraph,
    modulation: str,
) -> tuple[float, list[EvidenceItem]]:
    """
    Compute modulation evidence score from the EvidenceGraph.
    
    Returns (score in [0,1], evidence_items).
    """
    hypothesis = graph.hypotheses.get(modulation)
    if hypothesis is None:
        return 0.0, []
    
    evidence = graph.scoring_items_for(modulation)
    return hypothesis.score, evidence


def score_modulation_from_classifier(
    graph: EvidenceGraph,
    modulation: str,
) -> tuple[float, list[EvidenceItem]]:
    """
    Score based specifically on classifier evidence items.
    """
    evidence = [e for e in graph.scoring_items_for(modulation)
                if e.measurement.startswith(("classifier_", "fit_", "top_"))]
    if not evidence:
        return 0.0, []
    
    # Weighted average of classifier evidence
    total_w = sum(e.weight for e in evidence)
    if total_w < 1e-9:
        return 0.0, []
    
    weighted_sum = sum(e.value_norm * e.weight for e in evidence if e.supports)
    return clamp01(weighted_sum / total_w), evidence


# ---------------------------------------------------------------------------
# Demodulation scoring
# ---------------------------------------------------------------------------

def score_demodulation_from_evidence(
    graph: EvidenceGraph,
    modulation: str,
) -> tuple[float, list[EvidenceItem]]:
    """
    Compute demodulation quality score from EvidenceGraph.
    
    Returns (score in [0,1], evidence_items).
    """
    evidence = [e for e in graph.scoring_items_for(modulation)
                if e.stage == "demodulation"]
    if not evidence:
        return 0.0, []
    
    # Key metrics: EVM (high weight), num_symbols, residual_phase
    total_w = sum(e.weight for e in evidence)
    if total_w < 1e-9:
        return 0.0, []
    
    weighted_sum = sum(e.value_norm * e.weight for e in evidence if e.supports)
    return clamp01(weighted_sum / total_w), evidence


def score_demodulation_from_result(
    demod_result,
) -> tuple[float, list]:
    """
    Score demodulation from a DemodulationResult object directly.
    
    Returns (score, evidence_items_as_dicts).
    """
    if demod_result is None or demod_result.gated:
        if demod_result is not None and demod_result.gated:
            gated_ev = EvidenceItem(
                id=make_evidence_id("gated", "demod_gated"),
                stage="demodulation",
                candidate=demod_result.modulation or "unknown",
                measurement="gated",
                value=1.0,
                value_norm=0.0,
                supports=False,
                weight=1.0,
                source="demodulator",
                provenance="signal_demodulated",
                context={"why": demod_result.why[:160] if demod_result.why else "gated"},
            )
            return 0.0, [gated_ev]
        return 0.0, []

    score = 0.0
    weights = 0.0
    evidence: list[EvidenceItem] = []

    # Working-point provenance: which sps actually produced this demod, so
    # downstream consumers (WHY, HITL, report) can detect a fallback-default
    # demod (sps==1.0, one symbol per raw sample) instead of an estimated one.
    demod_ctx = {
        "samples_per_symbol": float(getattr(demod_result, "samples_per_symbol", 0.0) or 0.0),
        "num_symbols": int(getattr(demod_result, "num_symbols", 0) or 0),
        "symbol_rate_hz": float(getattr(demod_result, "symbol_rate_hz", 0.0) or 0.0),
        "gated": bool(getattr(demod_result, "gated", False)),
    }

    # EVM score (most important)
    if demod_result.evm is not None:
        evm_rms = demod_result.evm.evm_rms_percent
        evm_score = clamp01(1.0 - evm_rms / 30.0)  # 30% EVM -> 0
        score += evm_score * 1.5
        weights += 1.5
        evidence.append(
            EvidenceItem(
                id=make_evidence_id("evm", "evm_rms_percent"),
                stage="demodulation",
                candidate=demod_result.modulation or "unknown",
                measurement="evm_rms_percent",
                value=float(evm_rms),
                value_norm=evm_score,
                supports=evm_score > 0.5,
                weight=1.5,
                source="demodulator",
                provenance="signal_demodulated",
                unit="%",
                context=dict(demod_ctx),
            )
        )
        if demod_result.evm.evm_db is not None:
            evidence.append(
                EvidenceItem(
                    id=make_evidence_id("evm_db", demod_result.modulation or "unknown"),
                    stage="demodulation",
                    candidate=demod_result.modulation or "unknown",
                    measurement="evm_db",
                    value=float(demod_result.evm.evm_db),
                    value_norm=evm_score,
                    supports=evm_score > 0.5,
                    weight=0.8,
                    source="demodulator",
                    provenance="signal_demodulated",
                    unit="dB",
                )
            )

    # Symbol count
    n_sym = demod_result.num_symbols
    sym_score = clamp01(n_sym / 512.0)
    score += sym_score * 0.5
    weights += 0.5
    evidence.append(
        EvidenceItem(
            id=make_evidence_id("num_symbols", demod_result.modulation or "unknown"),
            stage="demodulation",
            candidate=demod_result.modulation or "unknown",
            measurement="num_symbols",
            value=float(n_sym),
            value_norm=sym_score,
            supports=sym_score > 0.4,
            weight=0.5,
            source="demodulator",
            provenance="signal_demodulated",
        )
    )

    # Residual phase
    residual = abs(demod_result.residual_phase_rad)
    phase_score = clamp01(1.0 - residual / (math.pi / 4.0))
    score += phase_score * 0.5
    weights += 0.5
    evidence.append(
        EvidenceItem(
            id=make_evidence_id("residual_phase", demod_result.modulation or "unknown"),
            stage="demodulation",
            candidate=demod_result.modulation or "unknown",
            measurement="residual_phase",
            value=float(demod_result.residual_phase_rad),
            value_norm=phase_score,
            supports=phase_score > 0.5,
            weight=0.5,
            source="demodulator",
            provenance="signal_demodulated",
            unit="rad",
        )
    )

    if weights < 1e-9:
        return 0.0, []

    return clamp01(score / weights), evidence


# ---------------------------------------------------------------------------
# Decoding scoring
# ---------------------------------------------------------------------------

def score_decoding_from_evidence(
    graph: EvidenceGraph,
    chain_name: str,
) -> tuple[float, list[EvidenceItem]]:
    """
    Compute decoding chain score from EvidenceGraph.
    """
    hypothesis = graph.hypotheses.get(chain_name)
    if hypothesis is None:
        return 0.0, []
    
    evidence = graph.scoring_items_for(chain_name)
    if not evidence:
        return 0.0, []
    
    total_w = sum(e.weight for e in evidence)
    if total_w < 1e-9:
        return 0.0, []
    
    weighted_sum = sum(e.value_norm * e.weight for e in evidence if e.supports)
    return clamp01(weighted_sum / total_w), evidence


def score_decoding_from_results(
    results: list[DecodingResult],
) -> tuple[float, list]:
    """
    Score decoding from a list of DecodingResult objects.
    
    Uses the same ranking criteria as search_decode:
    - CRC-valid frame > clean FEC > no frame
    - Fewer Viterbi errors
    - More RS corrections
    - Higher preamble correlation
    
    Bug C fix: when CRC fails, the score is proportional to how close
    the chain got (preamble lock, residual Viterbi/RS activity, payload
    recovery) — capped well below a CRC-passing chain so that different
    FEC/interleaver choices remain distinguishable.
    """
    if not results:
        return 0.0, []
    
    # Take the best result (already ranked by search_decode)
    best = results[0]
    
    score = 0.0
    weights = 0.0
    
    # CRC validity (strongest signal)
    if best.crc_valid is True:
        score += 2.0 * 1.5
        weights += 1.5
    elif best.crc_valid is False:
        # CRC failed: score proportional to proximity of a valid decode.
        # Preamble lock alone is NOT decode success — cap well below
        # a CRC-passing chain so different FEC modes remain distinguishable.
        #
        # Proximity factors: preamble correlation (40%), clean FEC /
        # low Viterbi errors (30%), RS corrections (20%), payload recovery (10%).
        proximity = (
            best.preamble_correlation * 0.40
            + (1.0 - clamp01(best.num_viterbi_errors / 32.0)) * 0.30
            + clamp01(best.num_rs_symbols_corrected / 16.0) * 0.20
            + (len(best.payload_bits) / 500.0 if best.payload_bits else 0.0) * 0.10
        )
        # Cap failed CRC at ~0.5 of the full CRC-passing score.
        # This ensures no failed chain can ever score as high as a passing one,
        # while still differentiating between FEC modes (e.g., a Viterbi+RS
        # chain that corrected 5 symbols scores higher than a "none" chain
        # that corrected 0, even if both have CRC=FAIL).
        failed_score = clamp01(proximity * 0.5)
        score += failed_score * 1.5
        weights += 1.5
    else:
        score += 0.5 * 1.0
        weights += 1.0
    
    # Viterbi residual errors (fewer is better)
    n_verr = best.num_viterbi_errors
    verr_score = clamp01(1.0 - n_verr / 32.0)
    score += verr_score * 1.0
    weights += 1.0
    
    # RS symbols corrected (more is better, up to a point)
    n_rs = best.num_rs_symbols_corrected
    rs_score = clamp01(n_rs / 16.0)
    score += rs_score * 0.5
    weights += 0.5
    
    # Preamble correlation
    corr = best.preamble_correlation
    score += corr * 0.5
    weights += 0.5
    
    # Payload length (longer valid payload is better)
    payload_len = len(best.payload_bits) if best.payload_bits else 0
    payload_score = clamp01(payload_len / 500.0)
    score += payload_score * 0.3
    weights += 0.3
    
    if weights < 1e-9:
        return 0.0, []
    
    return clamp01(score / weights), []


# ---------------------------------------------------------------------------
# SNR margin scoring
# ---------------------------------------------------------------------------

def score_snr_margin(
    graph: EvidenceGraph,
) -> float:
    """
    Score based on SNR margin above minimum operating threshold.
    """
    snr_items = [e for e in graph.items.values()
                 if e.candidate == "signal" and e.measurement == "snr_db"]
    if not snr_items:
        return 0.5  # neutral
    
    snr_db = snr_items[0].value
    # Minimum viable SNR ~ 3 dB, comfortable > 10 dB
    return clamp01((snr_db - 3.0) / 15.0)


# ---------------------------------------------------------------------------
# Composite chain scoring
# ---------------------------------------------------------------------------

def compute_composite_score(
    chain: HypothesisChain,
    graph: Optional[EvidenceGraph] = None,
    weights: Optional[ChainScoringWeights] = None,
    snr_db: Optional[float] = None,
) -> float:
    """
    Compute the composite chain score from component scores.
    
    The score is a weighted combination of:
    - Modulation evidence (classifier + feature traceability)
    - Demodulation quality (EVM + symbol count + phase lock)
    - Decoding success (CRC + FEC + frame sync)
    - SNR margin (operating headroom)
    """
    if weights is None:
        weights = ChainScoringWeights()
    
    # SNR margin score
    snr_score = 0.5
    if snr_db is not None:
        snr_score = clamp01((snr_db - 3.0) / 15.0)
    elif graph:
        snr_score = score_snr_margin(graph)
    
    composite = (
        weights.w_modulation * chain.modulation_score +
        weights.w_demodulation * chain.demodulation_score +
        weights.w_decoding * chain.decoding_score +
        weights.w_snr_margin * snr_score
    )
    
    return clamp01(composite)


def quantify_uncertainty(
    chain: HypothesisChain,
    graph: Optional[EvidenceGraph] = None,
    all_chains: Optional[list[HypothesisChain]] = None,
) -> ScoreUncertainty:
    """
    Quantify uncertainty in the chain score.
    
    Returns a ScoreUncertainty object with std_dev, confidence interval,
    and evidence-based uncertainty factors.
    """
    score = chain.composite_score
    
    # Evidence completeness: fraction of expected evidence items present
    expected_evidence = 0
    present_evidence = 0
    
    # Modulation stage
    expected_evidence += 3  # classifier_score, confidence, top_margin
    present_evidence += len(chain.modulation_evidence)
    
    # Demodulation stage
    expected_evidence += 3  # evm, num_symbols, residual_phase
    present_evidence += len(chain.demodulation_evidence)
    
    # Decoding stage
    expected_evidence += 4  # crc, decode_ok, verr, rs_corr
    present_evidence += len(chain.decoding_evidence)
    
    evidence_completeness = present_evidence / max(expected_evidence, 1)
    
    # Modulation margin: gap to 2nd best
    modulation_margin = 0.0
    if all_chains:
        other_mods = [c for c in all_chains if c.modulation != chain.modulation]
        if other_mods:
            other_best = max(c.composite_score for c in other_mods)
            modulation_margin = max(0.0, chain.composite_score - other_best)
    
    # Demodulation confidence from EVM
    demod_conf = 0.0
    for e in chain.demodulation_evidence:
        if "evm" in e.measurement:
            demod_conf = max(demod_conf, e.value_norm)
    
    # Decoding consistency: agreement among top decode chains for this modulation
    decoding_consistency = 0.0
    if all_chains:
        same_mod = [c for c in all_chains if c.modulation == chain.modulation]
        if len(same_mod) >= 2:
            top_scores = sorted([c.decoding_score for c in same_mod], reverse=True)
            decoding_consistency = 1.0 - (top_scores[0] - top_scores[1]) if top_scores[0] > 0 else 0.0
            decoding_consistency = clamp01(decoding_consistency)
        elif same_mod:
            decoding_consistency = same_mod[0].decoding_score
    
    # Combined uncertainty factors
    uncertainty_factors = [
        (1.0 - evidence_completeness) * 0.3,
        (1.0 - modulation_margin) * 0.25,
        (1.0 - demod_conf) * 0.25,
        (1.0 - decoding_consistency) * 0.2,
    ]
    std_dev = sum(uncertainty_factors)
    std_dev = min(std_dev, 0.5)  # cap
    
    # 95% CI
    ci_low = max(0.0, score - 1.96 * std_dev)
    ci_high = min(1.0, score + 1.96 * std_dev)
    
    return ScoreUncertainty(
        score=score,
        std_dev=std_dev,
        confidence_interval_95=(ci_low, ci_high),
        evidence_completeness=evidence_completeness,
        modulation_margin=modulation_margin,
        demodulation_confidence=demod_conf,
        decoding_consistency=decoding_consistency,
    )


# ---------------------------------------------------------------------------
# Chain ranking and comparison
# ---------------------------------------------------------------------------

def rank_chains(
    chains: list[HypothesisChain],
    k: int = 10,
) -> list[HypothesisChain]:
    """Return top-K chains by composite score."""
    chains.sort(key=lambda c: c.composite_score, reverse=True)
    return chains[:k]


def compare_chains(
    chain_a: HypothesisChain,
    chain_b: HypothesisChain,
) -> dict:
    """Compare two chains and return differences."""
    return {
        "score_diff": chain_a.composite_score - chain_b.composite_score,
        "modulation_diff": chain_a.modulation_score - chain_b.modulation_score,
        "demodulation_diff": chain_a.demodulation_score - chain_b.demodulation_score,
        "decoding_diff": chain_a.decoding_score - chain_b.decoding_score,
        "modulation_same": chain_a.modulation == chain_b.modulation,
        "fec_diff": chain_a.fec.mode != chain_b.fec.mode,
        "frame_diff": chain_a.frame.frame_name != chain_b.frame.frame_name,
    }