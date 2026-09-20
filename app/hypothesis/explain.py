"""
app/hypothesis/explain.py
--------------------------
Natural-language WHY explanation generation for hypothesis chains.

Produces human-readable justifications citing specific evidence items
from the EvidenceGraph at each stage of the chain.
"""

from __future__ import annotations

from typing import Optional, Sequence

from app.evidence.models import EvidenceGraph, EvidenceItem
from app.hypothesis.models import HypothesisChain


# ---------------------------------------------------------------------------
# Evidence item extraction helpers
# ---------------------------------------------------------------------------

def _get_evidence_items(
    items: list[EvidenceItem],
    measurement: str,
) -> list[EvidenceItem]:
    """Filter evidence items by measurement name."""
    return [e for e in items if measurement in e.measurement]


def _format_evidence(e: EvidenceItem, include_value: bool = True) -> str:
    """Format a single evidence item as text."""
    parts = []
    if include_value:
        if e.unit:
            parts.append(f"{e.measurement}={e.value:.2f} {e.unit}")
        else:
            parts.append(f"{e.measurement}={e.value:.3f}")
    if e.supports:
        parts.append("supports")
    else:
        parts.append("contradicts")
    if e.context.get("why"):
        parts.append(f"({e.context['why'][:120]})")
    return ", ".join(parts)


def _format_modulation_evidence(chain: HypothesisChain) -> list[str]:
    """Format modulation-stage evidence for WHY."""
    lines = []
    items = chain.modulation_evidence
    
    # Classifier score
    classifier_items = _get_evidence_items(items, "classifier_score")
    for e in classifier_items:
        lines.append(f"  Classifier: {chain.modulation} score={e.value:.3f} ({'supports' if e.supports else 'contradicts'})")
    
    # Feature fits
    fit_items = _get_evidence_items(items, "fit_")
    for e in fit_items[:3]:  # top 3
        feat = e.measurement.replace("fit_", "")
        lines.append(f"  Feature {feat}: fit={e.value:.2f}")
    
    # Top margin
    margin_items = _get_evidence_items(items, "top_margin")
    for e in margin_items:
        lines.append(f"  Top-candidate margin: {e.value:.3f}")
    
    return lines


def _format_demodulation_evidence(chain: HypothesisChain) -> list[str]:
    """Format demodulation-stage evidence for WHY."""
    lines = []
    items = chain.demodulation_evidence
    
    # EVM
    evm_items = _get_evidence_items(items, "evm")
    for e in evm_items:
        if "rms" in e.measurement:
            lines.append(f"  EVM RMS: {e.value:.1f}% ({'good' if e.value_norm > 0.7 else 'fair' if e.value_norm > 0.4 else 'poor'})")
        elif "db" in e.measurement:
            lines.append(f"  EVM: {e.value:.1f} dB")
    
    # Symbols
    sym_items = _get_evidence_items(items, "num_symbols")
    for e in sym_items:
        lines.append(f"  Symbols recovered: {int(e.value)}")
    
    # Residual phase
    phase_items = _get_evidence_items(items, "residual_phase")
    for e in phase_items:
        lines.append(f"  Residual phase: {e.value:.3f} rad")
    
    # Gated
    gated_items = _get_evidence_items(items, "gated")
    for e in gated_items:
        if not e.supports:
            lines.append(f"  DEMOD GATED: {e.context.get('why', 'insufficient symbols')}")
    
    return lines


def _format_decoding_evidence(chain: HypothesisChain) -> list[str]:
    """Format decoding-stage evidence for WHY."""
    lines = []
    items = chain.decoding_evidence
    
    # CRC
    crc_items = _get_evidence_items(items, "crc_valid")
    for e in crc_items:
        status = "PASS" if e.value > 0.5 else "FAIL" if e.value < 0.5 else "N/A"
        lines.append(f"  Frame CRC: {status}")
    
    # Decode OK
    ok_items = _get_evidence_items(items, "decode_ok")
    for e in ok_items:
        lines.append(f"  Chain OK: {'yes' if e.value > 0.5 else 'no'}")
    
    # Viterbi errors
    verr_items = _get_evidence_items(items, "viterbi_residual")
    for e in verr_items:
        lines.append(f"  Viterbi residual errors: {int(e.value)}")
    
    # RS corrections
    rs_items = _get_evidence_items(items, "rs_symbols")
    for e in rs_items:
        if e.value > 0:
            lines.append(f"  RS symbols corrected: {int(e.value)}")
    
    # Preamble correlation
    corr_items = _get_evidence_items(items, "preamble_correlation")
    for e in corr_items:
        lines.append(f"  Preamble correlation: {e.value:.3f}")
    
    return lines


def _format_decoding_results(chain: HypothesisChain) -> list[str]:
    """Format DecodingResult objects for WHY."""
    if not chain.decoding_results:
        return ["  No decoding results available"]
    
    lines = []
    for i, res in enumerate(chain.decoding_results[:3]):
        crc_status = "PASS" if res.crc_valid else ("FAIL" if res.crc_valid is False else "N/A")
        lines.append(
            f"  Rank {i+1}: {res.chain} | CRC={crc_status} | "
            f"Viterbi errs={res.num_viterbi_errors} | RS corr={res.num_rs_symbols_corrected} | "
            f"preamble corr={res.preamble_correlation:.3f}"
        )
        if res.why:
            lines.append(f"    -> {res.why[:200]}")
    return lines


# ---------------------------------------------------------------------------
# WHY generation
# ---------------------------------------------------------------------------

def generate_why(
    chain: HypothesisChain,
    graph: Optional[EvidenceGraph] = None,
    include_alternatives: bool = True,
) -> str:
    """
    Generate a natural-language WHY justification for a hypothesis chain.
    
    Cites specific evidence values from each pipeline stage.
    """
    lines = []
    
    # Header
    lines.append(f"Hypothesis: {chain.modulation} -> {chain.demod_params} -> "
                 f"{chain.deinterleaver} -> {chain.fec} -> {chain.frame}")
    lines.append(f"Composite Score: {chain.composite_score:.3f} "
                 f"(mod={chain.modulation_score:.2f}, demod={chain.demodulation_score:.2f}, "
                 f"decode={chain.decoding_score:.2f})")
    lines.append("")
    
    # Modulation evidence
    lines.append("WHY this modulation:")
    mod_lines = _format_modulation_evidence(chain)
    if mod_lines:
        lines.extend(mod_lines)
    else:
        lines.append("  (evidence from evidence graph)")
    lines.append("")
    
    # Demodulation evidence
    lines.append("WHY this demodulation:")
    demod_lines = _format_demodulation_evidence(chain)
    if demod_lines:
        lines.extend(demod_lines)
    else:
        lines.append("  (demodulation not run or gated)")
    lines.append("")
    
    # Decoding evidence
    lines.append("WHY this decode chain:")
    decode_lines = _format_decoding_evidence(chain)
    if decode_lines:
        lines.extend(decode_lines)
    lines.extend(_format_decoding_results(chain))
    lines.append("")
    
    # SNR context
    if graph:
        snr_items = [e for e in graph.items.values()
                     if e.candidate == "signal" and e.measurement == "snr_db"]
        if snr_items:
            lines.append(f"Operating context: SNR = {snr_items[0].value:.1f} dB")
    
    # Uncertainty
    if chain.score_uncertainty > 0:
        lines.append(f"Uncertainty: std_dev ≈ {chain.score_uncertainty:.3f}")
    
    # Alternatives
    if include_alternatives and chain.alternative_paths:
        lines.append("")
        lines.append("Alternative paths considered:")
        for alt in chain.alternative_paths[:3]:
            lines.append(f"  - {alt}")
    
    return "\n".join(lines)


def generate_short_why(chain: HypothesisChain) -> str:
    """Generate a concise one-line WHY."""
    parts = []
    
    # Modulation reason
    if chain.modulation_score > 0.7:
        parts.append(f"strong {chain.modulation} classifier match ({chain.modulation_score:.2f})")
    elif chain.modulation_score > 0.5:
        parts.append(f"moderate {chain.modulation} classifier support ({chain.modulation_score:.2f})")
    else:
        parts.append(f"weak {chain.modulation} classifier evidence ({chain.modulation_score:.2f})")
    
    # Demod reason
    if chain.demodulation_score > 0.7:
        parts.append(f"clean demod (EVM score {chain.demodulation_score:.2f})")
    elif chain.demodulation_score > 0.4:
        parts.append(f"fair demod (EVM score {chain.demodulation_score:.2f})")
    else:
        parts.append("poor/no demod")
    
    # Decode reason
    if chain.decoding_score > 0.7:
        parts.append("CRC-valid frame")
    elif chain.decoding_score > 0.4:
        parts.append("FEC-corrected")
    else:
        parts.append("no valid frame")
    
    return "; ".join(parts) + f" | composite={chain.composite_score:.3f}"


def generate_comparison_why(
    chain_a: HypothesisChain,
    chain_b: HypothesisChain,
) -> str:
    """Generate a WHY explaining why chain_a ranks above chain_b."""
    from app.hypothesis.scoring import compare_chains
    
    diff = compare_chains(chain_a, chain_b)
    
    lines = [f"Comparison: {chain_a.short_str()} vs {chain_b.short_str()}"]
    lines.append(f"Score difference: {diff['score_diff']:+.3f}")
    lines.append("")
    
    if diff["modulation_diff"] > 0.05:
        lines.append(f"+ {chain_a.modulation} has stronger modulation evidence ({diff['modulation_diff']:+.2f})")
    elif diff["modulation_diff"] < -0.05:
        lines.append(f"- {chain_b.modulation} has stronger modulation evidence ({diff['modulation_diff']:+.2f})")
    
    if diff["demodulation_diff"] > 0.05:
        lines.append(f"+ Better demodulation quality ({diff['demodulation_diff']:+.2f})")
    elif diff["demodulation_diff"] < -0.05:
        lines.append(f"- Worse demodulation quality ({diff['demodulation_diff']:+.2f})")
    
    if diff["decoding_diff"] > 0.05:
        lines.append(f"+ Better decoding result ({diff['decoding_diff']:+.2f})")
    elif diff["decoding_diff"] < -0.05:
        lines.append(f"- Worse decoding result ({diff['decoding_diff']:+.2f})")
    
    if diff["modulation_same"]:
        lines.append("Same modulation family, different chain parameters")
    else:
        lines.append(f"Different modulations: {chain_a.modulation} vs {chain_b.modulation}")
    
    if diff["fec_diff"]:
        lines.append(f"FEC differs: {chain_a.fec.mode} vs {chain_b.fec.mode}")
    if diff["frame_diff"]:
        lines.append(f"Frame differs: {chain_a.frame.frame_name} vs {chain_b.frame.frame_name}")
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_hypothesis_report(
    chains: Sequence[HypothesisChain],
    graph: Optional[EvidenceGraph] = None,
    include_full_why: bool = True,
) -> str:
    """Generate a full text report for ranked hypotheses."""
    lines = ["=" * 70]
    lines.append("HYPOTHESIS ENGINE REPORT")
    lines.append("=" * 70)
    lines.append(f"Total chains evaluated: {len(chains)}")
    lines.append("")
    
    for i, chain in enumerate(chains):
        lines.append(f"--- Rank {i+1} ---")
        lines.append(chain.short_str())
        if include_full_why:
            lines.append(generate_why(chain, graph, include_alternatives=(i < 3)))
        else:
            lines.append(generate_short_why(chain))
        lines.append("")
    
    return "\n".join(lines)


def generate_chain_json(
    chains: Sequence[HypothesisChain],
    graph: Optional[EvidenceGraph] = None,
) -> list[dict]:
    """Generate JSON-serializable chain summaries with WHY."""
    return [
        {
            "rank": i + 1,
            "chain": chain.to_dict(include_evidence=True),
            "why": generate_why(chain, graph, include_alternatives=(i < 3)),
            "why_short": generate_short_why(chain),
        }
        for i, chain in enumerate(chains)
    ]