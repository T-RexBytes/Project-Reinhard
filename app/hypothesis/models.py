"""
app/hypothesis/models.py
-----------------------
Core data structures for the Hypothesis Engine.

A HypothesisChain represents one complete path through the constrained
(Modulation -> Demodulation -> Deinterleaver -> FEC -> Frame) DAG.

The engine consumes an EvidenceGraph and produces ranked HypothesisChains
with composite scores and natural-language WHY justifications.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.evidence.models import EvidenceGraph, EvidenceItem
from app.decoding.decoder import DecodingResult, DecodingSpec, search_decode, make_default_specs


# ---------------------------------------------------------------------------
# Demodulation parameter candidates
# ---------------------------------------------------------------------------

@dataclass
class DemodParams:
    """Demodulation parameters for a chain candidate."""
    modulation: str
    samples_per_symbol: Optional[float] = None
    apply_matched_filter: bool = True
    rolloff_alpha: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "modulation": self.modulation,
            "samples_per_symbol": self.samples_per_symbol,
            "apply_matched_filter": self.apply_matched_filter,
            "rolloff_alpha": self.rolloff_alpha,
        }

    def __str__(self) -> str:
        parts = [self.modulation]
        if self.samples_per_symbol is not None:
            parts.append(f"sps={self.samples_per_symbol:.2f}")
        if self.rolloff_alpha is not None:
            parts.append(f"alpha={self.rolloff_alpha:.2f}")
        return " | ".join(parts)


@dataclass
class DeinterleaverParams:
    """Deinterleaver parameters for a chain candidate."""
    type: str                     # "none" | "block"
    block_n: Optional[int] = None
    block_m: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "block_n": self.block_n,
            "block_m": self.block_m,
        }

    def __str__(self) -> str:
        if self.type == "none":
            return "none"
        return f"block({self.block_n},{self.block_m})"


@dataclass
class FECParams:
    """FEC parameters for a chain candidate."""
    mode: str                     # "none" | "viterbi" | "rs255_223" | "rs255_239" | "viterbi+rs255_223" | "viterbi+rs255_239"
    use_soft_viterbi: bool = True

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "use_soft_viterbi": self.use_soft_viterbi,
        }

    def __str__(self) -> str:
        return self.mode


@dataclass
class FrameParams:
    """Frame synchronization parameters for a chain candidate."""
    frame_name: str               # e.g. "preamble+crc16", "preamble+crc32"
    preamble: list[int]
    payload_lengths: list[int]
    crc_variant: str              # "crc16_ibm" | "crc32_ieee"

    def to_dict(self) -> dict:
        return {
            "frame_name": self.frame_name,
            "preamble_len": len(self.preamble),
            "payload_lengths_count": len(self.payload_lengths),
            "crc_variant": self.crc_variant,
        }

    def __str__(self) -> str:
        return self.frame_name


# ---------------------------------------------------------------------------
# Full hypothesis chain
# ---------------------------------------------------------------------------

@dataclass
class HypothesisChain:
    """
    One complete decode chain hypothesis: Mod -> Demod -> Deint -> FEC -> Frame.
    
    The chain is scored by the HypothesisEngine and carries a natural-language
    WHY justification citing specific evidence from the EvidenceGraph.
    """
    id: str
    modulation: str
    demod_params: DemodParams
    deinterleaver: DeinterleaverParams
    fec: FECParams
    frame: FrameParams

    # Evidence-backed scores (populated after scoring)
    modulation_score: float = 0.0
    demodulation_score: float = 0.0
    decoding_score: float = 0.0
    composite_score: float = 0.0

    # Evidence provenance
    modulation_evidence: list[EvidenceItem] = field(default_factory=list)
    demodulation_evidence: list[EvidenceItem] = field(default_factory=list)
    decoding_evidence: list[EvidenceItem] = field(default_factory=list)
    decoding_results: list[DecodingResult] = field(default_factory=list)

    # WHY explanation
    why: str = ""

    # Uncertainty
    score_uncertainty: float = 0.0          # 0 = certain, 1 = highly uncertain
    alternative_paths: list[str] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)

    def to_dict(self, include_evidence: bool = False) -> dict:
        d = {
            "id": self.id,
            "modulation": self.modulation,
            "demod_params": self.demod_params.to_dict(),
            "deinterleaver": self.deinterleaver.to_dict(),
            "fec": self.fec.to_dict(),
            "frame": self.frame.to_dict(),
            "modulation_score": round(self.modulation_score, 4),
            "demodulation_score": round(self.demodulation_score, 4),
            "decoding_score": round(self.decoding_score, 4),
            "composite_score": round(self.composite_score, 4),
            "why": self.why,
            "score_uncertainty": round(self.score_uncertainty, 4),
            "alternative_paths": self.alternative_paths,
            "num_decoding_results": len(self.decoding_results),
        }
        if include_evidence:
            d["modulation_evidence"] = [e.to_dict() for e in self.modulation_evidence]
            d["demodulation_evidence"] = [e.to_dict() for e in self.demodulation_evidence]
            d["decoding_evidence"] = [e.to_dict() for e in self.decoding_evidence]
            d["decoding_results"] = [r.to_dict() for r in self.decoding_results]
        return d

    def short_str(self) -> str:
        return (f"Chain[{self.id[:8]}]: {self.modulation} -> {self.demod_params} -> "
                f"{self.deinterleaver} -> {self.fec} -> {self.frame} "
                f"| score={self.composite_score:.3f}")


# ---------------------------------------------------------------------------
# Scoring weights (composite chain score)
# ---------------------------------------------------------------------------

@dataclass
class ChainScoringWeights:
    """Weights for the composite chain score."""
    w_modulation: float = 0.25
    w_demodulation: float = 0.25
    w_decoding: float = 0.35
    w_snr_margin: float = 0.15

    def to_dict(self) -> dict:
        return {
            "w_modulation": self.w_modulation,
            "w_demodulation": self.w_demodulation,
            "w_decoding": self.w_decoding,
            "w_snr_margin": self.w_snr_margin,
        }


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

@dataclass
class HypothesisEngineConfig:
    """Configuration for the HypothesisEngine."""
    # Which modulation candidates to expand (top-N from classifier)
    top_modulations: int = 3
    # Demodulation parameter grid
    sps_candidates: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    # Deinterleaver candidates
    deinterleaver_types: list[str] = field(default_factory=lambda: ["none", "block"])
    block_sizes: list[tuple[int, int]] = field(default_factory=lambda: [(4, 8), (8, 16)])
    # FEC modes to try
    fec_modes: list[str] = field(default_factory=lambda: [
        "none", "viterbi", "rs255_223", "rs255_239",
        "viterbi+rs255_223", "viterbi+rs255_239"
    ])
    # Frame candidates
    frame_candidates: list[str] = field(default_factory=lambda: ["preamble+crc16", "preamble+crc32"])
    # Scoring
    scoring_weights: ChainScoringWeights = field(default_factory=ChainScoringWeights)
    # Limits
    max_chains: int = 200
    top_k_output: int = 10
    # Uncertainty
    uncertainty_threshold: float = 0.3
    # Whether to run actual demodulation (vs using evidence graph only)
    run_demodulation: bool = True
    run_decoding: bool = True
    # Decoding: the shared decode-search candidate grid (Bug B). ``None`` ->
    # the canonical ``make_default_specs()`` grid — the SAME grid the standalone
    # leaderboard (/api/decode) searches, so the leaderboard verdict and the
    # engine's per-chain decode evidence share one search space and ranking.
    # Overriding this pins deinterleaver/FEC/frame in one place (HITL/manual).
    decode_spec_grid: Optional[list[DecodingSpec]] = None

    def to_dict(self) -> dict:
        return {
            "top_modulations": self.top_modulations,
            "sps_candidates": self.sps_candidates,
            "deinterleaver_types": self.deinterleaver_types,
            "block_sizes": self.block_sizes,
            "fec_modes": self.fec_modes,
            "frame_candidates": self.frame_candidates,
            "scoring_weights": self.scoring_weights.to_dict(),
            "max_chains": self.max_chains,
            "top_k_output": self.top_k_output,
            "uncertainty_threshold": self.uncertainty_threshold,
            "run_demodulation": self.run_demodulation,
            "run_decoding": self.run_decoding,
            "decode_spec_grid": [s.to_dict() for s in self.decode_spec_grid]
            if self.decode_spec_grid else None,
        }


def make_chain_id() -> str:
    """Generate a unique chain ID."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Default parameter grids
# ---------------------------------------------------------------------------

DEFAULT_DEMOD_PARAMS = {
    "BPSK": DemodParams(modulation="BPSK", samples_per_symbol=2.0),
    "QPSK": DemodParams(modulation="QPSK", samples_per_symbol=2.0),
    "16QAM": DemodParams(modulation="16QAM", samples_per_symbol=2.0),
    "2-FSK": DemodParams(modulation="2-FSK", samples_per_symbol=8.0),
    "4-FSK": DemodParams(modulation="4-FSK", samples_per_symbol=8.0),
    "Noise/CW": DemodParams(modulation="Noise/CW", samples_per_symbol=1.0),
}


def make_demod_params_grid(modulation: str, sps_candidates: list[float]) -> list[DemodParams]:
    """Generate demodulation parameter candidates for a modulation family."""
    base = DEFAULT_DEMOD_PARAMS.get(modulation, DemodParams(modulation=modulation))
    params = []
    for sps in sps_candidates:
        if sps < 1.0:
            continue
        params.append(DemodParams(
            modulation=modulation,
            samples_per_symbol=sps,
            apply_matched_filter=(sps >= 2.0),
            rolloff_alpha=0.35 if sps >= 2.0 else None,
        ))
    return params


def make_deinterleaver_params_grid(types: list[str], block_sizes: list[tuple[int, int]]) -> list[DeinterleaverParams]:
    """Generate deinterleaver parameter candidates."""
    params = []
    if "none" in types:
        params.append(DeinterleaverParams(type="none"))
    if "block" in types:
        for n, m in block_sizes:
            params.append(DeinterleaverParams(type="block", block_n=n, block_m=m))
    return params


def make_fec_params_grid(modes: list[str]) -> list[FECParams]:
    """Generate FEC parameter candidates."""
    return [FECParams(mode=m) for m in modes]


def make_frame_params_grid(names: list[str]) -> list[FrameParams]:
    """Generate frame parameter candidates."""
    from app.decoding.frame import CRC16_IBM, CRC32_IEEE
    
    params = []
    for name in names:
        if name == "preamble+crc16":
            params.append(FrameParams(
                frame_name="preamble+crc16",
                preamble=[0, 1, 1, 1, 1, 1, 1, 0] * 2,
                payload_lengths=[16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64,
                               72, 80, 88, 96, 100, 112, 128, 144, 160, 192, 200, 224, 248],
                crc_variant="crc16_ibm",
            ))
        elif name == "preamble+crc32":
            params.append(FrameParams(
                frame_name="preamble+crc32",
                preamble=[0, 1, 1, 1, 1, 1, 1, 0] * 2,
                payload_lengths=[16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64,
                               72, 80, 88, 96, 100, 112, 128, 144, 160, 192, 200, 224, 248],
                crc_variant="crc32_ieee",
            ))
    return params


# ---------------------------------------------------------------------------
# Uncertainty quantification
# ---------------------------------------------------------------------------

@dataclass
class ScoreUncertainty:
    """Quantified uncertainty for a chain score."""
    score: float
    std_dev: float
    confidence_interval_95: tuple[float, float]
    # Evidence-based uncertainty factors
    evidence_completeness: float        # 0..1 fraction of expected evidence present
    modulation_margin: float            # gap to 2nd-best modulation
    demodulation_confidence: float      # based on EVM, symbol count
    decoding_consistency: float         # agreement among top decode chains

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "std_dev": round(self.std_dev, 4),
            "confidence_interval_95": [round(self.confidence_interval_95[0], 4),
                                        round(self.confidence_interval_95[1], 4)],
            "evidence_completeness": round(self.evidence_completeness, 4),
            "modulation_margin": round(self.modulation_margin, 4),
            "demodulation_confidence": round(self.demodulation_confidence, 4),
            "decoding_consistency": round(self.decoding_consistency, 4),
        }