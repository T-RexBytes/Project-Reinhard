"""
app/hypothesis
--------------
Hypothesis Engine: DAG search over valid decode chains with composite scoring
and explainable WHY justifications.

The engine consumes an EvidenceGraph and produces ranked HypothesisChains
representing complete (Modulation -> Demodulation -> Deinterleaver -> FEC -> Frame)
paths with calibrated scores and natural-language explanations.
"""

from __future__ import annotations

from app.hypothesis.models import (
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
)

from app.hypothesis.scoring import (
    score_modulation_from_evidence,
    score_demodulation_from_evidence,
    score_decoding_from_evidence,
    score_modulation_from_classifier,
    score_demodulation_from_result,
    score_decoding_from_results,
    score_snr_margin,
    compute_composite_score,
    quantify_uncertainty,
    rank_chains,
    compare_chains,
)

from app.hypothesis.search import (
    enumerate_chain_candidates,
    evaluate_chain_live,
    run_hypothesis_search,
    hypothesize,
)

from app.hypothesis.explain import (
    generate_why,
    generate_short_why,
    generate_comparison_why,
    generate_hypothesis_report,
    generate_chain_json,
)

__all__ = [
    # Models
    "HypothesisChain",
    "HypothesisEngineConfig",
    "ChainScoringWeights",
    "DemodParams",
    "DeinterleaverParams",
    "FECParams",
    "FrameParams",
    "ScoreUncertainty",
    "make_chain_id",
    "make_demod_params_grid",
    "make_deinterleaver_params_grid",
    "make_fec_params_grid",
    "make_frame_params_grid",
    "DEFAULT_DEMOD_PARAMS",
    # Scoring
    "score_modulation_from_evidence",
    "score_demodulation_from_evidence",
    "score_decoding_from_evidence",
    "score_modulation_from_classifier",
    "score_demodulation_from_result",
    "score_decoding_from_results",
    "score_snr_margin",
    "compute_composite_score",
    "quantify_uncertainty",
    "rank_chains",
    "compare_chains",
    # Search
    "enumerate_chain_candidates",
    "evaluate_chain_live",
    "run_hypothesis_search",
    "hypothesize",
    # Explain
    "generate_why",
    "generate_short_why",
    "generate_comparison_why",
    "generate_hypothesis_report",
    "generate_chain_json",
]