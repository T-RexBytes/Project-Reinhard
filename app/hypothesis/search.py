"""
app/hypothesis/search.py
------------------------
DAG search over valid (Mod -> Demod -> Deint -> FEC -> Frame) chains.

The search enumerates a bounded candidate grid constrained by the project's
philosophy: no open-ended brute force, only physically plausible combinations.

To keep the search tractable, chains with identical demodulation parameters
(modulation / samples-per-symbol / matched-filter) are *grouped*: demodulation
and the group's decode-chain grid run once, and every chain in the group
reuses the shared results. This turns an O(chains) DSP cost into a cost
proportional to the number of unique demodulation settings, with the
per-chain scoring staying O(chains).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

from app.evidence.models import EvidenceGraph
from app.decoding.decoder import (
    DecodingSpec,
    DecodingResult,
    search_decode,
    make_default_specs,
    none_spec,
    block_spec,
    FrameDescriptor,
    CRC16_IBM,
    CRC32_IEEE,
)
from app.demodulation.slicer import demodulate_signal
from app.models import ComplexSignal
from app.hypothesis.models import (
    HypothesisChain,
    HypothesisEngineConfig,
    DemodParams,
    DeinterleaverParams,
    FECParams,
    FrameParams,
    make_chain_id,
    make_demod_params_grid,
    make_deinterleaver_params_grid,
    make_fec_params_grid,
    make_frame_params_grid,
)


# ---------------------------------------------------------------------------
# Estimated samples-per-symbol seeding (max-safe reuse of Module 2/3 output)
# ---------------------------------------------------------------------------

def _signal_sps_from_graph(
    graph: Optional[EvidenceGraph],
    signal: ComplexSignal,
) -> Optional[float]:
    """Read the symbol-rate the main pipeline already estimated (if any).

    `build_evidence_graph` records ``symbol_rate_hz`` + ``symbol_rate_confidence``
    as signal-level evidence items. When they are present and confident, the
    hypothesis engine reuses that estimate instead of duplicating the DSP. This
    is the "demod must consume the same sps the main demod stage produced"
    invariant.
    """
    if graph is None:
        return None
    rsym_items = [e for e in graph.items.values()
                  if e.candidate == "signal" and e.measurement == "symbol_rate_hz"]
    if not rsym_items:
        return None
    conf_items = [e for e in graph.items.values()
                  if e.candidate == "signal" and e.measurement == "symbol_rate_confidence"]
    rsym = float(rsym_items[0].value)
    conf = float(conf_items[0].value) if conf_items else 0.0
    sr = signal.metadata.sample_rate or 1.0
    if rsym > 0 and conf > 0.3 and sr > 0:
        sps = sr / rsym
        if 1.0 <= sps <= 64.0:
            return float(sps)
    return None


def _signal_sps_estimate(signal: ComplexSignal) -> Optional[float]:
    """Fallback direct symbol-rate estimate when the graph carries no hint.

    Mirrors the auto-estimation path inside ``demodulate_signal`` so the
    seeded working point matches what the main demodulation stage would use.
    """
    from app.characterization.symbol_rate import estimate_symbol_rate
    from app.detection.fft import compute_psd
    from app.detection.bandwidth import estimate_bandwidth

    sr = signal.metadata.sample_rate or 1.0
    n_data = len(signal.data)
    bw_hint = None
    try:
        fft_n = min(1024, n_data)
        if fft_n >= 64:
            psd = compute_psd(signal, fft_size=fft_n)
            bw_res = estimate_bandwidth(psd, method="3db")
            bw_hint = bw_res.bandwidth_hz
    except Exception:
        pass
    try:
        sym = estimate_symbol_rate(signal.data[:32768], sample_rate=sr, bandwidth_hz=bw_hint)
    except Exception:
        return None
    if sym.confidence > 0.3 and sym.symbol_rate_hz > 0 and sr > 0:
        sps = sr / sym.symbol_rate_hz
        if 1.0 <= sps <= 64.0:
            return float(sps)
    return None


def _seed_sps_candidates(
    signal: ComplexSignal,
    graph: Optional[EvidenceGraph],
    sps_candidates: list[float],
) -> list[float]:
    """Augment the SPS grid with the estimated symbol-clock working point.

    The demodulation candidate grid is normally a fixed table (e.g.
    ``[1.0, 2.0, 4.0]``). Without the estimated sps in the grid, the group
    evaluation demods at the nearest table entry — ``1.0`` for anything slower
    than 2 samples/symbol — which treats every raw sample as a symbol and
    poisons the EVM / symbol-count evidence fed into the composite score.

    The estimated working point is seeded *first* so it is always evaluated
    and never cut off by ``max_chains``; a fallback default grid entry is
    retained only when no trustworthy estimate exists.
    """
    est = _signal_sps_from_graph(graph, signal)
    if est is None:
        est = _signal_sps_estimate(signal)
    if est is None:
        return list(sps_candidates)
    grid = [s for s in sps_candidates if abs(s - est) > 1e-6]
    return [est] + grid


# ---------------------------------------------------------------------------
# Chain candidate enumeration
# ---------------------------------------------------------------------------

def enumerate_chain_candidates(
    config: HypothesisEngineConfig,
    modulation_candidates: list[str],
    sps_candidates: Optional[list[float]] = None,
) -> list[HypothesisChain]:
    """
    Enumerate all valid chain candidates for the given modulation candidates.

    The search space is bounded by:
    - Top-N modulation candidates from the evidence graph / classifier
    - A discrete SPS grid per modulation (estimated working point first when
      available via ``_seed_sps_candidates``)
    - None + small block deinterleaver sizes
    - A small FEC mode table
    - Frame candidates (CRC-16, CRC-32)
    """
    chains: list[HypothesisChain] = []
    sps_grid = list(sps_candidates) if sps_candidates is not None else config.sps_candidates

    for modulation in modulation_candidates[: config.top_modulations]:
        # Non-digital classifications cannot be demodulated / decoded; they
        # yield a single placeholder chain (the search still surfaces it as a
        # ranked alternative).
        if modulation in ("Noise/CW",):
            chains.append(HypothesisChain(
                id=make_chain_id(),
                modulation=modulation,
                demod_params=DemodParams(modulation=modulation),
                deinterleaver=DeinterleaverParams(type="none"),
                fec=FECParams(mode="none"),
                frame=FrameParams(frame_name="none", preamble=[],
                                  payload_lengths=[], crc_variant="none"),
            ))
            continue

        demod_grid = make_demod_params_grid(modulation, sps_grid)
        deint_grid = make_deinterleaver_params_grid(config.deinterleaver_types,
                                                    config.block_sizes)
        fec_grid = make_fec_params_grid(config.fec_modes)
        frame_grid = make_frame_params_grid(config.frame_candidates)

        for demod_p, deint_p, fec_p, frame_p in itertools.product(
            demod_grid, deint_grid, fec_grid, frame_grid
        ):
            if not _is_valid_combination(modulation, demod_p, deint_p, fec_p, frame_p):
                continue

            chains.append(HypothesisChain(
                id=make_chain_id(),
                modulation=modulation,
                demod_params=demod_p,
                deinterleaver=deint_p,
                fec=fec_p,
                frame=frame_p,
            ))
            if len(chains) >= config.max_chains:
                return chains

    return chains


def _is_valid_combination(
    modulation: str,
    demod: DemodParams,
    deint: DeinterleaverParams,
    fec: FECParams,
    frame: FrameParams,
) -> bool:
    """Reject chain combinations that are physically implausible."""
    if "FSK" in modulation.upper():
        # The FSK path never applies a matched filter and needs a usable sps.
        if demod.samples_per_symbol is not None and demod.samples_per_symbol < 3.0:
            return False
        if deint.type != "none":
            return False
    # A "none" frame parser cannot validate a real FEC payload.
    if frame.frame_name == "none" and fec.mode != "none":
        return False
    return True


# ---------------------------------------------------------------------------
# Evidence-only scoring (no live DSP)
# ---------------------------------------------------------------------------

def score_chain_from_evidence(
    chain: HypothesisChain,
    graph: EvidenceGraph,
    snr_db: Optional[float] = None,
) -> None:
    """Score a chain using only the EvidenceGraph (no live demod/decoding)."""
    from app.hypothesis.scoring import (
        score_modulation_from_evidence,
        score_demodulation_from_evidence,
        score_decoding_from_evidence,
        compute_composite_score,
    )

    mod = chain.modulation
    chain.modulation_score, chain.modulation_evidence = score_modulation_from_evidence(graph, mod)
    chain.demodulation_score, chain.demodulation_evidence = score_demodulation_from_evidence(graph, mod)

    decode_candidates = [h for h in graph.hypotheses.values()
                         if h.kind == "decode_chain" and (not h.parents or mod in h.parents)]
    if decode_candidates:
        best = max(decode_candidates, key=lambda h: h.score)
        chain.decoding_score, chain.decoding_evidence = score_decoding_from_evidence(graph, best.name)
    else:
        chain.decoding_score = 0.0
        chain.decoding_evidence = []

    chain.composite_score = compute_composite_score(chain, graph=graph, snr_db=snr_db)


# ---------------------------------------------------------------------------
# Chain -> DecodingSpec
# ---------------------------------------------------------------------------

def _chain_to_decoding_spec(chain: HypothesisChain) -> DecodingSpec:
    """Convert a HypothesisChain to a DecodingSpec for the decoder."""
    if chain.deinterleaver.type == "block":
        deint = block_spec(chain.deinterleaver.block_n, chain.deinterleaver.block_m)
    else:
        deint = none_spec()

    if chain.frame.frame_name == "preamble+crc16":
        frame = FrameDescriptor(
            name="preamble+crc16",
            preamble=chain.frame.preamble or ([0, 1, 1, 1, 1, 1, 1, 0] * 2),
            header_bits=0,
            payload_symbols=(chain.frame.payload_lengths or [100])[0],
            crc_variant=CRC16_IBM,
        )
    elif chain.frame.frame_name == "preamble+crc32":
        frame = FrameDescriptor(
            name="preamble+crc32",
            preamble=chain.frame.preamble or ([0, 1, 1, 1, 1, 1, 1, 0] * 2),
            header_bits=0,
            payload_symbols=(chain.frame.payload_lengths or [100])[0],
            crc_variant=CRC32_IEEE,
        )
    else:
        frame = None

    fec_mode = chain.fec.mode
    return DecodingSpec(
        name=f"{fec_mode}|{chain.deinterleaver}|{chain.frame.frame_name}",
        deinterleaver=deint,
        fec=fec_mode,
        frame_descriptor=frame,
        apply_viterbi="viterbi" in fec_mode,
    )


# ---------------------------------------------------------------------------
# Live group evaluation (shared demodulation + decode grid per (mod, sps))
# ---------------------------------------------------------------------------

def _group_key(chain: HypothesisChain) -> tuple:
    dem = chain.demod_params
    return (chain.modulation, dem.samples_per_symbol, bool(dem.apply_matched_filter),
            bool(chain.fec.use_soft_viterbi))


def _evaluate_group(
    group_chains: list[HypothesisChain],
    signal: ComplexSignal,
    config: HypothesisEngineConfig,
    graph: Optional[EvidenceGraph],
) -> dict:
    """Run demodulation once, then the group's decode-chain grid once.

    Returns {chain_id: {demod_result, decoding_results}} and also writes per
    chain the modulation / demodulation / decoding / composite scores.
    """
    from app.hypothesis.scoring import (
        score_modulation_from_classifier,
        score_demodulation_from_result,
        score_decoding_from_results,
        compute_composite_score,
    )

    if not group_chains:
        return {}

    snr_db = None
    if graph:
        snr_items = [e for e in graph.items.values()
                     if e.candidate == "signal" and e.measurement == "snr_db"]
        if snr_items:
            snr_db = snr_items[0].value

    rep = group_chains[0].demod_params
    is_digital = group_chains[0].modulation not in ("Noise/CW",)

    # ---- 1. Demodulate once for the whole group ---------------------------
    demod_result = None
    if config.run_demodulation and is_digital:
        demod_result = demodulate_signal(
            signal,
            modulation=rep.modulation,
            samples_per_symbol=rep.samples_per_symbol,
            max_symbols=4096,
        )

    # ---- 2. Decode grid once for the group --------------------------------
    # Bug B: derive this group's decode search FROM the single shared grid the
    # standalone leaderboard (/api/decode, `search_decode` + `make_default_specs`)
    # searches. Only the combos this group's chains enumerate are probed — a
    # subset of the shared grid — keeping per-group DSP bounded while
    # guaranteeing per-combo results are IDENTICAL to the leaderboard's for the
    # same demoded bits (same `search_decode`, same spec naming, deterministic
    # `decode_bits`). A caller-pinned `decode_spec_grid` is used verbatim.
    group_results: list[DecodingResult] = []
    by_spec: dict[str, DecodingResult] = {}
    if (config.run_decoding and demod_result is not None
            and demod_result.bits and not demod_result.gated):
        shared_specs = config.decode_spec_grid or make_default_specs()
        wanted = {_chain_to_decoding_spec(c).name for c in group_chains}
        group_specs = [sp for sp in shared_specs if sp.name in wanted] or list(shared_specs)
        llrs = demod_result.llrs_sample or None
        group_results = search_decode(
            demod_result.bits,
            specs=group_specs,
            llrs=llrs,
            use_soft_viterbi=group_chains[0].fec.use_soft_viterbi,
            top_k=len(group_specs),
        )
        by_spec = {r.spec_name: r for r in group_results}

    # ---- 3. Per-chain scoring (cheap) ------------------------------------
    out = {}
    for chain in group_chains:
        if graph is not None:
            chain.modulation_score, chain.modulation_evidence = score_modulation_from_classifier(
                graph, chain.modulation)
        else:
            chain.modulation_score = 0.5
            chain.modulation_evidence = []

        if demod_result is not None:
            chain.demodulation_score, chain.demodulation_evidence = score_demodulation_from_result(
                demod_result)
        else:
            chain.demodulation_score = 0.0
            chain.demodulation_evidence = []

        spec = _chain_to_decoding_spec(chain)
        if spec.name in by_spec:
            chain.decoding_results = [by_spec[spec.name]]
        else:
            chain.decoding_results = list(group_results)
        if group_results:
            chain.decoding_score, chain.decoding_evidence = score_decoding_from_results(
                chain.decoding_results)
        else:
            chain.decoding_score = 0.0
            chain.decoding_evidence = []

        chain.composite_score = compute_composite_score(chain, graph=graph, snr_db=snr_db)
        out[chain.id] = {
            "demod_result": demod_result,
            "decoding_results": group_results,
        }

    return out


# ---------------------------------------------------------------------------
# Per-chain live evaluation (public, single-chain wrapper)
# ---------------------------------------------------------------------------

def evaluate_chain_live(
    chain: HypothesisChain,
    signal: ComplexSignal,
    config: HypothesisEngineConfig,
    graph: Optional[EvidenceGraph] = None,
) -> list[DecodingResult]:
    """Evaluate one chain live. Prefer ``run_hypothesis_search`` for all chains.

    Provided for isolated per-chain probing; the full search groups chains by
    demodulation setting to share the expensive DSP work.
    """
    from app.hypothesis.scoring import (
        score_modulation_from_classifier,
        score_demodulation_from_result,
        score_decoding_from_results,
        compute_composite_score,
    )

    snr_db = None
    if graph:
        snr_items = [e for e in graph.items.values()
                     if e.candidate == "signal" and e.measurement == "snr_db"]
        if snr_items:
            snr_db = snr_items[0].value

    if graph is not None:
        chain.modulation_score, chain.modulation_evidence = score_modulation_from_classifier(
            graph, chain.modulation)
    else:
        chain.modulation_score = 0.5

    demod_result = None
    if config.run_demodulation and chain.modulation not in ("Noise/CW",):
        demod_result = demodulate_signal(
            signal,
            modulation=chain.demod_params.modulation,
            samples_per_symbol=chain.demod_params.samples_per_symbol,
            max_symbols=4096,
        )
        chain.demodulation_score, chain.demodulation_evidence = score_demodulation_from_result(demod_result)
    else:
        chain.demodulation_score = 0.0
        chain.demodulation_evidence = []

    decoding_results: list[DecodingResult] = []
    if (config.run_decoding and demod_result is not None
            and demod_result.bits and not demod_result.gated):
        spec = _chain_to_decoding_spec(chain)
        # Bug B: same shared grid as the standalone leaderboard. Probe only the
        # chain's own combo (subset of the grid), attach its ranked result.
        shared_specs = config.decode_spec_grid or make_default_specs()
        live_specs = [sp for sp in shared_specs if sp.name == spec.name] or list(shared_specs)
        decoding_results = search_decode(
            demod_result.bits,
            specs=live_specs,
            llrs=demod_result.llrs_sample or None,
            use_soft_viterbi=chain.fec.use_soft_viterbi,
            top_k=len(live_specs),
        )
        own = next((r for r in decoding_results if r.spec_name == spec.name), None)
        chain.decoding_results = [own] if own else decoding_results
        chain.decoding_score, chain.decoding_evidence = score_decoding_from_results(chain.decoding_results)
    else:
        chain.decoding_score = 0.0
        chain.decoding_evidence = []

    chain.composite_score = compute_composite_score(chain, graph=graph, snr_db=snr_db)
    return decoding_results


# ---------------------------------------------------------------------------
# Full hypothesis search
# ---------------------------------------------------------------------------

def run_hypothesis_search(
    signal: ComplexSignal,
    evidence_graph: EvidenceGraph,
    config: Optional[HypothesisEngineConfig] = None,
    modulation_candidates: Optional[list[str]] = None,
) -> list[HypothesisChain]:
    """
    Run the full hypothesis search over the constrained chain DAG.

    Chains whose demodulation settings agree share one demodulation pass and one
    decode-grid pass; per-chain scoring then reuses those shared results. This
    keeps the cost bounded even though the candidate grid is large.

    Returns the ranked top-K chains (best composite score first).
    """
    from app.hypothesis.scoring import rank_chains, quantify_uncertainty

    if config is None:
        config = HypothesisEngineConfig()

    if modulation_candidates is None:
        mod_hypotheses = [h for h in evidence_graph.hypotheses.values()
                          if h.kind == "modulation"]
        mod_hypotheses.sort(key=lambda h: h.score, reverse=True)
        modulation_candidates = [h.name for h in mod_hypotheses]

    chains = enumerate_chain_candidates(config, modulation_candidates,
                                        sps_candidates=_seed_sps_candidates(
                                            signal, evidence_graph,
                                            config.sps_candidates)
                                        if (config.run_demodulation or config.run_decoding)
                                        else None)

    if config.run_demodulation or config.run_decoding:
        # Group chains by demodulation setting so DSP runs once per group.
        groups: dict[tuple, list[HypothesisChain]] = {}
        for chain in chains:
            groups.setdefault(_group_key(chain), []).append(chain)
        for group in groups.values():
            _evaluate_group(group, signal, config, evidence_graph)
    else:
        for chain in chains:
            score_chain_from_evidence(chain, evidence_graph)

    # Uncertainty quantification + alternative paths.
    for chain in chains:
        uncertainty = quantify_uncertainty(chain, evidence_graph, chains)
        chain.score_uncertainty = uncertainty.std_dev
        chain.alternative_paths = _find_alternative_paths(chain, chains)

    return rank_chains(chains, config.top_k_output)


def _find_alternative_paths(
    chain: HypothesisChain,
    all_chains: list[HypothesisChain],
) -> list[str]:
    """Nearby chains (same modulation or near-equal score) worth comparing."""
    alternatives = []
    for other in all_chains:
        if other.id == chain.id:
            continue
        if other.modulation == chain.modulation:
            if abs(other.composite_score - chain.composite_score) < 0.15:
                alternatives.append(other.short_str())
        elif abs(other.composite_score - chain.composite_score) < 0.05:
            alternatives.append(other.short_str())
        if len(alternatives) >= 3:
            break
    return alternatives


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def hypothesize(
    signal: ComplexSignal,
    evidence_graph: EvidenceGraph,
    top_k: int = 5,
) -> list[HypothesisChain]:
    """Convenience wrapper: default config, ranked top-K hypotheses."""
    config = HypothesisEngineConfig(top_k_output=top_k)
    return run_hypothesis_search(signal, evidence_graph, config)