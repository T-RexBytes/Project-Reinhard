"""
app/evidence/aggregate.py
-------------------------
Stage adapters that convert the RF pipeline's public result objects into
`EvidenceItem`s wired into an `EvidenceGraph`.

Each stage below emits items (a derived measurement) and then crosses them to
hypotheses:

  detection        -> "signal" context (SNR, BW, noise floor, segments)
  characterization -> "signal" context (symbol rate, CFO) + discriminating
                      features that are traceability-cross-linked to the
                      modulation families they discriminate
  modulation       -> one hypothesis node per family (BPSK, QPSK, ...) fed by
                      classifier score / confidence / per-feature fits
  demodulation     -> verification items for the family actually demodulated
                      (EVM, symbol count, residual phase) — these RAISE the
                      family score when clean
  decoding         -> one hypothesis node per decode chain (CRC validity,
                      Viterbi residual errors, RS corrections, preamble
                      correlation), with a DAG dependency onto the modulation
                      family it was fed by

The cross-stage wiring implements the project's core property: **downstream
success must influence upstream hypotheses**. A CRC-valid decode chain and a
low-EVM demodulation both contribute to the modulation family's net score.
"""

from __future__ import annotations

import math
from typing import Optional

from app.evidence.models import (  # noqa: E402  (documented below)
    EvidenceGraph,
    EvidenceItem,
    clamp01,
    make_evidence_id,
    make_evidence_id_unique,
    quality_from_ratio,
)

# Modulation families the classifier can produce (hypothesis keys).
MODULATION_FAMILIES = ("BPSK", "QPSK", "16QAM", "2-FSK", "4-FSK", "Noise/CW")

_SIGNAL = "signal"


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------


def _snr_norm(snr_db: float) -> float:
    """Map SNR(-5..30 dB) onto 0..1."""
    return clamp01((snr_db + 5.0) / 35.0)


def _evm_norm(evm_percent: float) -> float:
    """Map EVM (%) onto 0..1 (40% EVM -> 0)."""
    return clamp01(1.0 - evm_percent / 40.0)


# ---------------------------------------------------------------------------
# 1. Detection
# ---------------------------------------------------------------------------


def collect_detection(
    graph: EvidenceGraph,
    snr=None,
    bandwidth=None,
    noise_floor=None,
    segmentation=None,
) -> None:
    """Add signal-context facts produced by the spectral-analysis stage.

    Args (any may be None): the result dataclasses from ``app.detection`` —
    ``SNRResult``, ``BandwidthResult``, ``NoiseFloorResult``,
    ``SegmentationResult``.
    """
    if snr is not None:
        snr_db = float(getattr(snr, "snr_db", 0.0))
        graph.add_item(EvidenceItem(
            id=make_evidence_id(_SIGNAL, "snr_db"),
            stage="detection", candidate=_SIGNAL, measurement="snr_db",
            value=snr_db, value_norm=_snr_norm(snr_db),
            quality=quality_from_ratio(_snr_norm(snr_db)),
            supports=True, weight=0.6, source="snr.estimate_snr_from_psd",
            unit="dB", context={"method": getattr(snr, "method", "")},
        ))
    if bandwidth is not None:
        bw = float(getattr(bandwidth, "bandwidth_hz", 0.0))
        graph.add_item(EvidenceItem(
            id=make_evidence_id(_SIGNAL, "bandwidth_hz"),
            stage="detection", candidate=_SIGNAL, measurement="bandwidth_hz",
            value=bw, value_norm=clamp01(bw / 1e7),
            quality="good" if bw > 0 else "poor",
            supports=True, weight=0.4, source="bandwidth.estimate_bandwidth",
            unit="Hz",
        ))
    if noise_floor is not None:
        nf = float(getattr(noise_floor, "noise_floor_db", 0.0))
        graph.add_item(EvidenceItem(
            id=make_evidence_id(_SIGNAL, "noise_floor_db"),
            stage="detection", candidate=_SIGNAL, measurement="noise_floor_db",
            value=nf, value_norm=clamp01((nf + 120.0) / 100.0),
            quality="good", supports=True, weight=0.3,
            source="noise_floor.estimate_noise_floor", unit="dB",
            context={"method": getattr(noise_floor, "method", "")},
        ))
    if segmentation is not None:
        n_segments = int(getattr(segmentation, "num_segments", 0))
        graph.add_item(EvidenceItem(
            id=make_evidence_id(_SIGNAL, "num_segments"),
            stage="detection", candidate=_SIGNAL, measurement="num_segments",
            value=n_segments, value_norm=clamp01(n_segments / 4.0),
            quality="good" if n_segments >= 1 else "fair",
            supports=True, weight=0.3, source="segmentation.build_segments",
            context={"detection_threshold_db": getattr(
                segmentation, "detection_threshold_db", None)},
        ))


# ---------------------------------------------------------------------------
# 2. Characterization
# ---------------------------------------------------------------------------


def _feature_item(graph: EvidenceGraph, name: str, value: float, ref: float,
                  source: str, weight: float = 0.25) -> EvidenceItem:
    """Add a normalized dimension-less feature as a signal-context item."""
    item = EvidenceItem(
        id=make_evidence_id(_SIGNAL, name),
        stage="characterization", candidate=_SIGNAL, measurement=name,
        value=value, value_norm=clamp01(abs(value) / ref if ref else 0.0),
        quality=quality_from_ratio(clamp01(abs(value) / ref if ref else 0.0)),
        supports=True, weight=weight, source=source,
    )
    graph.add_item(item)
    return item


def collect_characterization(
    graph: EvidenceGraph,
    features=None,
    symbol_rate=None,
    freq_offset=None,
) -> None:
    """Add signal-context items from the characterization stage.

    Args: ``FeatureVector``, ``SymbolRateResult``, ``FreqOffsetResult``.
    """
    if features is not None:
        c42 = float(getattr(features, "C42_norm", 0.0) or 0.0)
        _feature_item(graph, "c42_norm", value=c42, ref=2.0,
                      source="features.C42_norm", weight=0.35)
        c21 = float(getattr(features, "C21_sym", 0.0) or 0.0)
        _feature_item(graph, "c21_sym", value=c21, ref=1.0,
                      source="features.C21_sym", weight=0.3)
        papr = float(getattr(features, "papr_db", 0.0) or 0.0)
        _feature_item(graph, "papr_db", value=papr, ref=9.5,
                      source="features.papr_db", weight=0.2)
        flat = float(getattr(features, "spectral_flatness", 0.0) or 0.0)
        _feature_item(graph, "spectral_flatness", value=flat, ref=1.0,
                      source="features.spectral_flatness", weight=0.25)
        amp_var = float(getattr(features, "amplitude_variance_norm", 0.0) or 0.0)
        _feature_item(graph, "amplitude_variance", value=amp_var, ref=1.0,
                      source="features.amplitude_variance_norm", weight=0.15)
        kurt = float(getattr(features, "kurtosis_amplitude", 0.0) or 0.0)
        _feature_item(graph, "kurtosis_amplitude", value=kurt, ref=2.0,
                      source="features.kurtosis_amplitude", weight=0.15)

    if symbol_rate is not None:
        rsym = float(getattr(symbol_rate, "symbol_rate_hz", 0.0) or 0.0)
        conf = float(getattr(symbol_rate, "confidence", 0.0) or 0.0)
        graph.add_item(EvidenceItem(
            id=make_evidence_id(_SIGNAL, "symbol_rate_hz"),
            stage="characterization", candidate=_SIGNAL,
            measurement="symbol_rate_hz", value=rsym,
            value_norm=clamp01(float(symbol_rate.symbol_rate_hz) / 1e7),
            quality=quality_from_ratio(conf, good=0.6, fair=0.35),
            supports=True, weight=0.5, source="symbol_rate.estimate_symbol_rate",
            unit="Hz", context={"confidence": round(conf, 3),
                                "method": getattr(symbol_rate, "method", "")},
        ))
        graph.add_item(EvidenceItem(
            id=make_evidence_id(_SIGNAL, "symbol_rate_confidence"),
            stage="characterization", candidate=_SIGNAL,
            measurement="symbol_rate_confidence", value=conf, value_norm=conf,
            quality=quality_from_ratio(conf, good=0.6, fair=0.35),
            supports=True, weight=0.3, source="symbol_rate.estimate_symbol_rate",
        ))
    if freq_offset is not None:
        cfo = float(getattr(freq_offset, "offset_hz", 0.0) or 0.0)
        conf = float(getattr(freq_offset, "confidence", 0.0) or 0.0)
        graph.add_item(EvidenceItem(
            id=make_evidence_id(_SIGNAL, "freq_offset_hz"),
            stage="characterization", candidate=_SIGNAL,
            measurement="freq_offset_hz", value=cfo,
            value_norm=clamp01(abs(cfo) / 1e6),
            quality=quality_from_ratio(conf, good=0.6, fair=0.35),
            supports=True, weight=0.4, source="freq_offset.estimate_freq_offset",
            unit="Hz", context={"confidence": round(conf, 3),
                                "method": getattr(freq_offset, "method", "")},
        ))


# Feature-to-hypothesis traceability map. Each entry maps a measured feature to
# the (family, target value, tolerance) triplets it discriminates. The linkage
# weight is derived from proximity to the target, bounded to a small value so
# it complements (never overrides) the classifier's own evidence.
FEATURE_TO_FAMILY = {
    "c42_norm": [("BPSK", 2.0, 0.4), ("QPSK", 0.85, 0.45), ("16QAM", 0.68, 0.4)],
    "c21_sym": [("BPSK", 1.0, 0.3)],
    "papr_db": [("16QAM", 7.5, 2.0)],
    "spectral_flatness": [("Noise/CW", 0.85, 0.15)],
    "amplitude_variance": [("16QAM", 0.5, 0.2)],
}


def link_features_to_families(graph: EvidenceGraph) -> None:
    """Cross-wire discriminating features to the modulation families.

    Uses the ``FEATURE_TO_FAMILY`` map to add small ``supports`` edges from the
    measured feature items to each family they help discriminate, weighted by
    how close the measured value is to the family's theoretical target. This
    provides the provenance-linked feature-to-hypothesis traceability without
    double-counting the classifier's own fit terms.
    """
    for feature_name, entries in FEATURE_TO_FAMILY.items():
        item_id = make_evidence_id(_SIGNAL, feature_name)
        item = graph.items.get(item_id)
        if item is None:
            continue
        for family, target, tolerance in entries:
            closeness = math.exp(-0.5 * ((item.value - target) / tolerance) ** 2)
            weight = 0.08 + 0.12 * clamp01(closeness)
            graph.add_edge(item_id, family, weight=round(weight, 4),
                           kind="trace")


# ---------------------------------------------------------------------------
# 3. Modulation classification
# ---------------------------------------------------------------------------


def collect_modulation(graph: EvidenceGraph, mod_result) -> None:
    """Add one hypothesis node + evidence items per modulation family."""
    if mod_result is None:
        return
    candidates = getattr(mod_result, "candidates", []) or []
    features_summary = getattr(mod_result, "features_summary", {}) or {}

    for cand in candidates:
        mod = cand.modulation
        graph.add_hypothesis(mod, kind="modulation", label=mod)

        graph.add_item(EvidenceItem(
            id=make_evidence_id(mod, "classifier_score"),
            stage="modulation", candidate=mod, measurement="classifier_score",
            value=float(cand.score), value_norm=clamp01(float(cand.score)),
            quality=quality_from_ratio(float(cand.score)),
            supports=True, weight=1.4, source="modulation.classifier.classify",
            context=features_summary,
        ))
        graph.add_item(EvidenceItem(
            id=make_evidence_id(mod, "classifier_confidence"),
            stage="modulation", candidate=mod,
            measurement="classifier_confidence",
            value=float(cand.confidence), value_norm=clamp01(float(cand.confidence)),
            quality=quality_from_ratio(float(cand.confidence)),
            supports=True, weight=0.6, source="modulation.classifier.classify",
            context={"why": cand.why},
        ))
        for fit_key, fit_val in (cand.feature_fit or {}).items():
            graph.add_item(EvidenceItem(
                id=make_evidence_id(mod, f"fit_{fit_key}"),
                stage="modulation", candidate=mod,
                measurement=f"fit_{fit_key}", value=float(fit_val),
                value_norm=clamp01(float(fit_val)),
                quality=quality_from_ratio(float(fit_val)),
                supports=True, weight=0.35,
                source="modulation.classifier.classify",
                context={"why": cand.why[:200]},
            ))

    # Top-candidate margin: large margin strongly supports the winner.
    if len(candidates) >= 2:
        top = candidates[0]
        margin = float(top.score) - float(candidates[1].score)
        graph.add_item(EvidenceItem(
            id=make_evidence_id(top.modulation, "top_margin"),
            stage="modulation", candidate=top.modulation,
            measurement="top_margin", value=margin,
            value_norm=clamp01(margin * 5.0),
            quality=quality_from_ratio(clamp01(margin * 5.0)),
            supports=True, weight=0.4, source="modulation.classifier.classify",
            unit="delta_score",
        ))

    # Wire every modulation-stage item to its family node.
    for mod in MODULATION_FAMILIES:
        if mod not in graph.hypotheses:
            continue
        for item in list(graph.items.values()):
            if item.stage == "modulation" and item.candidate == mod:
                graph.add_edge(item.id, mod, weight=item.weight)


# ---------------------------------------------------------------------------
# 4. Demodulation verification
# ---------------------------------------------------------------------------


def collect_demodulation(graph: EvidenceGraph, demod_result, family: Optional[str] = None) -> None:
    """Add verification items for the modulation actually demodulated.

    ``family`` defaults to ``demod_result.modulation``. Clean, non-gated
    demodulation with low EVM raises the family's score — evidence flowing
    *upstream* from the demodulator to the classifier.
    """
    if demod_result is None:
        return
    family = family or getattr(demod_result, "modulation", None)
    if family is None:
        return
    graph.add_hypothesis(family, kind="modulation", label=family)

    gated = bool(getattr(demod_result, "gated", False))
    evm = getattr(demod_result, "evm", None)

    if gated:
        graph.add_item(EvidenceItem(
            id=make_evidence_id(family, "demod_gated"),
            stage="demodulation", candidate=family, measurement="demod_gated",
            value=1.0, value_norm=1.0, quality="poor", supports=False,
            weight=0.7, source="demodulation.slicer.demodulate_signal",
            context={"why": getattr(demod_result, "why", "")},
        ))
    else:
        if evm is not None:
            rms = float(evm.evm_rms_percent)
            db = float(evm.evm_db)
            graph.add_item(EvidenceItem(
                id=make_evidence_id(family, "evm_rms_percent"),
                stage="demodulation", candidate=family,
                measurement="evm_rms_percent", value=rms,
                value_norm=_evm_norm(rms),
                quality=quality_from_ratio(_evm_norm(rms)),
                supports=True, weight=1.5, source="demodulation.slicer.compute_evm",
                unit="%",
            ))
            graph.add_item(EvidenceItem(
                id=make_evidence_id(family, "evm_db"),
                stage="demodulation", candidate=family,
                measurement="evm_db", value=db,
                value_norm=clamp01(-db / 40.0),
                quality=quality_from_ratio(clamp01(-db / 40.0)),
                supports=True, weight=0.5, source="demodulation.slicer.compute_evm",
                unit="dB",
            ))
        n_sym = int(getattr(demod_result, "num_symbols", 0))
        graph.add_item(EvidenceItem(
            id=make_evidence_id(family, "num_symbols"),
            stage="demodulation", candidate=family, measurement="num_symbols",
            value=n_sym, value_norm=clamp01(n_sym / 1024.0),
            quality=quality_from_ratio(clamp01(n_sym / 1024.0)),
            supports=True, weight=0.35,
            source="demodulation.slicer.demodulate_signal",
        ))
        residual = abs(float(getattr(demod_result, "residual_phase_rad", 0.0) or 0.0))
        norm_phase = clamp01(1.0 - residual / (math.pi / 8.0))
        graph.add_item(EvidenceItem(
            id=make_evidence_id(family, "residual_phase_rad"),
            stage="demodulation", candidate=family,
            measurement="residual_phase_rad", value=residual,
            value_norm=norm_phase,
            quality=quality_from_ratio(norm_phase),
            supports=True, weight=0.4, source="demodulation.synchronizer",
            unit="rad",
        ))

    # Wire demodulation items to the family.
    for item in list(graph.items.values()):
        if item.stage == "demodulation" and item.candidate == family:
            graph.add_edge(item.id, family, weight=item.weight)


# ---------------------------------------------------------------------------
# 5. Decoding chain
# ---------------------------------------------------------------------------


def collect_decoding(
    graph: EvidenceGraph,
    decoding_results,
    family: Optional[str] = None,
) -> None:
    """Add one hypothesis node per decode chain (+ DAG edge to its family).

    ``decoding_results`` is the ranked ``list[DecodingResult]`` from
    ``app.decoding.decoder.search_decode``. ``family`` is the modulation the
    bits were demodulated from (default: take from the metadata of the first
    non-empty result if present).
    """
    results = list(decoding_results or [])
    if not results:
        return

    if family is None and getattr(results[0], "family", None) is not None:
        family = results[0].family

    for res in results:
        spec = res.spec_name or res.chain
        label = res.chain or spec
        graph.add_hypothesis(spec, kind="decode_chain",
                             label=label, parents=[family] if family else [])
        if family:
            graph.add_dependency(spec, family)

        crc = res.crc_valid
        crc_item = EvidenceItem(
            id=make_evidence_id_unique(spec, "crc_valid"),
            stage="decoding", candidate=spec, measurement="crc_valid",
            value=float(bool(crc)), value_norm=1.0 if crc is True else (0.0 if crc is False else 0.5),
            quality="good" if crc is True else ("poor" if crc is False else "fair"),
            supports=(crc is True), weight=2.0,
            source="decoding.frame.frame_sync", family=family,
            context={"fec": res.fec, "used_soft": bool(res.used_soft)},
        )
        graph.add_item(crc_item)

        ok_item = EvidenceItem(
            id=make_evidence_id_unique(spec, "decode_ok"),
            stage="decoding", candidate=spec, measurement="decode_ok",
            value=float(bool(res.ok)),
            value_norm=1.0 if res.ok else 0.15,
            quality="good" if res.ok else "poor",
            supports=bool(res.ok), weight=1.0,
            source="decoding.decoder.decode_bits", family=family,
            context={"why": res.why[:300]},
        )
        graph.add_item(ok_item)

        n_verr = int(res.num_viterbi_errors or 0)
        verr_norm = clamp01(1.0 - n_verr / 32.0)
        graph.add_item(EvidenceItem(
            id=make_evidence_id_unique(spec, "viterbi_residual_errors"),
            stage="decoding", candidate=spec,
            measurement="viterbi_residual_errors", value=n_verr,
            value_norm=verr_norm,
            quality=quality_from_ratio(verr_norm),
            supports=(n_verr == 0), weight=0.8,
            source="decoding.convolutional.decode_viterbi", family=family,
            unit="errors",
        ))

        n_rs = int(res.num_rs_symbols_corrected or 0)
        rs_norm = clamp01(n_rs / 16.0)
        graph.add_item(EvidenceItem(
            id=make_evidence_id_unique(spec, "rs_symbols_corrected"),
            stage="decoding", candidate=spec,
            measurement="rs_symbols_corrected", value=n_rs,
            value_norm=rs_norm, quality=quality_from_ratio(rs_norm),
            supports=True, weight=0.5,
            source="decoding.reed_solomon.rs_decode", family=family,
            unit="symbols",
        ))

        corr = float(res.preamble_correlation or 0.0)
        graph.add_item(EvidenceItem(
            id=make_evidence_id_unique(spec, "preamble_correlation"),
            stage="decoding", candidate=spec,
            measurement="preamble_correlation", value=corr,
            value_norm=clamp01(corr), quality=quality_from_ratio(corr),
            supports=True, weight=0.6,
            source="decoding.frame.correlate_preamble", family=family,
        ))

        frame_offset = res.frame_offset
        if frame_offset is not None:
            graph.add_item(EvidenceItem(
                id=make_evidence_id_unique(spec, "frame_offset"),
                stage="decoding", candidate=spec, measurement="frame_offset",
                value=float(frame_offset), value_norm=1.0,
                quality="good", supports=True, weight=0.2,
                source="decoding.frame.frame_sync", family=family,
                unit="bits",
            ))

        # Wire the chain's items to the chain node.
        for item in list(graph.items.values()):
            if item.stage == "decoding" and item.candidate == spec:
                graph.add_edge(item.id, spec, weight=item.weight)

    # -- Downstream feedback to the modulation family -----------------------
    # Critical property: successful decoding RAISES confidence in the upstream
    # modulation hypothesis that produced the bitstream. We emit one aggregated
    # feedback item per family from the best-ranked chain (rank 0), so a
    # CRC-valid frame nudges the family's net score upward.
    best = results[0]
    if family and family in graph.hypotheses:
        if best.crc_valid is not None:
            crc_ok = bool(best.crc_valid)
            graph.add_item(EvidenceItem(
                id=make_evidence_id_unique(family, "chain_crc_valid"),
                stage="decoding", candidate=family, measurement="chain_crc_valid",
                value=float(crc_ok),
                value_norm=1.0 if crc_ok else 0.0,
                quality="good" if crc_ok else "poor",
                supports=True, weight=1.2 if crc_ok else 0.5,
                source="decoding.frame.frame_sync",
                context={"chain": best.spec_name,
                         "spec": best.spec_name,
                         "why": best.why[:200]},
            ))
        graph.add_item(EvidenceItem(
            id=make_evidence_id_unique(family, "chain_ranked_ok"),
            stage="decoding", candidate=family, measurement="chain_ranked_ok",
            value=float(bool(best.ok)),
            value_norm=1.0 if best.ok else 0.15,
            quality="good" if best.ok else "poor",
            supports=bool(best.ok), weight=0.5,
            source="decoding.decoder.search_decode",
            context={"chain": best.spec_name, "rank": 1},
        ))
        for item in graph.items.values():
            if item.stage == "decoding" and item.candidate == family:
                graph.add_edge(item.id, family, weight=item.weight)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_evidence_graph(
    snr=None,
    bandwidth=None,
    noise_floor=None,
    segmentation=None,
    features=None,
    symbol_rate=None,
    freq_offset=None,
    mod_classification=None,
    demodulation=None,
    decoding=None,
    demod_modulation: Optional[str] = None,
    metadata: Optional[dict] = None,
    auto_score: bool = True,
) -> EvidenceGraph:
    """Collect all provided stage results into a scored evidence graph.

    Each parameter is the public result object of the corresponding pipeline
    stage (or None to skip it). ``demod_modulation`` overrides the family
    attributed to the demodulation/decoding evidence when it cannot be read
    from the result objects themselves.

    Returns an ``EvidenceGraph`` whose hypotheses are scored (unless
    ``auto_score=False``) and whose ``top_hypotheses()`` yields the ranked
    leaderboard for the GUI / report.
    """
    graph = EvidenceGraph(name="rf_hypothesis", metadata=metadata or {})

    collect_detection(graph, snr, bandwidth, noise_floor, segmentation)
    collect_characterization(graph, features, symbol_rate, freq_offset)

    # Register the full modulation-family leaderboard early (neutral nodes) so
    # downstream demodulation / decoding feedback always has a target node.
    for fam in MODULATION_FAMILIES:
        graph.add_hypothesis(fam, kind="modulation", label=fam)

    mod_family = None
    if demod_modulation is not None:
        mod_family = demod_modulation
    elif demodulation is not None:
        mod_family = demodulation.modulation
    elif decoding and hasattr(decoding[0], "family") and decoding[0].family:
        mod_family = decoding[0].family

    collect_modulation(graph, mod_classification)
    collect_demodulation(graph, demodulation, family=mod_family)
    collect_decoding(graph, decoding, family=mod_family)

    # Cross-stage linkage: discriminative features -> modulation families, and
    # downstream verification items -> the families that drove them.
    link_features_to_families(graph)
    if mod_family and mod_family in graph.hypotheses:
        for h in list(graph.hypotheses.values()):
            if h.kind == "decode_chain" and h.parents == []:
                graph.add_dependency(h.name, mod_family)

    if auto_score:
        graph.compute_scores()
    return graph


# Alias retained for a compact public API.
collect_all = build_evidence_graph