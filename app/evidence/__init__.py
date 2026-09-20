"""
app/evidence
------------
Evidence Graph: converts every measurement the RF pipeline produces into a
weighted, provenance-linked graph of *evidence items* feeding *candidate
hypotheses*.

The graph is the foundation of the project's explainability story: modulation
candidates, demodulation verification (EVM / sync) and decode-chain outputs
(CRC validity, Viterbi residual errors, RS corrections, preamble correlation)
are all aggregated per candidate so that --- critically --- *downstream
success raises upstream confidence*.

Submodules
    models.py      EvidenceItem, HypothesisNode, SupportEdge, EvidenceGraph
    aggregate.py   per-stage collectors + cross-stage linkage + builder
    serializer.py  JSON round-trip and dependency-free GraphML export
"""

from app.evidence.models import (
    EvidenceItem,
    HypothesisNode,
    SupportEdge,
    EvidenceGraph,
    make_evidence_id,
    make_evidence_id_unique,
    quality_from_ratio,
    clamp01,
)
from app.evidence.aggregate import (
    MODULATION_FAMILIES,
    collect_detection,
    collect_characterization,
    collect_modulation,
    collect_demodulation,
    collect_decoding,
    link_features_to_families,
    build_evidence_graph,
    collect_all,
)
from app.evidence.serializer import (
    graph_to_json,
    graph_from_json,
    save_json,
    load_json,
    graph_to_graphml,
    save_graphml,
    export_evidence_graph,
)

__all__ = [
    # models
    "EvidenceItem", "HypothesisNode", "SupportEdge", "EvidenceGraph",
    "make_evidence_id", "make_evidence_id_unique",
    "quality_from_ratio", "clamp01",
    # aggregate
    "MODULATION_FAMILIES",
    "collect_detection", "collect_characterization", "collect_modulation",
    "collect_demodulation", "collect_decoding", "link_features_to_families",
    "build_evidence_graph", "collect_all",
    # serializer
    "graph_to_json", "graph_from_json", "save_json", "load_json",
    "graph_to_graphml", "save_graphml", "export_evidence_graph",
]