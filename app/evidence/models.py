"""
app/evidence/models.py
----------------------
Core data structures for the Evidence Graph.

Every measurement the RF pipeline produces becomes an `EvidenceItem`. Items
are gathered per candidate (a modulation family such as ``"QPSK"`` or a decode
chain such as ``"viterbi+rs255_223|block(4,8)|preamble+crc32"``) into a
`HypothesisNode`, and weighted `SupportEdge`s wire the items to the hypotheses
they speak about.

Design intent
    The graph deliberately lets *downstream* stages (demodulation, decoding)
    contribute evidence to the modulation families they verified. This is the
    "downstream success must influence upstream hypotheses" property that
    distinguishes the project from a mere chain of separate classifiers:
    a CRC-valid decode chain raises the score of the very modulation that fed
    it, because the chain's evidence items carry ``candidate == family``.

The graph is dependency-free (pure stdlib) and serializes to JSON / GraphML.
"""

from __future__ import annotations

import time
import math
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Quality / value helpers
# ---------------------------------------------------------------------------


def quality_from_ratio(value: float, good: float = 0.7, fair: float = 0.45) -> str:
    """Map a 0..1 ratio to a 'good' / 'fair' / 'poor' quality tier."""
    if value >= good:
        return "good"
    if value >= fair:
        return "fair"
    return "poor"


def clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


# ---------------------------------------------------------------------------
# Core evidence item
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """One atomic, provenance-linked measurement about the signal.

    ``candidate`` is the hypothesis key the item speaks about:
      - ``"signal"``            -> signal-context fact (detection/characterization)
      - ``"QPSK"`` etc.         -> modulation family
      - decode-chain spec name  -> full chain hypothesis

    ``supports`` is the polarity: True means the measurement *supports* the
    candidate (e.g. low EVM for QPSK), False means it *contradicts* it
    (e.g. EVM gated, classifier near zero). ``value_norm`` scales the raw
    ``value`` onto 0..1 (1 = fullest support / contradiction amplitude).
    """

    id: str
    stage: str                 # detection | characterization | modulation | demodulation | decoding
    candidate: str             # hypothesis key (see docstring)
    measurement: str           # e.g. "classifier_score", "evm_rms_percent"
    value: float = 0.0
    value_norm: float = 1.0    # 0..1 normalized amplitude of the measurement
    quality: str = "fair"
    supports: bool = True
    weight: float = 1.0        # significance of this item vs its siblings
    source: str = ""           # module that produced it
    provenance: str = "signal_inferred"  # metadata_provided | signal_inferred | synthetic_ground_truth | user_override
    family: Optional[str] = None  # modulation family for decode-chain items
    context: dict = field(default_factory=dict)
    unit: str = ""
    timestamp: float = field(default_factory=time.time)

    def contribution(self) -> float:
        """Signed, normalized, weighted contribution to its candidate."""
        sign = 1.0 if self.supports else -1.0
        return sign * self.value_norm * self.weight

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "stage": self.stage,
            "candidate": self.candidate,
            "measurement": self.measurement,
            "value": round(float(self.value), 5),
            "value_norm": round(float(self.value_norm), 4),
            "quality": self.quality,
            "supports": bool(self.supports),
            "weight": round(float(self.weight), 4),
            "source": self.source,
            "provenance": self.provenance,
            "family": self.family,
            "unit": self.unit,
            "context": self.context,
            "timestamp": self.timestamp,
        }


def make_evidence_id(candidate: str, measurement: str) -> str:
    """Deterministic, readable evidence id: ``<candidate>:<measurement>``."""
    return f"{candidate}:{measurement}"


def make_evidence_id_unique(candidate: str, measurement: str) -> str:
    """Unique evidence id for measurements that can appear several times."""
    return f"{candidate}:{measurement}:{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Hypothesis node
# ---------------------------------------------------------------------------


@dataclass
class HypothesisNode:
    """A candidate hypothesis (modulation family or decode chain).

    ``parents`` holds the DAG lineage — for a decode chain this is the
    ``(modFamilies)`` node it depends on. ``score`` is only meaningful after
    ``EvidenceGraph.compute_scores()``.
    """

    name: str
    kind: str                 # "signal" | "modulation" | "decode_chain"
    label: str = ""
    parents: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    support_weight: float = 0.0
    contra_weight: float = 0.0
    score: float = 0.5        # neutral when no evidence
    contradiction_ratio: float = 0.0
    quality: str = "poor"
    num_items: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "label": self.label,
            "parents": self.parents,
            "score": round(float(self.score), 4),
            "support_weight": round(float(self.support_weight), 4),
            "contra_weight": round(float(self.contra_weight), 4),
            "contradiction_ratio": round(float(self.contradiction_ratio), 4),
            "quality": self.quality,
            "num_items": self.num_items,
            "context": self.context,
        }


# ---------------------------------------------------------------------------
# Support edge (item -> hypothesis)
# ---------------------------------------------------------------------------


@dataclass
class SupportEdge:
    """Weighted edge from an evidence item to the hypothesis it addresses."""
    source: str               # evidence item id
    target: str               # hypothesis node name
    weight: float = 1.0
    kind: str = "supports"    # "supports" | "dependency"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(float(self.weight), 4),
            "kind": self.kind,
        }


# ---------------------------------------------------------------------------
# Evidence graph
# ---------------------------------------------------------------------------


class EvidenceGraph:
    """Directed evidence graph: items -> hypotheses, with run-level metadata.

    ▸ ``add_item``  — register an atomic measurement.
    ▸ ``add_edge``  — wire an item to a hypothesis node.
    ▸ ``compute_scores`` — aggregate weighted support / contradiction per
        hypothesis and apply the downstream-boost property.
    ▸ ``top_hypotheses`` — ranked leaderboard of the strongest hypotheses.
    """

    def __init__(self, name: str = "rf_hypothesis", metadata: Optional[dict] = None):
        self.name = name
        self.metadata: dict = metadata or {}
        self.items: dict[str, EvidenceItem] = {}
        self.edges: list[SupportEdge] = []
        self.hypotheses: dict[str, HypothesisNode] = {}
        # item id -> hypothesis names linked to it (for reverse lookups)
        self._item_targets: dict[str, list[str]] = {}
        # hypothesis -> items wired with kind="supports" (score-contributing)
        self._support_targets: dict[str, set] = {}

    # -- construction -------------------------------------------------------
    def add_item(self, item: EvidenceItem) -> EvidenceItem:
        self.items[item.id] = item
        return item

    def add_hypothesis(self, name: str, kind: str, label: str = "",
                       parents: Optional[list[str]] = None,
                       context: Optional[dict] = None) -> HypothesisNode:
        if name not in self.hypotheses:
            self.hypotheses[name] = HypothesisNode(
                name=name, kind=kind, label=label or name,
                parents=list(parents or []), context=context or {},
            )
        return self.hypotheses[name]

    def add_edge(self, source_item_id: str, target_hypothesis: str,
                 weight: float = 1.0, kind: str = "supports") -> SupportEdge:
        if source_item_id not in self.items:
            raise KeyError(f"unknown evidence item: {source_item_id}")
        if target_hypothesis not in self.hypotheses:
            self.add_hypothesis(target_hypothesis, kind="modulation")
        edge = SupportEdge(source=source_item_id, target=target_hypothesis,
                           weight=weight, kind=kind)
        self.edges.append(edge)
        self._item_targets.setdefault(source_item_id, []).append(target_hypothesis)
        if kind == "supports":
            self._support_targets.setdefault(target_hypothesis, set()).add(source_item_id)
        self.hypotheses[target_hypothesis].num_items += 1
        return edge

    def add_dependency(self, child: str, parent: str) -> None:
        """Record DAG lineage ``child -> parent`` (decode chain → family)."""
        if child not in self.hypotheses:
            self.add_hypothesis(child, kind="decode_chain")
        if parent not in self.hypotheses:
            self.add_hypothesis(parent, kind="modulation")
        if parent not in self.hypotheses[child].parents:
            self.hypotheses[child].parents.append(parent)

    # -- scoring ------------------------------------------------------------
    def items_for(self, hypothesis: str) -> list[EvidenceItem]:
        """All evidence items currently wired to a hypothesis."""
        return [self.items[i] for i, targets in self._item_targets.items()
                if hypothesis in targets]

    def scoring_items_for(self, hypothesis: str) -> list[EvidenceItem]:
        """Items contributing to a hypothesis's score (``kind == "supports"``)."""
        return [self.items[i] for i in self._support_targets.get(hypothesis, set())]

    # Uncertainity reserve: represents irreducible doubt in any single run.
    # Keeps the net score from saturating at 1.0 on weak-but-positive evidence.
    UNCERTAINTY_RESERVE = 2.0

    def compute_scores(self, eps: float = 1e-9) -> None:
        """Aggregate weighted support/contradiction per hypothesis.

        The result lives in ``hypotheses[name].score`` in [0, 1]: the net
        (support − contradiction) weighted sum is normalized against the total
        possible weighted evidence plus a fixed uncertainty reserve, then
        mapped 0.5 → no/balanced evidence, ~0.9+ → strongly supported.

        Because demodulation and decoding items are wired to the modulation
        family they verified, a successful decode chain raises the family's
        score — the downstream-boost property.
        """
        for name, node in self.hypotheses.items():
            support = 0.0
            contra = 0.0
            for item in self.scoring_items_for(name):
                c = item.contribution()
                if c >= 0:
                    support += c
                else:
                    contra += -c

            total_w = sum(i.weight for i in self.scoring_items_for(name))
            node.support_weight = support
            node.contra_weight = contra
            node.contradiction_ratio = contra / (support + contra + eps)
            if total_w < eps:
                node.score = 0.5
            else:
                net = (support - contra) / (total_w + self.UNCERTAINTY_RESERVE)
                node.score = float(min(1.0, max(0.0, 0.5 * (1.0 + net))))
            node.quality = quality_from_ratio(node.score,
                                              good=0.7, fair=0.45)

        # Downstream-boost pass: a verified child (e.g. a CRC-valid decode
        # chain) further lifts its parent family score, capped at 0.9 so a
        # chain can never single-handedly manufacture a near-certain verdict.
        for name, node in self.hypotheses.items():
            if node.parents:
                family_nodes = [self.hypotheses[p] for p in node.parents
                                if p in self.hypotheses]
                if family_nodes:
                    boost = sum(f.score for f in family_nodes) / len(family_nodes)
                    node.score = min(0.9, node.score + 0.05 * boost)
                    node.quality = quality_from_ratio(node.score,
                                                      good=0.7, fair=0.45)

    def top_hypotheses(self, k: int = 5, kinds: Optional[list[str]] = None) -> list[HypothesisNode]:
        """Ranked leaderboard of hypotheses, best-scoring first."""
        nodes = [n for n in self.hypotheses.values()
                 if (kinds is None or n.kind in kinds)]
        nodes.sort(key=lambda n: n.score, reverse=True)
        return nodes[:k]

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metadata": self.metadata,
            "items": [i.to_dict() for i in self.items.values()],
            "edges": [e.to_dict() for e in self.edges],
            "hypotheses": [h.to_dict() for h in self.hypotheses.values()],
            "top_hypotheses": [
                {"name": h.name, "kind": h.kind, "score": round(h.score, 3), "quality": h.quality}
                for h in self.top_hypotheses(k=5)
            ],
        }