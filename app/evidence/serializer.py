"""
app/evidence/serializer.py
--------------------------
Serialization for the Evidence Graph: JSON (round-trippable) and GraphML
(dependency-free, intended for network visualization tools such as Gephi /
yEd). No third-party graph library is required — GraphML is emitted directly
via stdlib ``xml.etree``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

from app.evidence.models import EvidenceGraph

# GraphML XML namespace (used by Gephi, yEd, igraph, ...)
GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
KEY_DOMAIN_NODE = "node"
KEY_DOMAIN_EDGE = "edge"

NODE_ATTRS = ("kind", "label", "score", "quality", "num_items")
EDGE_ATTRS = ("kind", "weight")

_NODE_KEYS = {  # attr -> (key id, type)
    "kind": ("kind", "string"),
    "label": ("label", "string"),
    "score": ("score", "double"),
    "quality": ("quality", "string"),
    "num_items": ("num_items", "int"),
}
_EDGE_KEYS = {
    "kind": ("kind", "string"),
    "weight": ("weight", "double"),
}


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def graph_to_json(graph: EvidenceGraph, indent: int = 2) -> str:
    """Serialize an evidence graph to a JSON string."""
    return json.dumps(graph.to_dict(), indent=indent, default=str)


def graph_from_json(text: Union[str, dict]) -> EvidenceGraph:
    """Reconstruct an evidence graph from a JSON string or parsed dict.

    Item timestamps are preserved from the payload because the dict is the
    reverse of ``EvidenceGraph.to_dict``.
    """
    data = text if isinstance(text, dict) else json.loads(text)
    graph = EvidenceGraph(name=str(data.get("name", "rf_hypothesis")),
                          metadata=dict(data.get("metadata", {}) or {}))

    for item in data.get("items", []):
        from app.evidence.models import EvidenceItem
        graph.add_item(EvidenceItem(**{k: v for k, v in item.items()
                                       if k in EvidenceItem.__dataclass_fields__}))
    for hyp in data.get("hypotheses", []):
        from app.evidence.models import HypothesisNode
        node = HypothesisNode(
            name=hyp["name"], kind=hyp.get("kind", "modulation"),
            label=hyp.get("label", hyp["name"]),
            parents=list(hyp.get("parents", [])),
            context=dict(hyp.get("context", {}) or {}),
        )
        node.score = float(hyp.get("score", 0.5))
        node.quality = hyp.get("quality", "poor")
        node.num_items = int(hyp.get("num_items", 0))
        graph.hypotheses[node.name] = node
    for edge in data.get("edges", []):
        graph.add_edge(edge["source"], edge["target"],
                       weight=edge.get("weight", 1.0),
                       kind=edge.get("kind", "supports"))
    return graph


def save_json(graph: EvidenceGraph, path: Union[str, Path], indent: int = 2) -> Path:
    """Persist the evidence graph as JSON."""
    path = Path(path)
    path.write_text(graph_to_json(graph, indent=indent), encoding="utf-8")
    return path


def load_json(path: Union[str, Path]) -> EvidenceGraph:
    """Load an evidence graph persisted with ``save_json``."""
    path = Path(path)
    return graph_from_json(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# GraphML
# ---------------------------------------------------------------------------


def graph_to_graphml(graph: EvidenceGraph) -> str:
    """Serialize the evidence graph to a GraphML XML string.

    Nodes: hypothesis nodes plus evidence items (as simple nodes carrying the
    item's measurement/quality). Edges: item → hypothesis (**supports**).
    Hypothesis → hypothesis dependency edges are emitted with kind
    ``dependency`` for the DAG lineage (decode chain → modulation family).
    """
    root = ET.Element(f"{{{GRAPHML_NS}}}graphml")
    root.set("xmlns", GRAPHML_NS)

    for _, (kid, ktype) in _NODE_KEYS.items():
        ET.SubElement(root, f"{{{GRAPHML_NS}}}key", id=kid, for_=KEY_DOMAIN_NODE,
                      attr_name=kid, attr_type=ktype)
    for _, (kid, ktype) in _EDGE_KEYS.items():
        ET.SubElement(root, f"{{{GRAPHML_NS}}}key", id=kid, for_=KEY_DOMAIN_EDGE,
                      attr_name=kid, attr_type=ktype)

    graph_el = ET.SubElement(root, f"{{{GRAPHML_NS}}}graph",
                             id="G", edgedefault="directed")

    # -- items as nodes -----------------------------------------------------
    for item in graph.items.values():
        node = ET.SubElement(graph_el, f"{{{GRAPHML_NS}}}node", id=item.id)
        _add_data(node, "kind", f"item:{item.stage}")
        _add_data(node, "label", f"{item.stage}:{item.measurement}\n{item.candidate}")
        _add_data(node, "score", _num(item.value_norm))
        _add_data(node, "quality", item.quality)
        _add_data(node, "num_items", "0")

    # -- hypotheses as nodes ------------------------------------------------
    for hyp in graph.hypotheses.values():
        node = ET.SubElement(graph_el, f"{{{GRAPHML_NS}}}node", id=f"hyp:{hyp.name}")
        _add_data(node, "kind", hyp.kind)
        _add_data(node, "label", hyp.label)
        _add_data(node, "score", _num(hyp.score))
        _add_data(node, "quality", hyp.quality)
        _add_data(node, "num_items", str(hyp.num_items))

    # -- edges --------------------------------------------------------------
    for edge in graph.edges:
        ET.SubElement(graph_el, f"{{{GRAPHML_NS}}}edge",
                      id=f"e{edge.source}:{edge.target}",
                      source=edge.source, target=f"hyp:{edge.target}")
    for hyp in graph.hypotheses.values():
        for parent in hyp.parents:
            if parent in graph.hypotheses:
                e = ET.SubElement(graph_el, f"{{{GRAPHML_NS}}}edge",
                                  id=f"d{hyp.name}:{parent}",
                                  source=f"hyp:{hyp.name}",
                                  target=f"hyp:{parent}")
                _add_data(e, "kind", "dependency")
                _add_data(e, "weight", "1.0")

    return ET.tostring(root, encoding="unicode")


def save_graphml(graph: EvidenceGraph, path: Union[str, Path]) -> Path:
    """Persist the evidence graph as GraphML (for Gephi / yEd)."""
    path = Path(path)
    path.write_text(graph_to_graphml(graph), encoding="utf-8")
    return path


def _add_data(el: ET.Element, key: str, value: str) -> None:
    data = ET.SubElement(el, f"{{{GRAPHML_NS}}}data", key=key)
    data.text = value


def _num(value: float) -> str:
    return f"{float(value):.4f}"


# ---------------------------------------------------------------------------
# Convenience CLI-like exporter
# ---------------------------------------------------------------------------


def export_evidence_graph(graph: EvidenceGraph, out_dir: Union[str, Path],
                          basename: str = "evidence",
                          fmt: tuple[str, ...] = ("json", "graphml")) -> dict:
    """Write the graph to JSON and/or GraphML files under ``out_dir``.

    Returns ``{"json": Path|None, "graphml": Path|None}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: dict = {"json": None, "graphml": None}
    if "json" in fmt:
        exported["json"] = save_json(graph, out_dir / f"{basename}.json")
    if "graphml" in fmt:
        exported["graphml"] = save_graphml(graph, out_dir / f"{basename}.graphml")
    return exported