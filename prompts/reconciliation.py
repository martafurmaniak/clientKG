"""
reconciliation.py — reconcile the client history KG against corroboration KGs.

For every relationship in the client history graph, classify it as one of:

  directly_corroborated   — an identical or semantically equivalent relationship
                            exists in at least one corroboration graph (same
                            source type, target type, and relationship type, where
                            both endpoints are aligned entities).

  path_corroborated       — a multi-hop path in a corroboration graph entails the
                            relationship according to ENTAILMENT_RULES.

  partially_corroborated  — both endpoints exist (as aligned entities) in a
                            corroboration graph but no matching or entailing path
                            was found.

  uncorroborated          — at least one endpoint has no aligned counterpart in
                            any corroboration graph.

Each relationship in the output gets two new attributes:
  status            — one of the four values above
  corroborating_doc — source_file of the corroboration document that provided
                      the best corroboration (empty string if uncorroborated)

For path_corroborated relationships, missing intermediary nodes (nodes that
appear in the entailing path in the corroboration graph but are absent from
the history graph) are surfaced in the reconciliation report.

Entity alignment
────────────────
Entities are aligned across graphs by normalised label (case-insensitive,
whitespace-stripped). Alignment does not require the same ID — it is common
for the same person to have different IDs across graphs.

ENTAILMENT_RULES
────────────────
A configurable dict mapping a tuple of relationship types (a path) to an
implied relationship type.  Example:

  ENTAILMENT_RULES = {
      ("OWNS", "ACCOUNT_AT"): "ACCOUNT_AT",   # P owns account A at bank B → P ACCOUNT_AT B
      ("LEADS", "OWNS"):      "BENEFICIAL_OWNER_OF",
  }

Set ENTAILMENT_RULES = {} to disable path corroboration entirely.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from schemas import KnowledgeGraph, Relationship


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Edit this to add domain-specific entailment rules.
# Keys are tuples of relationship type sequences; values are the implied type.
ENTAILMENT_RULES: dict[tuple[str, ...], str] = {
    ("OWNS", "ACCOUNT_AT"):         "ACCOUNT_AT",
    ("LEADS", "BENEFICIAL_OWNER_OF"): "BENEFICIAL_OWNER_OF",
    ("RELATED_TO", "OWNS"):         "OWNS",
}

# Maximum path length to search for entailment
MAX_PATH_LENGTH = 3


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RelationshipCorroboration:
    """Corroboration result for a single history relationship."""
    status:              str = "uncorroborated"
    corroborating_doc:   str = ""
    # Path corroboration extras
    entailing_path:      list[str] = field(default_factory=list)   # [rel_type, ...]
    missing_intermediaries: list[dict] = field(default_factory=list)  # nodes in path but not in history


@dataclass
class ReconciliationReport:
    """Full reconciliation result."""
    # Annotated history KG (relationships carry status + corroborating_doc)
    annotated_kg:             dict = field(default_factory=dict)
    # Summary counts
    directly_corroborated:    int = 0
    path_corroborated:        int = 0
    partially_corroborated:   int = 0
    uncorroborated:           int = 0
    # Relationships that are path-corroborated with missing intermediaries
    missing_intermediaries:   list[dict] = field(default_factory=list)
    # Per-relationship detail (source→target→type → corroboration result)
    details:                  dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Entity alignment
# ─────────────────────────────────────────────────────────────────────────────

def _norm(label: str) -> str:
    return label.strip().lower() if label else ""


def _build_alignment(
    history_kg: KnowledgeGraph,
    corr_kg: KnowledgeGraph,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Align entities between history and corroboration KG by normalised label.

    Returns:
      hist_to_corr : history entity id  → corroboration entity id
      corr_to_hist : corroboration entity id → history entity id
    """
    corr_by_label = {_norm(e.label): e.id for e in corr_kg.entities if e.label}
    hist_by_label = {_norm(e.label): e.id for e in history_kg.entities if e.label}

    hist_to_corr: dict[str, str] = {}
    corr_to_hist: dict[str, str] = {}

    for e in history_kg.entities:
        n = _norm(e.label)
        if n and n in corr_by_label:
            hist_to_corr[e.id] = corr_by_label[n]

    for e in corr_kg.entities:
        n = _norm(e.label)
        if n and n in hist_by_label:
            corr_to_hist[e.id] = hist_by_label[n]

    return hist_to_corr, corr_to_hist


# ─────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ─────────────────────────────────────────────────────────────────────────────

def _adjacency(kg: KnowledgeGraph) -> dict[str, list[tuple[str, str]]]:
    """
    Build adjacency list: entity_id → [(neighbour_id, rel_type), ...]
    Directed — includes both directions for path search.
    """
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in kg.relationships:
        adj[r.source].append((r.target, r.type))
        adj[r.target].append((r.source, r.type))   # undirected search
    return adj


def _find_paths(
    adj: dict[str, list[tuple[str, str]]],
    start: str,
    end: str,
    max_len: int,
) -> list[list[tuple[str, str]]]:
    """
    BFS for all simple paths from start → end of length ≤ max_len.
    Returns list of paths, each path = [(node_id, rel_type), ...] where
    rel_type is the edge used to reach that node.
    """
    if start == end:
        return []
    results = []
    # Queue: (current_node, path_so_far, visited)
    queue = [(start, [], {start})]
    while queue:
        node, path, visited = queue.pop(0)
        if len(path) >= max_len:
            continue
        for neighbour, rel_type in adj.get(node, []):
            if neighbour in visited:
                continue
            new_path = path + [(neighbour, rel_type)]
            if neighbour == end:
                results.append(new_path)
            else:
                queue.append((neighbour, new_path, visited | {neighbour}))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Direct corroboration check
# ─────────────────────────────────────────────────────────────────────────────

def _check_direct(
    hist_rel: Relationship,
    hist_to_corr: dict[str, str],
    corr_rels_by_key: dict[tuple[str, str, str], Relationship],
) -> bool:
    """
    Return True if an exact relationship (same type, aligned endpoints) exists
    in the corroboration graph.
    """
    corr_src = hist_to_corr.get(hist_rel.source)
    corr_tgt = hist_to_corr.get(hist_rel.target)
    if not corr_src or not corr_tgt:
        return False
    return (corr_src, corr_tgt, hist_rel.type) in corr_rels_by_key


# ─────────────────────────────────────────────────────────────────────────────
# Path corroboration check
# ─────────────────────────────────────────────────────────────────────────────

def _check_path(
    hist_rel: Relationship,
    hist_to_corr: dict[str, str],
    corr_kg: KnowledgeGraph,
    corr_adj: dict[str, list[tuple[str, str]]],
    history_entity_ids: set[str],
) -> tuple[bool, list[str], list[dict]]:
    """
    Return (found, entailing_path_types, missing_intermediaries).
    Checks ENTAILMENT_RULES against all paths in the corroboration graph
    between the aligned source and target.
    """
    corr_src = hist_to_corr.get(hist_rel.source)
    corr_tgt = hist_to_corr.get(hist_rel.target)
    if not corr_src or not corr_tgt:
        return False, [], []

    paths = _find_paths(corr_adj, corr_src, corr_tgt, MAX_PATH_LENGTH)
    corr_entity_by_id = {e.id: e for e in corr_kg.entities}

    for path in paths:
        # Extract the sequence of relationship types along the path
        rel_sequence = tuple(rel_type for _, rel_type in path)
        if rel_sequence in ENTAILMENT_RULES:
            implied = ENTAILMENT_RULES[rel_sequence]
            if implied == hist_rel.type:
                # Found entailing path — identify intermediary nodes
                intermediary_corr_ids = [node_id for node_id, _ in path[:-1]]
                missing: list[dict] = []
                for cid in intermediary_corr_ids:
                    entity = corr_entity_by_id.get(cid)
                    if entity and entity.id not in history_entity_ids:
                        missing.append({
                            "corr_entity_id": cid,
                            "label":          entity.label,
                            "type":           entity.type,
                            "attributes":     entity.attributes,
                        })
                return True, list(rel_sequence), missing

    return False, [], []


# ─────────────────────────────────────────────────────────────────────────────
# Per-relationship reconciliation
# ─────────────────────────────────────────────────────────────────────────────

def _reconcile_relationship(
    hist_rel: Relationship,
    source_file: str,
    hist_to_corr: dict[str, str],
    corr_rels_by_key: dict[tuple[str, str, str], Relationship],
    corr_kg: KnowledgeGraph,
    corr_adj: dict[str, list[tuple[str, str]]],
    history_entity_ids: set[str],
) -> RelationshipCorroboration:
    """Classify a single history relationship against one corroboration graph."""

    # 1. Direct corroboration
    if _check_direct(hist_rel, hist_to_corr, corr_rels_by_key):
        return RelationshipCorroboration(
            status="directly_corroborated",
            corroborating_doc=source_file,
        )

    # 2. Path corroboration
    found, path_types, missing = _check_path(
        hist_rel, hist_to_corr, corr_kg, corr_adj, history_entity_ids
    )
    if found:
        return RelationshipCorroboration(
            status="path_corroborated",
            corroborating_doc=source_file,
            entailing_path=path_types,
            missing_intermediaries=missing,
        )

    # 3. Partial corroboration — both endpoints aligned but no path match
    corr_src = hist_to_corr.get(hist_rel.source)
    corr_tgt = hist_to_corr.get(hist_rel.target)
    if corr_src and corr_tgt:
        return RelationshipCorroboration(
            status="partially_corroborated",
            corroborating_doc=source_file,
        )

    # 4. Uncorroborated
    return RelationshipCorroboration(status="uncorroborated")


# ─────────────────────────────────────────────────────────────────────────────
# Priority ordering for corroboration statuses
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_PRIORITY = {
    "directly_corroborated":  0,
    "path_corroborated":      1,
    "partially_corroborated": 2,
    "uncorroborated":         3,
}


def _better(a: RelationshipCorroboration, b: RelationshipCorroboration) -> bool:
    """Return True if a is a stronger corroboration than b."""
    return _STATUS_PRIORITY[a.status] < _STATUS_PRIORITY[b.status]


# ─────────────────────────────────────────────────────────────────────────────
# Main reconciliation entry point
# ─────────────────────────────────────────────────────────────────────────────

def reconcile(
    history_kg: KnowledgeGraph,
    corr_results: dict[str, KnowledgeGraph],
) -> ReconciliationReport:
    """
    Reconcile the client history KG against all corroboration KGs.

    Parameters
    ──────────
    history_kg   : the final client history KnowledgeGraph
    corr_results : dict mapping source_file → per-document KnowledgeGraph

    Returns a ReconciliationReport with an annotated copy of the history KG
    and a detailed breakdown per relationship.
    """
    if not corr_results:
        print("  [Reconciliation] No corroboration graphs — all relationships uncorroborated.")
        # Annotate all as uncorroborated and return
        annotated_rels = []
        for rel in history_kg.relationships:
            attrs = dict(rel.attributes)
            attrs["status"] = "uncorroborated"
            attrs["corroborating_doc"] = ""
            annotated_rels.append(rel.model_copy(update={"attributes": attrs}))
        annotated_kg = KnowledgeGraph(
            entities=history_kg.entities,
            relationships=annotated_rels,
        )
        return ReconciliationReport(
            annotated_kg=annotated_kg.to_output_format(),
            uncorroborated=len(annotated_rels),
        )

    history_entity_ids = {e.id for e in history_kg.entities}

    # Best corroboration result per relationship key (source, target, type)
    best: dict[tuple[str, str, str], RelationshipCorroboration] = {}

    for source_file, corr_kg in corr_results.items():
        hist_to_corr, _ = _build_alignment(history_kg, corr_kg)

        # Index corroboration relationships by (source, target, type)
        corr_rels_by_key: dict[tuple[str, str, str], Relationship] = {
            (r.source, r.target, r.type): r for r in corr_kg.relationships
        }
        corr_adj = _adjacency(corr_kg)

        for hist_rel in history_kg.relationships:
            key = (hist_rel.source, hist_rel.target, hist_rel.type)
            result = _reconcile_relationship(
                hist_rel=hist_rel,
                source_file=source_file,
                hist_to_corr=hist_to_corr,
                corr_rels_by_key=corr_rels_by_key,
                corr_kg=corr_kg,
                corr_adj=corr_adj,
                history_entity_ids=history_entity_ids,
            )
            # Keep the best corroboration found across all documents
            if key not in best or _better(result, best[key]):
                best[key] = result

    # Build annotated KG and report
    report = ReconciliationReport()
    annotated_rels = []

    for hist_rel in history_kg.relationships:
        key = (hist_rel.source, hist_rel.target, hist_rel.type)
        corr = best.get(key, RelationshipCorroboration(status="uncorroborated"))

        # Annotate the relationship
        attrs = dict(hist_rel.attributes)
        attrs["status"] = corr.status
        attrs["corroborating_doc"] = corr.corroborating_doc
        if corr.entailing_path:
            attrs["entailing_path"] = " → ".join(corr.entailing_path)
        annotated_rels.append(hist_rel.model_copy(update={"attributes": attrs}))

        # Count + record
        if corr.status == "directly_corroborated":
            report.directly_corroborated += 1
        elif corr.status == "path_corroborated":
            report.path_corroborated += 1
            if corr.missing_intermediaries:
                report.missing_intermediaries.append({
                    "relationship": {
                        "source": hist_rel.source,
                        "target": hist_rel.target,
                        "type":   hist_rel.type,
                    },
                    "entailing_path": corr.entailing_path,
                    "missing_intermediaries": corr.missing_intermediaries,
                    "corroborating_doc": corr.corroborating_doc,
                })
        elif corr.status == "partially_corroborated":
            report.partially_corroborated += 1
        else:
            report.uncorroborated += 1

        report.details[f"{hist_rel.source}→{hist_rel.target}[{hist_rel.type}]"] = {
            "status":                corr.status,
            "corroborating_doc":     corr.corroborating_doc,
            "entailing_path":        corr.entailing_path,
            "missing_intermediaries": corr.missing_intermediaries,
        }

    annotated_kg = KnowledgeGraph(
        entities=history_kg.entities,
        relationships=annotated_rels,
    )
    report.annotated_kg = annotated_kg.to_output_format()

    total = len(history_kg.relationships)
    print(f"  [Reconciliation] {total} relationships classified:")
    print(f"    directly_corroborated  : {report.directly_corroborated}")
    print(f"    path_corroborated      : {report.path_corroborated}")
    print(f"    partially_corroborated : {report.partially_corroborated}")
    print(f"    uncorroborated         : {report.uncorroborated}")
    if report.missing_intermediaries:
        print(f"    missing intermediaries : {len(report.missing_intermediaries)}")

    return report


def save_reconciliation_report(report: ReconciliationReport, output_dir: Path) -> None:
    """Save the annotated KG and full reconciliation report to output_dir."""
    annotated_path = output_dir / "reconciled_kg.json"
    report_path    = output_dir / "reconciliation_report.json"

    annotated_path.write_text(json.dumps(report.annotated_kg, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps({
        "summary": {
            "directly_corroborated":  report.directly_corroborated,
            "path_corroborated":      report.path_corroborated,
            "partially_corroborated": report.partially_corroborated,
            "uncorroborated":         report.uncorroborated,
        },
        "missing_intermediaries": report.missing_intermediaries,
        "details": report.details,
    }, indent=2), encoding="utf-8")

    print(f"\n✅  Reconciliation outputs saved:")
    print(f"   Annotated KG           → {annotated_path}")
    print(f"   Reconciliation report  → {report_path}")
