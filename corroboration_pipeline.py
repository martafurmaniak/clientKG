"""
corroboration_pipeline.py — extracts and refines a KG from each corroboration document.

Flow per document
─────────────────
Initial extraction (per page):
  • Build context: page_summary + document_summary + known entities from history KG
  • Extract entities (all types in one call — summaries are short)
  • Extract relationships
  • Consolidate into a running per-document KG

After all pages are processed:
  • Run the shared refinement loop (stray-node + curator iterations)
    via kg_refinement_loop.run_refinement_loop()
  • Save the per-document KG to its output_path
"""

from __future__ import annotations

import json
from pathlib import Path

from schemas import KnowledgeGraph, EntityExtractionResult, RelationshipExtractionResult
from agents import kg_consolidation_agent
from llm_utils import call_llm, parse_and_validate
from prompt_loader import render_with_ontology, get_system_prompt
from kg_refinement_loop import run_refinement_loop
from corroboration_loader import CorroborationDoc


# ─────────────────────────────────────────────────────────────────────────────
# Per-page extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page(
    page: dict,
    document_summary: str,
    entity_ontology: dict,
    relationship_ontology: dict,
    history_kg: KnowledgeGraph,
    doc_kg_so_far: KnowledgeGraph,
    page_idx: int,
    total_pages: int,
) -> tuple[EntityExtractionResult, RelationshipExtractionResult]:
    """
    Extract entities and relationships from a single corroboration page.
    Uses page_summary + document_summary to keep prompts token-efficient.
    """
    page_summary = page.get("page_summary", page.get("page_text", ""))
    page_number  = page.get("page_number", page_idx)

    # Full entity list: pass complete attributes so the model can match on content
    # and reuse the exact same IDs from the history KG
    known_entities = [
        e.to_dict()
        for e in (history_kg.entities + doc_kg_so_far.entities)
    ]

    label = f"CorrExtract[page {page_number}/{total_pages - 1}]"
    system_prompt = get_system_prompt("corroboration_extraction")

    # ── Entity extraction ─────────────────────────────────────────────────────
    ent_user = render_with_ontology(
        "corroboration_extraction.j2",
        entity_ontology=entity_ontology,
        relationship_ontology=relationship_ontology,
        existing_kg=doc_kg_so_far,
        page_summary=page_summary,
        document_summary=document_summary,
        known_entities=known_entities,
        extraction_target="entities",
        page_number=page_number,
    )
    raw_ent    = call_llm(system_prompt, ent_user, label=f"{label}:entities")
    ent_result = parse_and_validate(raw_ent, EntityExtractionResult, label=label)

    # ── Relationship extraction ───────────────────────────────────────────────
    combined_for_rels = KnowledgeGraph(
        entities=doc_kg_so_far.entities + ent_result.entities,
        relationships=doc_kg_so_far.relationships,
    )
    rel_user = render_with_ontology(
        "corroboration_extraction.j2",
        entity_ontology=entity_ontology,
        relationship_ontology=relationship_ontology,
        existing_kg=combined_for_rels,
        page_summary=page_summary,
        document_summary=document_summary,
        known_entities=combined_for_rels.to_serialisable()["entities"],
        extraction_target="relationships",
        page_number=page_number,
    )
    raw_rel    = call_llm(system_prompt, rel_user, label=f"{label}:relationships")
    rel_result = parse_and_validate(raw_rel, RelationshipExtractionResult, label=label)

    print(f"  {label} → entities={len(ent_result.entities)}  "
          f"relationships={len(rel_result.relationships)}")
    return ent_result, rel_result


# ─────────────────────────────────────────────────────────────────────────────
# Per-document pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_corroboration_document(
    doc: CorroborationDoc,
    history_kg: KnowledgeGraph,
    entity_ontology: dict,
    relationship_ontology: dict,
) -> KnowledgeGraph:
    """
    Process a single corroboration document:
      1. Page-by-page initial extraction
      2. Shared stray-node + curator refinement loop
    Returns the refined per-document KG.
    """
    print(f"\n  ── Corroboration document: {doc.source_file} "
          f"({len(doc.pages)} pages) ──")

    doc_kg = KnowledgeGraph()

    # ── Initial extraction page by page ──────────────────────────────────────
    for page_idx, page in enumerate(doc.pages):
        ent_result, rel_result = _extract_page(
            page=page,
            document_summary=doc.document_summary,
            entity_ontology=entity_ontology,
            relationship_ontology=relationship_ontology,
            history_kg=history_kg,
            doc_kg_so_far=doc_kg,
            page_idx=page_idx,
            total_pages=len(doc.pages),
        )
        doc_kg = kg_consolidation_agent(
            existing_kg=doc_kg,
            new_entities=ent_result,
            new_relationships=rel_result,
        )

    print(f"  ── Initial extraction complete: {len(doc_kg.entities)} entities, "
          f"{len(doc_kg.relationships)} relationships")

    # ── Refinement loop (stray-node + curator) ────────────────────────────────
    # Build plain-text representations for the refinement loop:
    # document_text = all page summaries joined (used by stray-node agent)
    # document_pages = per-page summaries (used by curator improvement agents)
    doc_text  = doc.document_summary + "\n\n" + "\n\n".join(
        p.get("page_summary", p.get("page_text", "")) for p in doc.pages
    )
    doc_pages = [
        p.get("page_summary", p.get("page_text", "")) for p in doc.pages
    ]

    label = Path(doc.source_file).stem
    refinement = run_refinement_loop(
        kg=doc_kg,
        document_text=doc_text,
        document_pages=doc_pages,
        entity_ontology=entity_ontology,
        relationship_ontology=relationship_ontology,
        label=label,
    )

    doc_kg = refinement.kg
    print(f"  ── After refinement: {len(doc_kg.entities)} entities, "
          f"{len(doc_kg.relationships)} relationships")

    if refinement.ontology_gap_report.get("total_gaps"):
        n = refinement.ontology_gap_report["total_gaps"]
        print(f"  ── Ontology gaps recorded for this document: {n}")

    return doc_kg


# ─────────────────────────────────────────────────────────────────────────────
# Full corroboration phase
# ─────────────────────────────────────────────────────────────────────────────

def run_corroboration_phase(
    docs: list[CorroborationDoc],
    history_kg: KnowledgeGraph,
    entity_ontology: dict,
    relationship_ontology: dict,
) -> dict[str, KnowledgeGraph]:
    """
    Process all corroboration documents.
    Saves each KG to its output_path and returns a dict of source_file → KG.
    """
    if not docs:
        print("  [CorrPhase] No corroboration documents to process.")
        return {}

    results: dict[str, KnowledgeGraph] = {}

    for doc in docs:
        doc_kg = run_corroboration_document(
            doc=doc,
            history_kg=history_kg,
            entity_ontology=entity_ontology,
            relationship_ontology=relationship_ontology,
        )

        doc.output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.output_path.write_text(
            json.dumps(doc_kg.to_output_format(), indent=2),
            encoding="utf-8",
        )
        print(f"  [CorrPhase] Saved → {doc.output_path}")
        results[doc.source_file] = doc_kg

    return results
