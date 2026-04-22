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

from schemas import KnowledgeGraph, EntityExtractionResult, RelationshipExtractionResult, CorroborationEntityResult
from ontology_utils import assign_ids, GlobalIDRegistry
from agents import kg_consolidation_agent
from llm_utils import call_llm, parse_and_validate
from prompt_loader import render_with_ontology, get_system_prompt
from kg_refinement_loop import run_refinement_loop
from corroboration_loader import CorroborationDoc


# ─────────────────────────────────────────────────────────────────────────────
# Fix 7: Python-side entity deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_against_existing(
    new_result: EntityExtractionResult,
    history_kg: KnowledgeGraph,
    doc_kg_so_far: KnowledgeGraph,
) -> EntityExtractionResult:
    """
    Resolve extracted entities against the history KG and doc-so-far:

    - If an extracted entity matches one already in history_kg or doc_kg_so_far
      (by normalised label), replace it with the existing entity so the original
      ID and attributes are preserved.
    - Genuinely new entities (no match) are kept as-is.

    Either way, all entities — reused and new — are returned so they all end
    up in doc_kg. The document KG is self-contained: every entity referenced
    by a relationship exists as a node within it.
    """
    # Build label → entity lookup across both known sources
    existing_by_label: dict[str, object] = {}
    for e in (history_kg.entities + doc_kg_so_far.entities):
        if e.label:
            existing_by_label[e.label.strip().lower()] = e

    resolved: list = []
    reused = 0
    for entity in new_result.entities:
        norm = entity.label.strip().lower() if entity.label else ""
        if norm and norm in existing_by_label:
            # Replace with the canonical existing entity (original ID + attributes)
            resolved.append(existing_by_label[norm])
            reused += 1
            print(f"  [CorrDedup] Reused existing entity '{entity.label}' "
                  f"(id={existing_by_label[norm].id})")
        else:
            resolved.append(entity)

    if reused:
        print(f"  [CorrDedup] Reused {reused} existing entity(ies), "
              f"{len(resolved) - reused} genuinely new")

    return EntityExtractionResult(
        entities=resolved,
        entities_to_remove=new_result.entities_to_remove,
    )


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
    id_registry: "GlobalIDRegistry | None" = None,
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
    # combined_kg is used solely for ID seeding — it contains all known entities
    # across both the history KG and this document so far, so new IDs never clash
    combined_kg = KnowledgeGraph(entities=history_kg.entities + doc_kg_so_far.entities)

    ent_user = render_with_ontology(
        "corroboration_extraction.j2",
        entity_ontology=entity_ontology,
        relationship_ontology=relationship_ontology,
        page_summary=page_summary,
        document_summary=document_summary,
        known_entities=known_entities,
        extraction_target="entities",
        page_number=page_number,
    )
    raw_ent    = call_llm(system_prompt, ent_user, label=f"{label}:entities")
    corr_result = parse_and_validate(raw_ent, CorroborationEntityResult, label=label)

    # ── Resolve reused_ids → canonical entities from combined_kg ─────────────
    # The LLM explicitly listed the IDs of existing entities it recognises.
    # Look them up directly — no label guessing needed.
    combined_id_map = {e.id: e for e in combined_kg.entities}
    reused_entities = []
    for eid in corr_result.reused_ids:
        if eid in combined_id_map:
            reused_entities.append(combined_id_map[eid])
            print(f"  [CorrDedup] Reused entity id={eid} ({combined_id_map[eid].label})")
        else:
            print(f"  [CorrDedup] ⚠ reused_id '{eid}' not found in combined KG — skipped")

    # ── Assign deterministic IDs to genuinely new entities ────────────────────
    # Seed from combined_kg so new IDs never clash with any existing entity.
    new_entities = corr_result.new_entities
    if new_entities:
        new_entities, id_map = assign_ids(new_entities, combined_kg, entity_ontology)
        new_to_remove = [id_map.get(eid, eid) for eid in corr_result.entities_to_remove]
    else:
        new_to_remove = corr_result.entities_to_remove

    # ── Combine: reused (canonical) + new (freshly assigned IDs) ─────────────
    ent_result = EntityExtractionResult(
        entities=reused_entities + new_entities,
        entities_to_remove=new_to_remove,
    )
    print(f"  {label} entities: {len(reused_entities)} reused, {len(new_entities)} new")

    # ── Relationship extraction ───────────────────────────────────────────────
    # Include history_kg entities in the relationship context so the LLM can
    # reference existing IDs (P1, O2, etc.) when linking to history entities.
    # These are passed as known_entities only — not added to doc_kg.
    all_entities_for_rels = (
        history_kg.entities + doc_kg_so_far.entities + ent_result.entities
    )
    rel_user = render_with_ontology(
        "corroboration_extraction.j2",
        entity_ontology=entity_ontology,
        relationship_ontology=relationship_ontology,
        page_summary=page_summary,
        document_summary=document_summary,
        known_entities=[e.to_dict() for e in all_entities_for_rels],
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
    id_registry: "GlobalIDRegistry | None" = None,
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
            id_registry=id_registry,
        )
        doc_kg = kg_consolidation_agent(
            existing_kg=doc_kg,
            new_entities=ent_result,
            new_relationships=rel_result,
            entity_ontology=entity_ontology,
            relationship_ontology=relationship_ontology,
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
        id_registry=id_registry,
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
    id_registry: "GlobalIDRegistry | None" = None,
) -> dict[str, KnowledgeGraph]:
    """
    Process all corroboration documents in order.

    A cumulative_kg is maintained across documents: it starts as the client
    history KG and grows with every new entity discovered in each document.
    This ensures that entities found in document N are available as known
    entities when processing document N+1, preventing re-extraction with
    new IDs.

    Each document KG is still saved independently (self-contained with all
    its own entities), but the shared entity pool grows across documents.
    """
    if not docs:
        print("  [CorrPhase] No corroboration documents to process.")
        return {}

    results: dict[str, KnowledgeGraph] = {}

    # Starts as the client history KG; accumulates new entities from each doc
    cumulative_kg = history_kg

    # Seed registry from history KG if not provided externally
    if id_registry is None:
        id_registry = GlobalIDRegistry.from_kg(history_kg)
    else:
        id_registry.register_kg(history_kg)

    for doc in docs:
        doc_kg = run_corroboration_document(
            doc=doc,
            history_kg=cumulative_kg,
            entity_ontology=entity_ontology,
            relationship_ontology=relationship_ontology,
            id_registry=id_registry,
        )

        doc.output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.output_path.write_text(
            json.dumps(doc_kg.to_output_format(), indent=2),
            encoding="utf-8",
        )
        print(f"  [CorrPhase] Saved → {doc.output_path}")
        results[doc.source_file] = doc_kg

        # Grow the cumulative entity pool with any new entities from this doc.
        # Dedup by ID — existing entities are never overwritten.
        existing_ids = {e.id for e in cumulative_kg.entities}
        new_entities  = [e for e in doc_kg.entities if e.id not in existing_ids]
        for e in new_entities:
            id_registry.register_id(e.id, e.type)
        if new_entities:
            cumulative_kg = KnowledgeGraph(
                entities=cumulative_kg.entities + new_entities,
                relationships=cumulative_kg.relationships,
            )
            print(f"  [CorrPhase] Cumulative entity pool: "
                  f"{len(cumulative_kg.entities)} entities "
                  f"(+{len(new_entities)} new from {doc.source_file})")

    return results
