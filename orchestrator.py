"""
orchestrator.py — client history KG extraction pipeline.

Flow
────
Phase 1 · Entity extraction
  PeopleOrgsAgent + AssetsAgent + TransactionsAgent → KGConsolidationAgent

Phase 2 · Relationship extraction
  RelationshipAgent (page by page) → KGConsolidationAgent

Phase 3+4 · Stray node + curator refinement loop
  Delegated to kg_refinement_loop.run_refinement_loop()
  (shared with corroboration pipeline)

Phase 5 · Contradiction spotting
"""

import json
from agents import (
    people_and_orgs_agent,
    assets_agent,
    transactions_agent,
    kg_consolidation_agent,
    relationship_extraction_agent,
    contradiction_spotting_agent,
)
from schemas import KnowledgeGraph, EntityExtractionResult, RelationshipExtractionResult
from kg_refinement_loop import run_refinement_loop
from ontology_utils import GlobalIDRegistry


def _banner(text: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)


def _section(text: str) -> None:
    print(f"\n── {text} {'─' * max(0, 65 - len(text))}")


def run_pipeline(
    document_text: str,
    document_pages: list[str],
    entity_ontology: dict,
    relationship_ontology: dict,
    id_registry: "GlobalIDRegistry | None" = None,
) -> dict:
    """
    Execute the full client-history KG extraction pipeline.
    Returns a plain dict (serialisable) with the final KG and reports.
    """
    _banner("KNOWLEDGE GRAPH EXTRACTION PIPELINE — START")

    # Global ID registry — ensures every entity ID is unique across the whole run.
    # Created here if not provided (history-only run); passed in from main.py
    # when a corroboration phase follows so IDs remain unique across both phases.
    if id_registry is None:
        id_registry = GlobalIDRegistry()

    # ── Phase 1: Entity Extraction ───────────────────────────────────────────
    _banner("PHASE 1 · Entity Extraction")

    # Each agent seeds IDs from the accumulated results of the previous agents
    # so no two agents can assign the same ID even for the same entity type.
    _section("Running PeopleOrgsAgent")
    people_orgs: EntityExtractionResult = people_and_orgs_agent(document_text, entity_ontology, id_registry=id_registry)

    _section("Running AssetsAgent")
    _seed_after_people = KnowledgeGraph(entities=people_orgs.entities)
    assets: EntityExtractionResult = assets_agent(
        document_text, entity_ontology, id_seed_kg=_seed_after_people,
        id_registry=id_registry,
    )

    _section("Running TransactionsAgent")
    _seed_after_assets = KnowledgeGraph(entities=people_orgs.entities + assets.entities)
    transactions: EntityExtractionResult = transactions_agent(
        document_text, entity_ontology, id_seed_kg=_seed_after_assets,
        id_registry=id_registry,
    )

    _section("KGConsolidationAgent — merging initial entity extractions")
    combined_entities = EntityExtractionResult(
        entities=[e for r in (people_orgs, assets, transactions) for e in r.entities]
    )
    kg: KnowledgeGraph = kg_consolidation_agent(
        existing_kg=KnowledgeGraph(),
        new_entities=combined_entities,
        entity_ontology=entity_ontology,
        relationship_ontology=relationship_ontology,
    )

    # ── Phase 2: Relationship Extraction ─────────────────────────────────────
    _banner("PHASE 2 · Relationship Extraction")

    _section("Running RelationshipAgent (page by page)")
    rel_result: RelationshipExtractionResult = relationship_extraction_agent(
        document_pages=document_pages,
        relationship_ontology=relationship_ontology,
        kg=kg,
    )

    _section("KGConsolidationAgent — merging relationships")
    kg = kg_consolidation_agent(
        existing_kg=kg, new_relationships=rel_result,
        entity_ontology=entity_ontology,
        relationship_ontology=relationship_ontology,
    )

    # ── Phases 3+4: Stray node + curator loop (shared) ───────────────────────
    refinement = run_refinement_loop(
        kg=kg,
        document_text=document_text,
        document_pages=document_pages,
        entity_ontology=entity_ontology,
        relationship_ontology=relationship_ontology,
        label="ClientHistory",
        id_registry=id_registry,
    )
    kg = refinement.kg

    # ── Phase 5: Contradiction Spotting ──────────────────────────────────────
    _banner("PHASE 5 · Contradiction Spotting")

    _section("Running ContradictionAgent")
    contradiction_report = contradiction_spotting_agent(kg, document_text)

    # ── Done ──────────────────────────────────────────────────────────────────
    _banner("PIPELINE COMPLETE")
    print(f"\n  Final KG : {len(kg.entities)} entities, {len(kg.relationships)} relationships")
    print(f"  Contradictions found: {len(contradiction_report.contradictions)}")

    gap_report = refinement.ontology_gap_report
    if gap_report.get("total_gaps"):
        print(f"  Ontology gaps recorded: {gap_report['total_gaps']} "
              f"(see ontology_gap_report.json)")

    return {
        "status": "complete",
        "kg": kg.to_output_format(),
        "contradiction_report": contradiction_report.model_dump(exclude_none=True),
        "ontology_gap_report": gap_report,
        "id_registry": id_registry,   # returned so corroboration phase can continue from same registry
    }
