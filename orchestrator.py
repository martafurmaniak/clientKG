"""
orchestrator.py — the orchestrator agent that drives the full multi-agent pipeline.

All inter-agent state is carried as typed Pydantic models (KnowledgeGraph,
KGCompletenessResult, etc.).  Plain dicts are only used at the boundary
where we serialise for LLM prompts or file output.

Flow
────
Phase 1 · Entity extraction
  PeopleOrgsAgent + AssetsAgent + TransactionsAgent → KGConsolidationAgent

Phase 2 · Relationship extraction
  RelationshipAgent (page by page) → KGConsolidationAgent

Phase 3 · Stray node feedback loop  [repeats until clean or ontology_gap HALT]
  StrayNodeAgent → (resolved) KGConsolidationAgent → back to StrayNodeAgent
               → (ontology_gap) HALT — user must refine ontology

Phase 4 · KG completeness loop  [up to MAX_COMPLETENESS_ITERATIONS]
  KGCompletenessJudge
    → complete      → Phase 5
    → needs_improvement:
        _select_agents_from_feedback() maps judge output to the minimal set
        of agents needed.  Each selected agent receives:
          • the source document
          • the EXISTING KG  (new in this version)
          • the judge's feedback  (new in this version)
        so it can both add missing items AND recommend removals in one pass.
        → KGConsolidationAgent (applies additions + removals atomically)
        → back to Phase 3

Phase 5 · Contradiction spotting
  ContradictionAgent → DONE
"""

import json
from agents import (
    people_and_orgs_agent,
    assets_agent,
    transactions_agent,
    kg_consolidation_agent,
    relationship_extraction_agent,
    stray_node_agent,
    kg_curator_agent,
    contradiction_spotting_agent,
)
from schemas import (
    KnowledgeGraph,
    EntityExtractionResult,
    RelationshipExtractionResult,
    KGCuratorResult,
)

MAX_STRAY_NODE_ITERATIONS = 5
MAX_COMPLETENESS_ITERATIONS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Console helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)


def _section(text: str) -> None:
    print(f"\n── {text} {'─' * max(0, 65 - len(text))}")


# ─────────────────────────────────────────────────────────────────────────────
# Agent selection helper
# ─────────────────────────────────────────────────────────────────────────────

def _select_agents_from_feedback(
    judge: KGCuratorResult,
    entity_ontology: dict,
) -> dict[str, bool]:
    """
    Inspect the completeness judge output and return which agents need to run.

    Works with both fixed mock keys ("people", "assets") and arbitrary real
    ontology keys ("Person", "FinancialAccount") by delegating the same
    keyword-matching logic that the agent dispatchers use.

    Mapping
    ───────
    missing entity types matching person/org keywords  → PeopleOrgsAgent
    missing entity types matching asset keywords        → AssetsAgent
    missing entity types matching transaction keywords  → TransactionsAgent
    any missing_relationships                           → RelationshipAgent
    hallucinated items → direct deletion, no agent needed

    Fallback: unrecognised missing type → RelationshipAgent.
    """
    missing_types_raw = {e.entity_type for e in judge.add_entities}
    missing_lower     = {t.lower() for t in missing_types_raw}

    # Keyword sets mirror the dispatchers in agents.py
    PEOPLE_ORG_KW  = {"person", "people", "org", "organisation", "organization", "company", "compan", "bank"}
    ASSET_KW       = {"asset", "account", "holding"}
    TRANSACTION_KW = {"transaction", "transfer", "payment"}

    def _matches(kw_set: set[str]) -> bool:
        return any(any(kw in t for kw in kw_set) for t in missing_lower)

    selection = {
        "people_orgs":   _matches(PEOPLE_ORG_KW),
        "assets":        _matches(ASSET_KW),
        "transactions":  _matches(TRANSACTION_KW),
        "relationships": bool(judge.missing_relationships),
    }

    if not any(selection.values()):
        print("  [Orchestrator] Could not map missing types to specific agents "
              f"(missing types: {missing_types_raw}) — defaulting to RelationshipAgent.")
        selection["relationships"] = True

    chosen = [k for k, v in selection.items() if v]
    print(f"  [Orchestrator] Agents selected for improvement run: {chosen}")
    return selection


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _agents_needed_for_entities(missing_entities: list, entity_ontology: dict) -> dict:
    """Build a stub KGCuratorResult and delegate to _select_agents_from_feedback."""
    from schemas import KGCuratorResult as _KGC
    stub = _KGC(add_entities=missing_entities)
    return _select_agents_from_feedback(stub, entity_ontology)


def run_pipeline(
    document_text: str,
    document_pages: list[str],
    entity_ontology: dict,
    relationship_ontology: dict,
) -> dict:
    """
    Execute the full KG extraction multi-agent pipeline.
    Returns a plain dict (serialisable) with the final KG and reports.
    """
    _banner("KNOWLEDGE GRAPH EXTRACTION PIPELINE — START")

    # ── Phase 1: Entity Extraction ───────────────────────────────────────────
    _banner("PHASE 1 · Entity Extraction")

    _section("Running PeopleOrgsAgent")
    people_orgs: EntityExtractionResult = people_and_orgs_agent(document_text, entity_ontology)

    _section("Running AssetsAgent")
    assets: EntityExtractionResult = assets_agent(document_text, entity_ontology)

    _section("Running TransactionsAgent")
    transactions: EntityExtractionResult = transactions_agent(document_text, entity_ontology)

    _section("KGConsolidationAgent — merging initial entity extractions")
    # Combine all three extraction results into one EntityExtractionResult (flat list).
    combined_entities = EntityExtractionResult(
        entities=[e for r in (people_orgs, assets, transactions) for e in r.entities]
    )
    kg: KnowledgeGraph = kg_consolidation_agent(
        existing_kg=KnowledgeGraph(),
        new_entities=combined_entities,
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
    kg = kg_consolidation_agent(existing_kg=kg, new_relationships=rel_result)

    # ── Phases 3 + 4: Stray node + completeness loop ─────────────────────────
    # Collect ontology gaps across all iterations — pipeline continues regardless
    all_ontology_gaps: list[dict] = []

    for completeness_iter in range(1, MAX_COMPLETENESS_ITERATIONS + 1):

        # ── Phase 3: Stray Node Feedback Loop ────────────────────────────────
        _banner(f"PHASE 3 · Stray Node Detection (completeness iter {completeness_iter})")

        for stray_iter in range(1, MAX_STRAY_NODE_ITERATIONS + 1):
            _section(f"StrayNodeAgent — iteration {stray_iter}")
            stray = stray_node_agent(kg, relationship_ontology, document_text)

            if stray.status == "ontology_gap":
                # Collect gaps and continue — no longer halts execution
                gaps = [g.model_dump(exclude_none=True) for g in stray.ontology_gaps]
                all_ontology_gaps.extend(gaps)
                print(f"  [Orchestrator] Ontology gap(s) recorded ({len(gaps)} new, "
                      f"{len(all_ontology_gaps)} total) — continuing pipeline.")
                # Also merge any relationships resolved alongside the gaps
                if stray.new_relationships:
                    kg = kg_consolidation_agent(
                        existing_kg=kg,
                        new_relationships=RelationshipExtractionResult(
                            relationships=stray.new_relationships
                        ),
                    )
                break  # gaps cannot be resolved without ontology change — move on

            if stray.status == "clean":
                _section("All nodes connected — proceeding to KG Completeness Judge")
                break

            # status == "resolved": add the new relationships
            _section("KGConsolidationAgent — adding resolved stray-node relationships")
            stray_rel_result = RelationshipExtractionResult(
                relationships=stray.new_relationships
            )
            kg = kg_consolidation_agent(existing_kg=kg, new_relationships=stray_rel_result)

        else:
            print(f"  [Orchestrator] ⚠ Stray node loop hit max ({MAX_STRAY_NODE_ITERATIONS}). Continuing.")

        # ── Phase 4: KG Completeness Judge ───────────────────────────────────
        _banner(f"PHASE 4 · KG Curator Agent (iteration {completeness_iter})")

        _section("Running KGCuratorAgent")
        curator: KGCuratorResult = kg_curator_agent(kg, document_pages)

        if curator.status == "complete":
            _section("KG deemed complete — proceeding to Contradiction Spotting")
            break

        # ── Curator-driven improvement — grouped by page ─────────────────────
        _section("KG needs improvement — applying curator actions by page")

        curator_dict = curator.model_dump(exclude_none=True)

        def _page(item) -> int:
            return item.page_number if item.page_number is not None else -1

        def _page_text(pg: int) -> str:
            if pg == -1:
                return document_text
            return document_pages[pg] if pg < len(document_pages) else document_text

        def _page_label(pg: int) -> str:
            return "full doc (no page tag)" if pg == -1 else f"page {pg}"

        # ── ADD: group by page and run targeted extraction agents ─────────────
        add_entity_pages:  dict[int, list] = {}
        add_rel_pages:     dict[int, list] = {}

        for me in curator.add_entities:
            add_entity_pages.setdefault(_page(me), []).append(me.model_dump(exclude_none=True))
        for mr in curator.add_relationships:
            add_rel_pages.setdefault(_page(mr), []).append(mr.model_dump(by_alias=True, exclude_none=True))

        all_entity_parts: list[EntityExtractionResult] = []
        all_rel_parts:    list[RelationshipExtractionResult] = []

        for pg, missing_on_page in add_entity_pages.items():
            pt = _page_text(pg)
            pl = _page_label(pg)
            page_missing = [me for me in curator.add_entities if _page(me) == pg]
            page_sel = _agents_needed_for_entities(page_missing, entity_ontology)
            page_feedback = {**curator_dict, "missing_entities": missing_on_page}

            if page_sel["people_orgs"]:
                _section(f"PeopleOrgsAgent — {pl}")
                all_entity_parts.append(people_and_orgs_agent(pt, entity_ontology,
                    existing_kg=kg, judge_feedback=page_feedback))
            if page_sel["assets"]:
                _section(f"AssetsAgent — {pl}")
                all_entity_parts.append(assets_agent(pt, entity_ontology,
                    existing_kg=kg, judge_feedback=page_feedback))
            if page_sel["transactions"]:
                _section(f"TransactionsAgent — {pl}")
                all_entity_parts.append(transactions_agent(pt, entity_ontology,
                    existing_kg=kg, judge_feedback=page_feedback))

        for pg, missing_on_page in add_rel_pages.items():
            pages_for_call = [_page_text(pg)] if pg != -1 else document_pages
            pl = _page_label(pg)
            page_feedback = {**curator_dict, "missing_relationships": missing_on_page}
            _section(f"RelationshipAgent (add) — {pl}")
            all_rel_parts.append(relationship_extraction_agent(
                document_pages=pages_for_call,
                relationship_ontology=relationship_ontology,
                kg=kg,
                judge_feedback=page_feedback,
            ))

        merged_entity_update: EntityExtractionResult | None = None
        if all_entity_parts:
            merged_entity_update = EntityExtractionResult(
                entities=[e for p in all_entity_parts for e in p.entities],
                entities_to_remove=[eid for p in all_entity_parts for eid in p.entities_to_remove],
            )

        merged_rel_update: RelationshipExtractionResult | None = None
        if all_rel_parts:
            merged_rel_update = RelationshipExtractionResult(
                relationships=[r for p in all_rel_parts for r in p.relationships],
                relationships_to_remove=[],
            )

        # ── REMOVE: collect entity and relationship IDs to drop ───────────────
        remove_entity_ids = [e.entity_id for e in curator.remove_entities]
        # Relationship removals cascade automatically when source/target removed;
        # explicit rel removals handled via entities_to_remove cascade in consolidation
        # For standalone rel removals we pass them as entities_to_remove=[] and
        # handle via update_relationships overwrite — so nothing extra needed here.

        # ── UPDATE: pass patches directly to consolidation ────────────────────
        _section("KGConsolidationAgent — applying add / remove / update actions")
        kg = kg_consolidation_agent(
            existing_kg=kg,
            new_entities=merged_entity_update,
            new_relationships=merged_rel_update,
            entities_to_remove=remove_entity_ids,
            entities_to_update=curator.update_entities or None,
            relationships_to_update=curator.update_relationships or None,
        )

    else:
        print(f"  [Orchestrator] ⚠ Completeness loop hit max ({MAX_COMPLETENESS_ITERATIONS}). Proceeding.")

    # ── Phase 5: Contradiction Spotting ──────────────────────────────────────
    _banner("PHASE 5 · Contradiction Spotting")

    _section("Running ContradictionAgent")
    contradiction_report = contradiction_spotting_agent(kg, document_text)

    # ── Done ──────────────────────────────────────────────────────────────────
    total_entities = len(kg.entities)
    _banner("PIPELINE COMPLETE")
    print(f"\n  Final KG : {total_entities} entities, {len(kg.relationships)} relationships")
    print(f"  Contradictions found: {len(contradiction_report.contradictions)}")

    if all_ontology_gaps:
        print(f"  Ontology gaps recorded: {len(all_ontology_gaps)} "
              f"(see ontology_gap_report.json)")

    return {
        "status": "complete",
        "kg": kg.to_output_format(),
        "contradiction_report": contradiction_report.model_dump(exclude_none=True),
        "ontology_gap_report": {
            "total_gaps": len(all_ontology_gaps),
            "gaps": all_ontology_gaps,
            "recommendation": (
                "The following entities could not be connected using the current "
                "relationship ontology. Consider adding new relationship types to "
                "represent these connections."
            ) if all_ontology_gaps else None,
        },
    }
