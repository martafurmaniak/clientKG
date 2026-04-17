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
    kg_completeness_judge,
    contradiction_spotting_agent,
)
from schemas import (
    KnowledgeGraph,
    EntityExtractionResult,
    RelationshipExtractionResult,
    KGCompletenessResult,
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
    judge: KGCompletenessResult,
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
    missing_types_raw = {e.entity_type for e in judge.missing_entities}
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
    # Combine all three extraction results into one EntityExtractionResult
    combined_entities = EntityExtractionResult(
        people=people_orgs.people,
        organisations=people_orgs.organisations,
        assets=assets.assets,
        transactions=transactions.transactions,
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
    for completeness_iter in range(1, MAX_COMPLETENESS_ITERATIONS + 1):

        # ── Phase 3: Stray Node Feedback Loop ────────────────────────────────
        _banner(f"PHASE 3 · Stray Node Detection (completeness iter {completeness_iter})")

        for stray_iter in range(1, MAX_STRAY_NODE_ITERATIONS + 1):
            _section(f"StrayNodeAgent — iteration {stray_iter}")
            stray = stray_node_agent(kg, relationship_ontology, document_text)

            if stray.status == "ontology_gap":
                _banner("⚠  PIPELINE HALTED — Ontology Gap Detected")
                print("\nEntities found whose relationships cannot be represented")
                print("with the current ontology. Please refine it and re-run.\n")
                print("STRAY NODE AGENT OUTPUT:")
                print(json.dumps(stray.model_dump(exclude_none=True), indent=2))
                return {
                    "status": "halted_ontology_gap",
                    "kg": kg.to_serialisable(),
                    "stray_node_report": stray.model_dump(exclude_none=True),
                }

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
        _banner(f"PHASE 4 · KG Completeness Judge (iteration {completeness_iter})")

        _section("Running KGCompletenessJudge")
        judge: KGCompletenessResult = kg_completeness_judge(kg, document_text)

        if judge.status == "complete":
            _section("KG deemed complete — proceeding to Contradiction Spotting")
            break

        # ── Targeted improvement run ──────────────────────────────────────────
        _section("KG needs improvement — selecting agents based on judge feedback")
        agent_selection = _select_agents_from_feedback(judge, entity_ontology)

        # Serialise the judge feedback once for agent prompts
        judge_feedback_dict = judge.model_dump(exclude_none=True)

        new_entities_parts: list[EntityExtractionResult] = []
        rel_improvement: RelationshipExtractionResult | None = None

        if agent_selection["people_orgs"]:
            _section("Re-running PeopleOrgsAgent with existing KG + judge feedback")
            update = people_and_orgs_agent(
                document_text, entity_ontology,
                existing_kg=kg, judge_feedback=judge_feedback_dict,
            )
            new_entities_parts.append(update)
        else:
            print("  [Orchestrator] Skipping PeopleOrgsAgent — no missing people/org entities.")

        if agent_selection["assets"]:
            _section("Re-running AssetsAgent with existing KG + judge feedback")
            update = assets_agent(
                document_text, entity_ontology,
                existing_kg=kg, judge_feedback=judge_feedback_dict,
            )
            new_entities_parts.append(update)
        else:
            print("  [Orchestrator] Skipping AssetsAgent — no missing asset entities.")

        if agent_selection["transactions"]:
            _section("Re-running TransactionsAgent with existing KG + judge feedback")
            update = transactions_agent(
                document_text, entity_ontology,
                existing_kg=kg, judge_feedback=judge_feedback_dict,
            )
            new_entities_parts.append(update)
        else:
            print("  [Orchestrator] Skipping TransactionsAgent — no missing transaction entities.")

        if agent_selection["relationships"]:
            _section("Re-running RelationshipAgent with existing KG + judge feedback")
            rel_improvement = relationship_extraction_agent(
                document_pages=document_pages,
                relationship_ontology=relationship_ontology,
                kg=kg,
                judge_feedback=judge_feedback_dict,
            )
        else:
            print("  [Orchestrator] Skipping RelationshipAgent — no missing relationships.")

        # Merge all entity update parts into one
        merged_entity_update: EntityExtractionResult | None = None
        if new_entities_parts:
            merged_entity_update = EntityExtractionResult(
                people       =[e for p in new_entities_parts for e in p.people],
                organisations=[e for p in new_entities_parts for e in p.organisations],
                assets       =[e for p in new_entities_parts for e in p.assets],
                transactions =[e for p in new_entities_parts for e in p.transactions],
                entities_to_remove=[
                    eid for p in new_entities_parts for eid in p.entities_to_remove
                ],
            )

        # Also collect hallucinations flagged directly by the judge
        hallucinated_entity_ids = [e.entity_id for e in judge.hallucinated_entities]
        hallucinated_rel_ids    = [r.rel_id    for r in judge.hallucinated_relationships]

        _section("KGConsolidationAgent — merging targeted improvements + removals")
        kg = kg_consolidation_agent(
            existing_kg=kg,
            new_entities=merged_entity_update,
            new_relationships=rel_improvement,
            entities_to_remove=hallucinated_entity_ids,
            relationships_to_remove=hallucinated_rel_ids,
        )

    else:
        print(f"  [Orchestrator] ⚠ Completeness loop hit max ({MAX_COMPLETENESS_ITERATIONS}). Proceeding.")

    # ── Phase 5: Contradiction Spotting ──────────────────────────────────────
    _banner("PHASE 5 · Contradiction Spotting")

    _section("Running ContradictionAgent")
    contradiction_report = contradiction_spotting_agent(kg, document_text)

    # ── Done ──────────────────────────────────────────────────────────────────
    total_entities = sum([
        len(kg.entities.people), len(kg.entities.organisations),
        len(kg.entities.assets), len(kg.entities.transactions),
    ])
    _banner("PIPELINE COMPLETE")
    print(f"\n  Final KG : {total_entities} entities, {len(kg.relationships)} relationships")
    print(f"  Contradictions found: {len(contradiction_report.contradictions)}")

    return {
        "status": "complete",
        "kg": kg.to_serialisable(),
        "contradiction_report": contradiction_report.model_dump(exclude_none=True),
    }
