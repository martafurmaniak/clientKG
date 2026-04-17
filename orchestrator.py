"""
orchestrator.py — the orchestrator agent that drives the full multi-agent pipeline.

Flow
────
Phase 1 · Entity extraction (parallel-like, sequential calls)
  └─ PeopleOrgsAgent + AssetsAgent + TransactionsAgent → KGConsolidationAgent

Phase 2 · Relationship extraction
  └─ RelationshipAgent (page by page) → KGConsolidationAgent

Phase 3 · Stray node feedback loop
  └─ StrayNodeAgent → (if resolved) KGConsolidationAgent → repeat
     (if ontology_gap) → HALT and surface to user

Phase 4 · KG completeness loop
  └─ KGCompletenessJudge
     → if needs_improvement:
         _select_agents_from_feedback() inspects judge output and decides
         which subset of agents to re-run (people/orgs, assets, transactions,
         relationships) — only agents relevant to the flagged gaps are called.
         → KGConsolidation → remove hallucinations → restart phase 3
     → if complete → ContradictionAgent → DONE
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

MAX_STRAY_NODE_ITERATIONS = 5
MAX_COMPLETENESS_ITERATIONS = 3


def _banner(text: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def _section(text: str) -> None:
    print(f"\n── {text} {'─' * max(0, 65 - len(text))}")


def _select_agents_from_feedback(judge_result: dict) -> dict[str, bool]:
    """
    Inspect the completeness judge's output and decide which agents need to
    be re-run.  Returns a dict of booleans keyed by agent name.

    Logic
    ─────
    • missing_entities whose entity_type is in {people, organisations}
      → run PeopleOrgsAgent
    • missing_entities whose entity_type is "assets"
      → run AssetsAgent
    • missing_entities whose entity_type is "transactions"
      → run TransactionsAgent
    • any missing_relationships present
      → run RelationshipAgent
    • hallucinated_entities are handled by direct deletion (no agent needed)
    • hallucinated_relationships are handled by direct deletion (no agent needed)
    """
    missing_entities      = judge_result.get("missing_entities", [])
    missing_relationships = judge_result.get("missing_relationships", [])

    # Collect the entity types that need attention
    missing_types: set[str] = {e.get("entity_type", "").lower() for e in missing_entities}

    run_people_orgs   = bool(missing_types & {"people", "organisations", "person", "organisation", "organization"})
    run_assets        = bool(missing_types & {"assets", "asset"})
    run_transactions  = bool(missing_types & {"transactions", "transaction"})
    run_relationships = bool(missing_relationships)

    selection = {
        "people_orgs":   run_people_orgs,
        "assets":        run_assets,
        "transactions":  run_transactions,
        "relationships": run_relationships,
    }

    active = [name for name, flag in selection.items() if flag]
    if not active:
        # Judge flagged something but we couldn't map it — run relationships
        # as a safe fallback (covers missing edges from existing nodes).
        print("  [Orchestrator] Could not map missing types to specific agents — defaulting to RelationshipAgent.")
        selection["relationships"] = True

    print(f"  [Orchestrator] Agents selected for improvement run: {[k for k, v in selection.items() if v]}")
    return selection


def run_pipeline(
    document_text: str,
    document_pages: list[str],
    entity_ontology: dict,
    relationship_ontology: dict,
) -> dict:
    """
    Execute the full KG extraction multi-agent pipeline.
    Returns the final knowledge graph and the contradiction report.
    """

    _banner("KNOWLEDGE GRAPH EXTRACTION PIPELINE — START")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Entity Extraction
    # ─────────────────────────────────────────────────────────────────────────
    _banner("PHASE 1 · Entity Extraction")

    _section("Running PeopleOrgsAgent")
    people_orgs = people_and_orgs_agent(document_text, entity_ontology)

    _section("Running AssetsAgent")
    assets = assets_agent(document_text, entity_ontology)

    _section("Running TransactionsAgent")
    transactions = transactions_agent(document_text, entity_ontology)

    # First KG consolidation — merge all entity results
    _section("KGConsolidationAgent — merging entity extractions")
    initial_entities = {**people_orgs, **assets, **transactions}
    kg = kg_consolidation_agent(
        existing_kg={"entities": {}, "relationships": []},
        new_data={"entities": initial_entities, "relationships": []},
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Relationship Extraction
    # ─────────────────────────────────────────────────────────────────────────
    _banner("PHASE 2 · Relationship Extraction")

    _section("Running RelationshipAgent (page by page)")
    new_relationships = relationship_extraction_agent(
        document_pages=document_pages,
        relationship_ontology=relationship_ontology,
        entities=kg["entities"],
    )

    _section("KGConsolidationAgent — merging relationships")
    kg = kg_consolidation_agent(
        existing_kg=kg,
        new_data={"entities": {}, "relationships": new_relationships},
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: run phases 3 + 4 in a loop (completeness improvements restart them)
    # ─────────────────────────────────────────────────────────────────────────
    for completeness_iter in range(1, MAX_COMPLETENESS_ITERATIONS + 1):

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 3 — Stray Node Feedback Loop
        # ─────────────────────────────────────────────────────────────────────
        _banner(f"PHASE 3 · Stray Node Detection (completeness iteration {completeness_iter})")

        for stray_iter in range(1, MAX_STRAY_NODE_ITERATIONS + 1):
            _section(f"StrayNodeAgent — iteration {stray_iter}")
            stray_result = stray_node_agent(kg, relationship_ontology, document_text)

            if stray_result["status"] == "ontology_gap":
                _banner("⚠  PIPELINE HALTED — Ontology Gap Detected")
                print("\nThe stray node agent found entities whose relationships cannot be")
                print("represented with the current relationship ontology.\n")
                print("Please review the findings below, refine your relationship ontology,")
                print("and re-run the pipeline.\n")
                print("STRAY NODE AGENT OUTPUT:")
                print(json.dumps(stray_result, indent=2))
                return {
                    "status": "halted_ontology_gap",
                    "kg": kg,
                    "stray_node_report": stray_result,
                }

            if stray_result["status"] == "clean":
                _section("All nodes connected — proceeding to KG Completeness Judge")
                break

            if stray_result["status"] == "resolved":
                _section("KGConsolidationAgent — adding resolved stray-node relationships")
                kg = kg_consolidation_agent(
                    existing_kg=kg,
                    new_data={"entities": {}, "relationships": stray_result["new_relationships"]},
                )

        else:
            print(f"\n  [Orchestrator] ⚠ Stray node loop hit max iterations ({MAX_STRAY_NODE_ITERATIONS}). Continuing.")

        # ─────────────────────────────────────────────────────────────────────
        # PHASE 4 — KG Completeness Judge
        # ─────────────────────────────────────────────────────────────────────
        _banner(f"PHASE 4 · KG Completeness Judge (iteration {completeness_iter})")

        _section("Running KGCompletenessJudge")
        judge_result = kg_completeness_judge(kg, document_text)

        if judge_result["status"] == "complete":
            _section("KG deemed complete — proceeding to Contradiction Spotting")
            break

        # ── KG needs improvement: selectively re-run only relevant agents ──
        _section("KG needs improvement — selecting agents based on judge feedback")

        # Decide which agents are actually needed
        agent_selection = _select_agents_from_feedback(judge_result)

        # Build a focused prompt that includes the judge's specific feedback
        improvement_notes = json.dumps({
            "missing_entities":           judge_result.get("missing_entities", []),
            "missing_relationships":      judge_result.get("missing_relationships", []),
            "hallucinated_entities":      judge_result.get("hallucinated_entities", []),
            "hallucinated_relationships": judge_result.get("hallucinated_relationships", []),
        }, indent=2)

        improvement_doc = (
            f"IMPROVEMENT FEEDBACK FROM COMPLETENESS JUDGE:\n{improvement_notes}\n\n"
            f"ORIGINAL DOCUMENT:\n{document_text}"
        )

        all_new_entities: dict = {}
        rels_update: list = []

        if agent_selection["people_orgs"]:
            _section("Re-running PeopleOrgsAgent (missing people/org entities flagged)")
            people_orgs_update = people_and_orgs_agent(improvement_doc, entity_ontology)
            all_new_entities.update(people_orgs_update)
        else:
            print("  [Orchestrator] Skipping PeopleOrgsAgent — no missing people/org entities.")

        if agent_selection["assets"]:
            _section("Re-running AssetsAgent (missing asset entities flagged)")
            assets_update = assets_agent(improvement_doc, entity_ontology)
            all_new_entities.update(assets_update)
        else:
            print("  [Orchestrator] Skipping AssetsAgent — no missing asset entities.")

        if agent_selection["transactions"]:
            _section("Re-running TransactionsAgent (missing transaction entities flagged)")
            transactions_update = transactions_agent(improvement_doc, entity_ontology)
            all_new_entities.update(transactions_update)
        else:
            print("  [Orchestrator] Skipping TransactionsAgent — no missing transaction entities.")

        if agent_selection["relationships"]:
            _section("Re-running RelationshipAgent (missing relationships flagged)")
            rels_update = relationship_extraction_agent(
                document_pages=[improvement_doc],
                relationship_ontology=relationship_ontology,
                entities=kg["entities"],
            )
        else:
            print("  [Orchestrator] Skipping RelationshipAgent — no missing relationships.")

        _section("KGConsolidationAgent — merging targeted improvements")
        kg = kg_consolidation_agent(
            existing_kg=kg,
            new_data={"entities": all_new_entities, "relationships": rels_update},
        )

        # Also remove hallucinations if the judge flagged any
        hallucinated_entity_ids = {e["entity_id"] for e in judge_result.get("hallucinated_entities", [])}
        hallucinated_rel_ids    = {r["rel_id"]    for r in judge_result.get("hallucinated_relationships", [])}

        if hallucinated_entity_ids or hallucinated_rel_ids:
            _section("Removing hallucinated nodes/edges flagged by judge")
            for etype, elist in kg["entities"].items():
                kg["entities"][etype] = [e for e in elist if e["id"] not in hallucinated_entity_ids]
            kg["relationships"] = [r for r in kg["relationships"] if r["id"] not in hallucinated_rel_ids]
            print(f"  Removed {len(hallucinated_entity_ids)} entity(s) and {len(hallucinated_rel_ids)} relationship(s)")

    else:
        print(f"\n  [Orchestrator] ⚠ Completeness loop hit max iterations ({MAX_COMPLETENESS_ITERATIONS}). Proceeding.")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5 — Contradiction Spotting
    # ─────────────────────────────────────────────────────────────────────────
    _banner("PHASE 5 · Contradiction Spotting")

    _section("Running ContradictionAgent")
    contradiction_report = contradiction_spotting_agent(kg, document_text)

    # ─────────────────────────────────────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────────────────────────────────────
    _banner("PIPELINE COMPLETE")
    print(f"\n  Final KG: {sum(len(v) for v in kg['entities'].values())} entities, "
          f"{len(kg['relationships'])} relationships")
    print(f"  Contradictions found: {len(contradiction_report['contradictions'])}")

    return {
        "status": "complete",
        "kg": kg,
        "contradiction_report": contradiction_report,
    }
