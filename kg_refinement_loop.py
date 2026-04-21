"""
kg_refinement_loop.py — shared KG refinement loop.

Contains the stray-node feedback loop and KG curator loop that are run
after initial extraction — whether from a client history document or a
corroboration document.

Both pipelines call run_refinement_loop() with their KG and document
context.  All logic lives here once; neither pipeline duplicates it.

Interface
─────────
run_refinement_loop(
    kg,
    document_text,        # full concatenated plain text (for stray-node agent)
    document_pages,       # list of page strings (for curator + improvement agents)
    entity_ontology,
    relationship_ontology,
    label,                # short string prefix for console output e.g. "ClientHistory"
) -> RefinementResult(kg, ontology_gap_report)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents import (
    people_and_orgs_agent,
    assets_agent,
    transactions_agent,
    kg_consolidation_agent,
    relationship_extraction_agent,
    stray_node_agent,
    kg_curator_agent,
)
from schemas import (
    KnowledgeGraph,
    EntityExtractionResult,
    RelationshipExtractionResult,
    KGCuratorResult,
)

MAX_STRAY_NODE_ITERATIONS  = 5
MAX_COMPLETENESS_ITERATIONS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RefinementResult:
    kg:                  KnowledgeGraph
    ontology_gap_report: dict = field(default_factory=dict)


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
# Agent selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _select_agents_from_feedback(
    curator: KGCuratorResult,
    entity_ontology: dict,
) -> dict[str, bool]:
    """Map curator add_entities to the specific agents that should handle them."""
    missing_types_raw = {e.entity_type for e in curator.add_entities}
    missing_lower     = {t.lower() for t in missing_types_raw}

    PEOPLE_ORG_KW  = {"person", "people", "org", "organisation", "organization", "company", "compan", "bank"}
    ASSET_KW       = {"asset", "account", "holding"}
    TRANSACTION_KW = {"transaction", "transfer", "payment", "corporate", "event"}

    def _matches(kw_set: set[str]) -> bool:
        return any(any(kw in t for kw in kw_set) for t in missing_lower)

    selection = {
        "people_orgs":   _matches(PEOPLE_ORG_KW),
        "assets":        _matches(ASSET_KW),
        "transactions":  _matches(TRANSACTION_KW),
        "relationships": bool(curator.add_relationships),
    }

    if not any(selection.values()):
        print("  [Refinement] Could not map missing types to specific agents "
              f"({missing_types_raw}) — defaulting to RelationshipAgent.")
        selection["relationships"] = True

    chosen = [k for k, v in selection.items() if v]
    print(f"  [Refinement] Agents selected for improvement run: {chosen}")
    return selection


def _agents_needed_for_entities(missing_entities: list, entity_ontology: dict) -> dict:
    stub = KGCuratorResult(add_entities=missing_entities)
    return _select_agents_from_feedback(stub, entity_ontology)


# ─────────────────────────────────────────────────────────────────────────────
# Core refinement loop
# ─────────────────────────────────────────────────────────────────────────────

def run_refinement_loop(
    kg: KnowledgeGraph,
    document_text: str,
    document_pages: list[str],
    entity_ontology: dict,
    relationship_ontology: dict,
    label: str = "",
) -> RefinementResult:
    """
    Run the stray-node feedback loop and KG curator loop on an already-extracted KG.

    Parameters
    ──────────
    kg                    : the KG produced by initial extraction
    document_text         : full concatenated plain text — used by stray-node agent
    document_pages        : per-page strings — used by curator + improvement agents
    entity_ontology       : shared ontology dict
    relationship_ontology : shared ontology dict
    label                 : short string shown in console banners e.g. "ClientHistory"
                            or "BankStatement"

    Returns
    ───────
    RefinementResult with the refined KG and an ontology_gap_report dict.
    """
    prefix = f"[{label}] " if label else ""
    all_ontology_gaps: list[dict] = []

    def _page(item) -> int:
        return item.page_number if item.page_number is not None else -1

    def _page_text(pg: int) -> str:
        if pg == -1:
            return document_text
        return document_pages[pg] if pg < len(document_pages) else document_text

    def _page_label(pg: int) -> str:
        return "full doc (no page tag)" if pg == -1 else f"page {pg}"

    for completeness_iter in range(1, MAX_COMPLETENESS_ITERATIONS + 1):

        # ── Stray Node Feedback Loop ──────────────────────────────────────────
        _banner(f"{prefix}Stray Node Detection (iteration {completeness_iter})")

        for stray_iter in range(1, MAX_STRAY_NODE_ITERATIONS + 1):
            _section(f"StrayNodeAgent — iteration {stray_iter}")
            stray = stray_node_agent(kg, relationship_ontology, document_text)

            if stray.status == "ontology_gap":
                gaps = [g.model_dump(exclude_none=True) for g in stray.ontology_gaps]
                all_ontology_gaps.extend(gaps)
                print(f"  {prefix}Ontology gap(s) recorded ({len(gaps)} new, "
                      f"{len(all_ontology_gaps)} total) — continuing.")
                if stray.new_relationships:
                    kg = kg_consolidation_agent(
                        existing_kg=kg,
                        new_relationships=RelationshipExtractionResult(
                            relationships=stray.new_relationships
                        ),
                    )
                break

            if stray.status == "clean":
                _section("All nodes connected — proceeding to KG Curator")
                break

            _section("KGConsolidationAgent — adding resolved stray-node relationships")
            kg = kg_consolidation_agent(
                existing_kg=kg,
                new_relationships=RelationshipExtractionResult(
                    relationships=stray.new_relationships
                ),
            )
        else:
            print(f"  {prefix}⚠ Stray node loop hit max ({MAX_STRAY_NODE_ITERATIONS}).")

        # ── KG Curator Loop ───────────────────────────────────────────────────
        _banner(f"{prefix}KG Curator Agent (iteration {completeness_iter})")

        _section("Running KGCuratorAgent")
        curator: KGCuratorResult = kg_curator_agent(kg, document_pages)

        if curator.status == "complete":
            _section("KG deemed complete")
            break

        # ── ADD: group by page, run targeted extraction agents ────────────────
        _section("KG needs improvement — applying curator actions by page")
        curator_dict = curator.model_dump(exclude_none=True)

        add_entity_pages: dict[int, list] = {}
        add_rel_pages:    dict[int, list] = {}

        for me in curator.add_entities:
            add_entity_pages.setdefault(_page(me), []).append(me.model_dump(exclude_none=True))
        for mr in curator.add_relationships:
            add_rel_pages.setdefault(_page(mr), []).append(
                mr.model_dump(by_alias=True, exclude_none=True)
            )

        all_entity_parts: list[EntityExtractionResult] = []
        all_rel_parts:    list[RelationshipExtractionResult] = []

        for pg, missing_on_page in add_entity_pages.items():
            pt = _page_text(pg)
            pl = _page_label(pg)
            page_missing = [me for me in curator.add_entities if _page(me) == pg]
            page_sel     = _agents_needed_for_entities(page_missing, entity_ontology)
            page_feedback = {**curator_dict, "missing_entities": missing_on_page}

            if page_sel["people_orgs"]:
                _section(f"PeopleOrgsAgent — {pl}")
                all_entity_parts.append(
                    people_and_orgs_agent(pt, entity_ontology,
                                          existing_kg=kg, judge_feedback=page_feedback)
                )
            if page_sel["assets"]:
                _section(f"AssetsAgent — {pl}")
                all_entity_parts.append(
                    assets_agent(pt, entity_ontology,
                                 existing_kg=kg, judge_feedback=page_feedback)
                )
            if page_sel["transactions"]:
                _section(f"TransactionsAgent — {pl}")
                all_entity_parts.append(
                    transactions_agent(pt, entity_ontology,
                                       existing_kg=kg, judge_feedback=page_feedback)
                )

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

        remove_entity_ids = [e.entity_id for e in curator.remove_entities]

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
        print(f"  {prefix}⚠ Curator loop hit max ({MAX_COMPLETENESS_ITERATIONS}). Proceeding.")

    gap_report = {
        "total_gaps": len(all_ontology_gaps),
        "gaps": all_ontology_gaps,
        "recommendation": (
            "The following entities could not be connected using the current "
            "relationship ontology. Consider adding new relationship types."
        ) if all_ontology_gaps else None,
    }

    return RefinementResult(kg=kg, ontology_gap_report=gap_report)
