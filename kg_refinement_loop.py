"""
kg_refinement_loop.py — shared KG refinement loop.

Improvements in this version
─────────────────────────────
Fix 1  : Curator runs BEFORE stray-node check. Clean the graph first,
         then verify connectivity. Prevents wasting stray-node calls on
         nodes the curator was about to remove.

Fix 4  : Distinguish "missing entity" (full extraction) vs "incomplete entity"
         (attribute patch only). update_entities from the curator go directly to
         consolidation without re-running an extraction agent.

Fix 5  : _agents_needed_for_entities / _select_agents_from_feedback simplified
         into a single clean function with no stub-object workaround.

Fix 9  : Early exit if a curator iteration produced zero net change to the KG
         (entity + relationship count delta == 0 and no removals/updates).

Fix 10 : curator.remove_relationships now correctly passed to consolidation.
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

MAX_STRAY_NODE_ITERATIONS   = 5
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
# Fix 5: simplified agent selection
# ─────────────────────────────────────────────────────────────────────────────

_PEOPLE_ORG_KW  = frozenset({"person", "people", "org", "organisation",
                               "organization", "company", "compan", "bank"})
_ASSET_KW       = frozenset({"asset", "account", "holding"})
_TRANSACTION_KW = frozenset({"transaction", "transfer", "payment",
                               "corporate", "event"})


def _select_agents(missing_entities: list, missing_relationships: list) -> dict[str, bool]:
    """
    Given lists of AddEntity and AddRelationship items from the curator,
    return which extraction agents need to run.
    No stub objects, no intermediate helpers.
    """
    missing_lower = {e.entity_type.lower() for e in missing_entities}

    def _hits(kw_set: frozenset) -> bool:
        return any(any(kw in t for kw in kw_set) for t in missing_lower)

    selection = {
        "people_orgs":   _hits(_PEOPLE_ORG_KW),
        "assets":        _hits(_ASSET_KW),
        "transactions":  _hits(_TRANSACTION_KW),
        "relationships": bool(missing_relationships),
    }

    if missing_entities and not any(selection.values()):
        print("  [Refinement] Could not map missing types to agents "
              f"({[e.entity_type for e in missing_entities]}) — defaulting to RelationshipAgent.")
        selection["relationships"] = True

    chosen = [k for k, v in selection.items() if v]
    print(f"  [Refinement] Agents selected: {chosen}")
    return selection


# ─────────────────────────────────────────────────────────────────────────────
# Fix 9: change detection
# ─────────────────────────────────────────────────────────────────────────────

def _kg_signature(kg: KnowledgeGraph) -> tuple[int, int]:
    """Return (entity_count, relationship_count) for change detection."""
    return len(kg.entities), len(kg.relationships)


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
    Run the curator loop followed by stray-node check on an already-extracted KG.

    Order per iteration (Fix 1):
      1. KG Curator — clean, add, remove, update
      2. Stray node check — verify all nodes are connected

    Early exit (Fix 9) if a curator iteration produces zero net change.
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

        sig_before = _kg_signature(kg)

        # ── Fix 1: Curator runs FIRST ─────────────────────────────────────────
        _banner(f"{prefix}KG Curator Agent (iteration {completeness_iter})")
        _section("Running KGCuratorAgent")

        curator: KGCuratorResult = kg_curator_agent(
            kg, document_pages,
            entity_ontology=entity_ontology,
            relationship_ontology=relationship_ontology,
        )

        if curator.status == "complete":
            _section("KG deemed complete — running final stray-node check")
            # Still run one stray-node pass to catch any connectivity issues
            # before declaring the graph done
            _final_stray = stray_node_agent(kg, relationship_ontology, document_text)
            if _final_stray.status != "clean":
                if _final_stray.new_relationships:
                    kg = kg_consolidation_agent(
                        existing_kg=kg,
                        new_relationships=RelationshipExtractionResult(
                            relationships=_final_stray.new_relationships
                        ),
                    )
                if _final_stray.ontology_gaps:
                    all_ontology_gaps.extend(
                        g.model_dump(exclude_none=True)
                        for g in _final_stray.ontology_gaps
                    )
            break

        # ── Apply curator actions ─────────────────────────────────────────────
        _section("KG needs improvement — applying curator actions by page")
        curator_dict = curator.model_dump(exclude_none=True)

        # Fix 4: route update_entities directly to consolidation — no agent call needed
        # Only "add" items require an extraction agent re-run
        add_entity_pages:  dict[int, list] = {}
        add_rel_pages:     dict[int, list] = {}

        for me in curator.add_entities:
            add_entity_pages.setdefault(_page(me), []).append(me.model_dump(exclude_none=True))
        for mr in curator.add_relationships:
            add_rel_pages.setdefault(_page(mr), []).append(
                mr.model_dump(by_alias=True, exclude_none=True)
            )

        all_entity_parts: list[EntityExtractionResult] = []
        all_rel_parts:    list[RelationshipExtractionResult] = []

        # Entity additions — grouped by page
        for pg, missing_on_page in add_entity_pages.items():
            pt = _page_text(pg)
            pl = _page_label(pg)
            # Fix 5: direct call to simplified agent selector
            page_missing = [me for me in curator.add_entities if _page(me) == pg]
            sel = _select_agents(page_missing, [])
            page_feedback = {**curator_dict, "missing_entities": missing_on_page}

            if sel["people_orgs"]:
                _section(f"PeopleOrgsAgent — {pl}")
                all_entity_parts.append(
                    people_and_orgs_agent(pt, entity_ontology,
                                          existing_kg=kg, judge_feedback=page_feedback,
                                          id_seed_kg=kg)
                )
            if sel["assets"]:
                _section(f"AssetsAgent — {pl}")
                all_entity_parts.append(
                    assets_agent(pt, entity_ontology,
                                 existing_kg=kg, judge_feedback=page_feedback,
                                 id_seed_kg=kg)
                )
            if sel["transactions"]:
                _section(f"TransactionsAgent — {pl}")
                all_entity_parts.append(
                    transactions_agent(pt, entity_ontology,
                                       existing_kg=kg, judge_feedback=page_feedback,
                                       id_seed_kg=kg)
                )

        # Relationship additions — grouped by page
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

        # Fix 10: pass remove_relationships to consolidation
        remove_entity_ids = [e.entity_id for e in curator.remove_entities]
        remove_rel_keys   = [
            (r.source, r.target, r.type) for r in curator.remove_relationships
        ]

        _section("KGConsolidationAgent — applying add / remove / update (Fix 4 + Fix 10)")
        kg = kg_consolidation_agent(
            existing_kg=kg,
            new_entities=merged_entity_update,
            new_relationships=merged_rel_update,
            entities_to_remove=remove_entity_ids,
            # Fix 10: relationship removals by (source, target, type) key
            relationship_keys_to_remove=remove_rel_keys if remove_rel_keys else None,
            # Fix 4: update_entities go straight to consolidation, no agent re-run
            entities_to_update=curator.update_entities or None,
            relationships_to_update=curator.update_relationships or None,
        )

        # ── Fix 9: early exit if no net change ───────────────────────────────
        sig_after = _kg_signature(kg)
        had_removals = bool(remove_entity_ids or remove_rel_keys)
        had_updates  = bool(curator.update_entities or curator.update_relationships)
        if sig_after == sig_before and not had_removals and not had_updates:
            print(f"  {prefix}No net change in iteration {completeness_iter} — "
                  "exiting curator loop early.")
            break

        # ── Stray node check AFTER curator (Fix 1) ────────────────────────────
        _banner(f"{prefix}Stray Node Check (after curator iteration {completeness_iter})")

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
                _section("All nodes connected")
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

    else:
        print(f"  {prefix}⚠ Curator loop hit max ({MAX_COMPLETENESS_ITERATIONS}). Proceeding.")

    # Final stray-node pass if loop exited via max iterations
    gap_report = {
        "total_gaps": len(all_ontology_gaps),
        "gaps": all_ontology_gaps,
        "recommendation": (
            "The following entities could not be connected using the current "
            "relationship ontology. Consider adding new relationship types."
        ) if all_ontology_gaps else None,
    }

    return RefinementResult(kg=kg, ontology_gap_report=gap_report)
