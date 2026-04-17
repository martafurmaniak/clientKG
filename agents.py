"""
agents.py — all sub-agent implementations.

Works with both mock (fixed entity types) and real (arbitrary ontology keys)
inputs, because the KnowledgeGraph now stores entities as dict[str, list[Entity]]
rather than named fields.
"""

from __future__ import annotations

import json
from llm_utils import call_llm, parse_and_validate
from schemas import (
    KnowledgeGraph,
    EntityExtractionResult,
    RelationshipExtractionResult,
    StrayNodeResult,
    KGCompletenessResult,
    ContradictionResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kg_json(kg: KnowledgeGraph) -> str:
    return json.dumps(kg.to_serialisable(), indent=2)


def _j(obj: object) -> str:
    return json.dumps(obj, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Entity extraction agents  (initial + improvement runs)
# ─────────────────────────────────────────────────────────────────────────────

def _entity_extraction_agent(
    agent_name: str,
    entity_types: list[str],
    document_text: str,
    entity_ontology: dict,
    existing_kg: KnowledgeGraph | None = None,
    judge_feedback: dict | None = None,
) -> EntityExtractionResult:
    """
    Generic entity extractor.

    Initial run  — extracts all entities of `entity_types` from `document_text`.
    Improvement  — receives existing KG + judge feedback; returns only the
                   delta (new entities to add, IDs to remove).
    """
    ontology_subset = {k: v for k, v in entity_ontology.items() if k in entity_types}
    is_improvement  = existing_kg is not None and judge_feedback is not None

    # Build an id→type lookup so the agent knows which IDs belong to which type
    existing_ids_by_type = {
        etype: [e.to_flat_dict() for e in elist]
        for etype, elist in existing_kg.entities.items()
        if etype in entity_types
    } if existing_kg else {}

    if is_improvement:
        relevant_missing = [
            e for e in judge_feedback.get("missing_entities", [])
            if e.get("entity_type", "") in entity_types
        ]
        relevant_hallucinated = judge_feedback.get("hallucinated_entities", [])

        system_prompt = (
            "You are a precise knowledge-graph entity correction engine. "
            "Return ONLY a valid JSON object — no markdown, no explanations."
        )
        user_prompt = f"""Correct the knowledge graph for entity types: {entity_types}.

ENTITY ONTOLOGY:
{_j(ontology_subset)}

EXISTING ENTITIES IN KG (for these types):
{_j(existing_ids_by_type)}

JUDGE FEEDBACK — missing entities to ADD:
{_j(relevant_missing)}

JUDGE FEEDBACK — hallucinated entity IDs to REMOVE:
{_j(relevant_hallucinated)}

SOURCE DOCUMENT:
{document_text}

Return a JSON object with:
  - One key per entity type in {entity_types} → list of NEW entity objects to add.
    Each entity needs an "id" that does not clash with existing IDs above.
    Include all ontology attributes (use null if unknown).
  - "entities_to_remove": list of entity ID strings to drop from the KG.

Only include entities that need to change. Do NOT re-list already-correct entities.

Example (for entity types ["Person"]):
{{
  "Person": [{{"id": "person_3", "fullName": "New Person", "dateOfBirth": null}}],
  "entities_to_remove": ["person_99"]
}}
"""
    else:
        system_prompt = (
            "You are a precise knowledge-graph entity extraction engine. "
            "Return ONLY a valid JSON object — no markdown, no explanations. "
            "Use exactly the entity type keys provided in the ontology."
        )
        user_prompt = f"""Extract all entities of the following types from the document.

ENTITY TYPES AND THEIR ATTRIBUTES (ontology):
{_j(ontology_subset)}

DOCUMENT:
{document_text}

Return a JSON object where each key is an entity type from the ontology,
and the value is a list of entity objects containing:
  - "id"  : unique string (e.g. "person_1", "org_1") — use snake_case prefix matching the type
  - one key per attribute defined in the ontology (use null if unknown)

Example (for entity types ["Person", "Organisation"]):
{{
  "Person": [{{"id": "person_1", "fullName": "John Smith", "dateOfBirth": null}}],
  "Organisation": [{{"id": "org_1", "name": "Alpine Bank", "type": "bank"}}]
}}
"""

    raw    = call_llm(system_prompt, user_prompt, label=agent_name)
    result = parse_and_validate(raw, EntityExtractionResult, label=agent_name)

    mode   = "improvement" if is_improvement else "initial"
    counts = {etype: len(elist) for etype, elist in result.entities.items()}
    print(f"  [{agent_name}] {mode} → entities={counts}  to_remove={result.entities_to_remove}")
    return result


# ── Type-classification keyword sets ─────────────────────────────────────────
# Substrings matched case-insensitively against ontology key names.
# Covers the mock ontology (people, organisations, assets, transactions)
# AND the real ontology (Person, Organization, Client Profile,
# Asset, Account, Transaction, CorporateEvent).

_PEOPLE_ORG_KW: frozenset = frozenset({
    "people", "person",
    "organization", "organisation",
    "compan", "bank", "fund",
    "client profile", "client_profile",
    "profile", "role",
})

_ASSET_KW: frozenset = frozenset({
    "asset", "assets",
    "account",
    "holding", "portfolio", "security", "instrument",
})

_TRANSACTION_KW: frozenset = frozenset({
    "transaction", "transactions",
    "transfer", "payment",
    "corporate", "event",
})


def _matches_any(key: str, kw_set: frozenset) -> bool:
    k = key.lower()
    return any(kw in k for kw in kw_set)


def _pick_types(ontology: dict, kw_set: frozenset) -> list:
    return [k for k in ontology if _matches_any(k, kw_set)]


def _people_org_types(ontology: dict) -> list:
    """Used by orchestrator agent-selection; mirrors people_and_orgs_agent."""
    return _pick_types(ontology, _PEOPLE_ORG_KW)


def people_and_orgs_agent(
    document_text: str,
    entity_ontology: dict,
    existing_kg: KnowledgeGraph | None = None,
    judge_feedback: dict | None = None,
) -> EntityExtractionResult:
    types = _pick_types(entity_ontology, _PEOPLE_ORG_KW)
    if not types:
        types = [k for k in entity_ontology
                 if not _matches_any(k, _ASSET_KW) and not _matches_any(k, _TRANSACTION_KW)]
    return _entity_extraction_agent("PeopleOrgsAgent", types, document_text, entity_ontology,
                                     existing_kg, judge_feedback)


def assets_agent(
    document_text: str,
    entity_ontology: dict,
    existing_kg: KnowledgeGraph | None = None,
    judge_feedback: dict | None = None,
) -> EntityExtractionResult:
    types = _pick_types(entity_ontology, _ASSET_KW)
    return _entity_extraction_agent("AssetsAgent", types, document_text, entity_ontology,
                                     existing_kg, judge_feedback)


def transactions_agent(
    document_text: str,
    entity_ontology: dict,
    existing_kg: KnowledgeGraph | None = None,
    judge_feedback: dict | None = None,
) -> EntityExtractionResult:
    types = _pick_types(entity_ontology, _TRANSACTION_KW)
    return _entity_extraction_agent("TransactionsAgent", types, document_text, entity_ontology,
                                     existing_kg, judge_feedback)


# ─────────────────────────────────────────────────────────────────────────────
# 2. KG Consolidation agent
# ─────────────────────────────────────────────────────────────────────────────

def kg_consolidation_agent(
    existing_kg: KnowledgeGraph,
    new_entities: EntityExtractionResult | None = None,
    new_relationships: RelationshipExtractionResult | None = None,
    entities_to_remove: list[str] | None = None,
    relationships_to_remove: list[str] | None = None,
) -> KnowledgeGraph:
    """
    Pure deterministic merge — no LLM call.

    Operations performed in order:
      1. Append new entities under their type key (keyed by id, no duplicates).
      2. Append new relationships (keyed by id, no duplicates).
      3. Remove entities by ID (+ cascade to any relationship referencing them).
      4. Remove relationships by ID.
    """
    # Work on a copy so we never mutate the caller's object
    entities: dict[str, list] = {
        etype: list(elist)
        for etype, elist in existing_kg.entities.items()
    }
    relationships: list = list(existing_kg.relationships)

    # ── 1. Add new entities ──────────────────────────────────────────────────
    if new_entities:
        existing_ids = {e.id for elist in entities.values() for e in elist}
        for etype, elist in new_entities.entities.items():
            bucket = entities.setdefault(etype, [])
            for entity in elist:
                if entity.id not in existing_ids:
                    bucket.append(entity)
                    existing_ids.add(entity.id)
                # silently skip exact-ID duplicates — entity extraction agents
                # are responsible for assigning unique IDs

    # ── 2. Add new relationships ─────────────────────────────────────────────
    if new_relationships:
        existing_rel_ids = {r.id for r in relationships}
        for rel in new_relationships.relationships:
            if rel.id not in existing_rel_ids:
                relationships.append(rel)
                existing_rel_ids.add(rel.id)

    # ── 3 & 4. Collect all IDs to remove ────────────────────────────────────
    remove_entity_ids: set[str] = set(entities_to_remove or [])
    remove_rel_ids:    set[str] = set(relationships_to_remove or [])
    if new_entities:
        remove_entity_ids |= set(new_entities.entities_to_remove)
    if new_relationships:
        remove_rel_ids |= set(new_relationships.relationships_to_remove)

    if remove_entity_ids:
        for etype in entities:
            entities[etype] = [e for e in entities[etype] if e.id not in remove_entity_ids]
        # cascade: drop any relationship that touched a removed entity
        relationships = [
            r for r in relationships
            if r.from_id not in remove_entity_ids and r.to_id not in remove_entity_ids
        ]

    if remove_rel_ids:
        relationships = [r for r in relationships if r.id not in remove_rel_ids]

    kg = KnowledgeGraph(entities=entities, relationships=relationships)

    total_e = sum(len(v) for v in kg.entities.values())
    print(f"  [KGConsolidationAgent] entities={total_e}  relationships={len(kg.relationships)}"
          + (f"  +removed_entities={len(remove_entity_ids)}" if remove_entity_ids else "")
          + (f"  +removed_rels={len(remove_rel_ids)}" if remove_rel_ids else ""))
    return kg

# ─────────────────────────────────────────────────────────────────────────────
# 3. Relationship extraction agent  (initial + improvement runs)
# ─────────────────────────────────────────────────────────────────────────────

def relationship_extraction_agent(
    document_pages: list[str],
    relationship_ontology: dict,
    kg: KnowledgeGraph,
    judge_feedback: dict | None = None,
) -> RelationshipExtractionResult:
    """
    Extracts relationships page-by-page.
    In improvement runs, also recommends additions and removals from judge feedback.
    """
    is_improvement = judge_feedback is not None
    all_relationships: list = []
    all_to_remove:     list[str] = []
    rel_counter = len(kg.relationships) + 1

    for page_idx, page_text in enumerate(document_pages):
        label = f"RelationshipAgent[page {page_idx + 1}, {'improvement' if is_improvement else 'initial'}]"

        if is_improvement:
            system_prompt = (
                "You are a knowledge-graph relationship correction engine. "
                "Return ONLY a valid JSON object — no markdown, no prose."
            )
            user_prompt = f"""Correct relationships in the knowledge graph.

RELATIONSHIP ONTOLOGY (only use these types):
{_j(relationship_ontology)}

EXISTING KNOWLEDGE GRAPH (use entity IDs from here):
{_kg_json(kg)}

JUDGE FEEDBACK — missing relationships to ADD:
{_j(judge_feedback.get("missing_relationships", []))}

JUDGE FEEDBACK — hallucinated relationship IDs to REMOVE:
{_j(judge_feedback.get("hallucinated_relationships", []))}

DOCUMENT PAGE {page_idx + 1}:
{page_text}

Return:
{{
  "relationships": [
    {{"id": "rel_{rel_counter}", "type": "<ONTOLOGY_TYPE>", "from_id": "<id>", "to_id": "<id>", "evidence": "..."}}
  ],
  "relationships_to_remove": ["rel_id_1"]
}}

Only include NEW relationships. Do NOT repeat existing correct ones.
"""
        else:
            system_prompt = (
                "You are a knowledge-graph relationship extraction engine. "
                "Return ONLY a valid JSON object — no markdown, no prose."
            )
            user_prompt = f"""Extract relationships from the document page below.

RELATIONSHIP ONTOLOGY (only use these relationship types):
{_j(relationship_ontology)}

EXTRACTED ENTITIES (use their exact IDs):
{_kg_json(kg)}

DOCUMENT PAGE {page_idx + 1}:
{page_text}

Return:
{{
  "relationships": [
    {{"id": "rel_{rel_counter}", "type": "<ONTOLOGY_TYPE>", "from_id": "<id>", "to_id": "<id>", "evidence": "..."}}
  ],
  "relationships_to_remove": []
}}
"""

        raw         = call_llm(system_prompt, user_prompt, label=label)
        page_result = parse_and_validate(raw, RelationshipExtractionResult, label=label)

        for rel in page_result.relationships:
            rel.id = f"rel_{rel_counter}"
            rel_counter += 1

        all_relationships.extend(page_result.relationships)
        all_to_remove.extend(page_result.relationships_to_remove)
        print(f"  [RelationshipAgent] page {page_idx + 1}: "
              f"add={len(page_result.relationships)}  remove={len(page_result.relationships_to_remove)}")

    return RelationshipExtractionResult(
        relationships=all_relationships,
        relationships_to_remove=all_to_remove,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stray node detection agent
# ─────────────────────────────────────────────────────────────────────────────

def stray_node_agent(
    kg: KnowledgeGraph,
    relationship_ontology: dict,
    document_text: str,
) -> StrayNodeResult:
    all_ids   = kg.entity_ids()
    connected = {r.from_id for r in kg.relationships} | {r.to_id for r in kg.relationships}
    stray_ids = all_ids - connected

    if not stray_ids:
        print("  [StrayNodeAgent] No stray nodes — KG is clean.")
        return StrayNodeResult(status="clean")

    print(f"  [StrayNodeAgent] Found {len(stray_ids)} stray node(s): {stray_ids}")

    stray_entities = [
        e.to_flat_dict()
        for _, e in kg.all_entities_flat()
        if e.id in stray_ids
    ]

    system_prompt = (
        "You are a knowledge-graph quality agent. "
        "Find relationships for stray nodes (entities with no edges). "
        "Return ONLY a valid JSON object — no markdown, no prose."
    )
    user_prompt = f"""These entities currently have NO relationships in the KG.

STRAY ENTITIES:
{_j(stray_entities)}

FULL KNOWLEDGE GRAPH:
{_kg_json(kg)}

RELATIONSHIP ONTOLOGY:
{_j(relationship_ontology)}

DOCUMENT:
{document_text}

For each stray entity:
1. Try to find at least one relationship using only ontology types.
2. If no ontology type fits, add to ontology_gaps.

Return:
{{
  "new_relationships": [
    {{"id": "rel_stray_1", "type": "<type>", "from_id": "<id>", "to_id": "<id>",
      "evidence": "...", "reasoning": "..."}}
  ],
  "ontology_gaps": [
    {{"entity_id": "<id>", "reasoning": "...", "evidence": "..."}}
  ]
}}
"""

    raw    = call_llm(system_prompt, user_prompt, label="StrayNodeAgent")
    result = parse_and_validate(raw, StrayNodeResult, label="StrayNodeAgent")

    if result.status == "ontology_gap":
        print(f"  [StrayNodeAgent] ⚠ Ontology gap for {len(result.ontology_gaps)} node(s).")
    elif result.status == "resolved":
        print(f"  [StrayNodeAgent] Resolved {len(result.new_relationships)} relationship(s).")
    else:
        print("  [StrayNodeAgent] All stray nodes resolved.")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. KG Completeness judge
# ─────────────────────────────────────────────────────────────────────────────

def kg_completeness_judge(
    kg: KnowledgeGraph,
    document_text: str,
) -> KGCompletenessResult:
    system_prompt = (
        "You are a knowledge-graph completeness judge. "
        "Compare the KG against the source document. "
        "Return ONLY a valid JSON object — no markdown, no prose."
    )
    user_prompt = f"""Review the knowledge graph against the source document.

KNOWLEDGE GRAPH:
{_kg_json(kg)}

SOURCE DOCUMENT:
{document_text}

Return:
{{
  "status": "complete" or "needs_improvement",
  "missing_entities": [
    {{"entity_type": "<exact ontology type key>", "name": "...", "reasoning": "...", "evidence": "..."}}
  ],
  "missing_relationships": [
    {{"type": "<ONTOLOGY_TYPE>", "from": "<name or id>", "to": "<name or id>",
      "reasoning": "...", "evidence": "..."}}
  ],
  "hallucinated_entities": [
    {{"entity_id": "...", "reasoning": "...", "evidence": "..."}}
  ],
  "hallucinated_relationships": [
    {{"rel_id": "...", "reasoning": "...", "evidence": "..."}}
  ],
  "reasoning": "Overall assessment."
}}

Set status "complete" ONLY when there are zero missing items AND zero hallucinations.
"""

    raw    = call_llm(system_prompt, user_prompt, label="KGCompletenessJudge")
    result = parse_and_validate(raw, KGCompletenessResult, label="KGCompletenessJudge")

    print(f"  [KGCompletenessJudge] status={result.status} | "
          f"missing_e={len(result.missing_entities)} | "
          f"missing_r={len(result.missing_relationships)} | "
          f"halluc_e={len(result.hallucinated_entities)} | "
          f"halluc_r={len(result.hallucinated_relationships)}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. Contradiction spotting agent
# ─────────────────────────────────────────────────────────────────────────────

def contradiction_spotting_agent(
    kg: KnowledgeGraph,
    document_text: str,
) -> ContradictionResult:
    system_prompt = (
        "You are a financial intelligence analyst specialising in client profile consistency. "
        "Identify contradictions or suspicious inconsistencies. "
        "Return ONLY a valid JSON object — no markdown, no prose."
    )
    user_prompt = f"""Analyse the knowledge graph and document for contradictions.

KNOWLEDGE GRAPH:
{_kg_json(kg)}

SOURCE DOCUMENT:
{document_text}

Look for:
- Conflicting attribute values for the same entity
- Relationships that contradict each other
- Transactions inconsistent with stated roles or relationships
- Any other logical inconsistency

Return:
{{
  "contradictions": [
    {{"description": "...", "entities_involved": ["id1", "id2"], "evidence": "..."}}
  ],
  "assessment": "Overall summary."
}}

Return an empty list if no contradictions found.
"""

    raw    = call_llm(system_prompt, user_prompt, label="ContradictionAgent")
    result = parse_and_validate(raw, ContradictionResult, label="ContradictionAgent")
    print(f"  [ContradictionAgent] {len(result.contradictions)} contradiction(s)")
    return result
