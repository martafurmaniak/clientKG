"""
agents.py — all sub-agent implementations.

Every agent:
  • accepts typed inputs (KnowledgeGraph, ontology dicts, etc.)
  • calls the LLM via llm_utils.call_llm()
  • validates the raw LLM output through a Pydantic schema via parse_and_validate()
  • returns a typed Pydantic model

Improvement-run entity/relationship agents additionally receive the existing KG
and the judge's feedback so they can add *and* remove nodes/edges in one pass.
"""

from __future__ import annotations

import json
from llm_utils import call_llm, parse_and_validate
from schemas import (
    KnowledgeGraph,
    EntityExtractionResult,
    RelationshipExtractionResult,
    KGConsolidationResult,
    StrayNodeResult,
    KGCompletenessResult,
    ContradictionResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kg_json(kg: KnowledgeGraph) -> str:
    return json.dumps(kg.to_serialisable(), indent=2)


def _fmt(obj: object) -> str:
    return json.dumps(obj if not isinstance(obj, KnowledgeGraph) else obj.to_serialisable(), indent=2)


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

    When existing_kg and judge_feedback are supplied (improvement run) the
    agent is instructed to:
      • add entities that the judge flagged as missing
      • recommend removal of entity IDs the judge flagged as hallucinated
      • leave everything else unchanged
    """
    ontology_subset = {k: v for k, v in entity_ontology.items() if k in entity_types}
    is_improvement = existing_kg is not None and judge_feedback is not None

    if is_improvement:
        system_prompt = (
            "You are a precise knowledge-graph entity correction engine. "
            "You will receive an existing knowledge graph, a completeness judge's feedback, "
            "and the source document. "
            "Your job is to return ONLY the changes needed: "
            "new entities to add AND entity IDs to remove. "
            "Return ONLY a valid JSON object — no explanations, no markdown fences."
        )

        # Summarise only the feedback relevant to this agent's entity types
        relevant_missing = [
            e for e in judge_feedback.get("missing_entities", [])
            if e.get("entity_type", "").lower() in entity_types
        ]
        relevant_hallucinated = judge_feedback.get("hallucinated_entities", [])

        user_prompt = f"""You are correcting the knowledge graph for entity types: {entity_types}.

ENTITY ONTOLOGY:
{json.dumps(ontology_subset, indent=2)}

EXISTING KNOWLEDGE GRAPH:
{_kg_json(existing_kg)}

COMPLETENESS JUDGE FEEDBACK:
Missing entities (add these):
{json.dumps(relevant_missing, indent=2)}

Hallucinated entities (recommend removing these IDs):
{json.dumps(relevant_hallucinated, indent=2)}

SOURCE DOCUMENT:
{document_text}

Return a JSON object with:
  - keys matching the entity types ({entity_types}) → lists of NEW entity objects to add
  - "entities_to_remove": list of entity ID strings to remove

Only include entities that need to change. Do NOT re-list existing correct entities.
Each new entity needs an "id" field (e.g. "person_3", "org_2") that does not clash
with existing IDs in the knowledge graph.

Example:
{{
  "people": [{{"id": "person_3", "name": "New Person", "age": null, "role": null}}],
  "entities_to_remove": ["person_99", "org_5"]
}}
"""
    else:
        system_prompt = (
            "You are a precise knowledge-graph entity extraction engine. "
            "Return ONLY a valid JSON object — no explanations, no markdown fences. "
            "Use exactly the entity type keys provided in the ontology."
        )

        user_prompt = f"""Extract all entities of the following types from the document.

ENTITY TYPES AND THEIR ATTRIBUTES (ontology):
{json.dumps(ontology_subset, indent=2)}

DOCUMENT:
{document_text}

Return a JSON object where each key is an entity type from the ontology,
and the value is a list of entity objects with the attributes defined for that type,
plus an "id" field (e.g. "person_1", "org_1", "asset_1", "txn_1").

Example:
{{
  "people": [{{"id": "person_1", "name": "...", "age": null, "role": null}}],
  "organisations": [{{"id": "org_1", "name": "...", "type": null}}]
}}
"""

    raw = call_llm(system_prompt, user_prompt, label=agent_name)
    result = parse_and_validate(raw, EntityExtractionResult, label=agent_name)

    counts = {
        "people": len(result.people),
        "organisations": len(result.organisations),
        "assets": len(result.assets),
        "transactions": len(result.transactions),
        "to_remove": len(result.entities_to_remove),
    }
    mode = "improvement" if is_improvement else "initial"
    print(f"  [{agent_name}] {mode} run → {counts}")
    return result


def people_and_orgs_agent(
    document_text: str,
    entity_ontology: dict,
    existing_kg: KnowledgeGraph | None = None,
    judge_feedback: dict | None = None,
) -> EntityExtractionResult:
    return _entity_extraction_agent(
        "PeopleOrgsAgent", ["people", "organisations"],
        document_text, entity_ontology, existing_kg, judge_feedback,
    )


def assets_agent(
    document_text: str,
    entity_ontology: dict,
    existing_kg: KnowledgeGraph | None = None,
    judge_feedback: dict | None = None,
) -> EntityExtractionResult:
    return _entity_extraction_agent(
        "AssetsAgent", ["assets"],
        document_text, entity_ontology, existing_kg, judge_feedback,
    )


def transactions_agent(
    document_text: str,
    entity_ontology: dict,
    existing_kg: KnowledgeGraph | None = None,
    judge_feedback: dict | None = None,
) -> EntityExtractionResult:
    return _entity_extraction_agent(
        "TransactionsAgent", ["transactions"],
        document_text, entity_ontology, existing_kg, judge_feedback,
    )


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
    Merges additions into existing_kg and removes flagged IDs.
    Uses an LLM for de-duplication and reconciliation.
    Returns a validated KnowledgeGraph.
    """
    new_data: dict = {"entities": {}, "relationships": []}

    if new_entities:
        if new_entities.people:
            new_data["entities"]["people"] = [e.model_dump(exclude_none=True) for e in new_entities.people]
        if new_entities.organisations:
            new_data["entities"]["organisations"] = [e.model_dump(exclude_none=True) for e in new_entities.organisations]
        if new_entities.assets:
            new_data["entities"]["assets"] = [e.model_dump(exclude_none=True) for e in new_entities.assets]
        if new_entities.transactions:
            new_data["entities"]["transactions"] = [e.model_dump(exclude_none=True) for e in new_entities.transactions]

    if new_relationships:
        new_data["relationships"] = [r.model_dump(exclude_none=True) for r in new_relationships.relationships]

    # Aggregate all IDs to remove
    remove_entity_ids: set[str] = set(entities_to_remove or [])
    remove_rel_ids: set[str] = set(relationships_to_remove or [])
    if new_entities:
        remove_entity_ids |= set(new_entities.entities_to_remove)
    if new_relationships:
        remove_rel_ids |= set(new_relationships.relationships_to_remove)

    system_prompt = (
        "You are a knowledge-graph consolidation engine. "
        "Merge the two JSON knowledge graphs below into one coherent graph. "
        "De-duplicate entities (same real-world thing → keep one entry, merge attributes). "
        "Preserve all unique relationships. "
        "Return ONLY a valid JSON object with keys 'entities' and 'relationships'. "
        "No markdown, no explanations."
    )

    user_prompt = f"""EXISTING KNOWLEDGE GRAPH:
{_kg_json(existing_kg)}

NEW DATA TO MERGE:
{json.dumps(new_data, indent=2)}

Return the fully merged knowledge graph.
'entities' must be a dict with keys: people, organisations, assets, transactions.
'relationships' must be a list of objects with fields: id, type, from_id, to_id, evidence (optional).
"""

    raw = call_llm(system_prompt, user_prompt, label="KGConsolidationAgent")
    result = parse_and_validate(raw, KGConsolidationResult, label="KGConsolidationAgent")
    kg = result.to_knowledge_graph()

    # Apply removals directly — don't trust the LLM for deletions
    if remove_entity_ids:
        kg.entities.people        = [e for e in kg.entities.people        if e.id not in remove_entity_ids]
        kg.entities.organisations = [e for e in kg.entities.organisations if e.id not in remove_entity_ids]
        kg.entities.assets        = [e for e in kg.entities.assets        if e.id not in remove_entity_ids]
        kg.entities.transactions  = [e for e in kg.entities.transactions  if e.id not in remove_entity_ids]
        # Also drop any relationships that referenced a removed entity
        kg.relationships = [
            r for r in kg.relationships
            if r.from_id not in remove_entity_ids and r.to_id not in remove_entity_ids
        ]

    if remove_rel_ids:
        kg.relationships = [r for r in kg.relationships if r.id not in remove_rel_ids]

    total_entities = sum([
        len(kg.entities.people), len(kg.entities.organisations),
        len(kg.entities.assets), len(kg.entities.transactions),
    ])
    print(f"  [KGConsolidationAgent] total entities={total_entities}  relationships={len(kg.relationships)}"
          + (f"  removed_entities={len(remove_entity_ids)}" if remove_entity_ids else "")
          + (f"  removed_rels={len(remove_rel_ids)}" if remove_rel_ids else ""))
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

    In an improvement run (judge_feedback supplied) the agent is additionally
    asked to:
      • add relationships the judge flagged as missing
      • recommend removal of relationship IDs the judge flagged as hallucinated
    """
    is_improvement = judge_feedback is not None
    all_relationships = []
    all_to_remove: list[str] = []
    rel_counter = len(kg.relationships) + 1  # avoid ID collisions with existing rels

    for page_idx, page_text in enumerate(document_pages):
        if is_improvement:
            system_prompt = (
                "You are a knowledge-graph relationship correction engine. "
                "Return ONLY a valid JSON object — no markdown, no prose."
            )

            relevant_missing = judge_feedback.get("missing_relationships", [])
            relevant_hallucinated = judge_feedback.get("hallucinated_relationships", [])

            user_prompt = f"""You are correcting relationships in the knowledge graph.

RELATIONSHIP ONTOLOGY (only use these types):
{json.dumps(relationship_ontology, indent=2)}

EXISTING KNOWLEDGE GRAPH (use entity IDs from here):
{_kg_json(kg)}

COMPLETENESS JUDGE FEEDBACK:
Missing relationships (add these):
{json.dumps(relevant_missing, indent=2)}

Hallucinated relationships (recommend removing these rel IDs):
{json.dumps(relevant_hallucinated, indent=2)}

DOCUMENT PAGE {page_idx + 1}:
{page_text}

Return a JSON object:
{{
  "relationships": [
    {{
      "id": "rel_{rel_counter}",
      "type": "<ontology type>",
      "from_id": "<entity id>",
      "to_id": "<entity id>",
      "evidence": "<short text evidence>"
    }}
  ],
  "relationships_to_remove": ["rel_id_1", "rel_id_2"]
}}

Only include NEW relationships to add. Do NOT re-list existing correct ones.
"""
        else:
            system_prompt = (
                "You are a knowledge-graph relationship extraction engine. "
                "Return ONLY a valid JSON object — no markdown, no prose."
            )

            user_prompt = f"""Extract relationships from the document page below.

RELATIONSHIP ONTOLOGY (only use these relationship types):
{json.dumps(relationship_ontology, indent=2)}

ALREADY EXTRACTED ENTITIES (use their exact IDs):
{_kg_json(kg)}

DOCUMENT PAGE {page_idx + 1}:
{page_text}

Return a JSON object:
{{
  "relationships": [
    {{
      "id": "rel_{rel_counter}",
      "type": "<ontology type>",
      "from_id": "<entity id>",
      "to_id": "<entity id>",
      "evidence": "<short text evidence>"
    }}
  ],
  "relationships_to_remove": []
}}
"""

        mode = "improvement" if is_improvement else "initial"
        raw = call_llm(system_prompt, user_prompt,
                       label=f"RelationshipAgent[page {page_idx + 1}, {mode}]")
        page_result = parse_and_validate(raw, RelationshipExtractionResult,
                                         label=f"RelationshipAgent[page {page_idx + 1}]")

        # Re-index IDs to avoid collisions
        for rel in page_result.relationships:
            rel.id = f"rel_{rel_counter}"
            rel_counter += 1

        all_relationships.extend(page_result.relationships)
        all_to_remove.extend(page_result.relationships_to_remove)
        print(f"  [RelationshipAgent] page {page_idx + 1} ({mode}): "
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
    """
    Detects entities with no relationships, attempts to find edges for them,
    flags nodes whose relationships cannot be represented by the current ontology.
    """
    all_entity_ids = kg.entity_ids()
    connected_ids: set[str] = set()
    for rel in kg.relationships:
        connected_ids.add(rel.from_id)
        connected_ids.add(rel.to_id)

    stray_ids = all_entity_ids - connected_ids

    if not stray_ids:
        print("  [StrayNodeAgent] No stray nodes — KG is clean.")
        return StrayNodeResult(status="clean")

    print(f"  [StrayNodeAgent] Found {len(stray_ids)} stray node(s): {stray_ids}")

    stray_entities = []
    for lst in [kg.entities.people, kg.entities.organisations,
                kg.entities.assets, kg.entities.transactions]:
        for e in lst:
            if e.id in stray_ids:
                stray_entities.append(e.model_dump(exclude_none=True))

    system_prompt = (
        "You are a knowledge-graph quality agent. "
        "Find relationships for stray nodes (entities with no edges). "
        "Return ONLY a valid JSON object — no markdown, no prose."
    )

    user_prompt = f"""The following entities currently have NO relationships in the knowledge graph.

STRAY ENTITIES:
{json.dumps(stray_entities, indent=2)}

FULL KNOWLEDGE GRAPH:
{_kg_json(kg)}

RELATIONSHIP ONTOLOGY:
{json.dumps(relationship_ontology, indent=2)}

DOCUMENT:
{document_text}

For each stray entity:
1. Try to find at least one relationship to another entity in the KG using only ontology types.
2. If no valid relationship type exists in the ontology, add to ontology_gaps.

Return:
{{
  "new_relationships": [
    {{
      "id": "rel_stray_1",
      "type": "<ontology type>",
      "from_id": "<entity id>",
      "to_id": "<entity id>",
      "evidence": "<short text evidence>",
      "reasoning": "<why this relationship holds>"
    }}
  ],
  "ontology_gaps": [
    {{
      "entity_id": "<id>",
      "reasoning": "<why no ontology type fits>",
      "evidence": "<text evidence>"
    }}
  ]
}}
"""

    raw = call_llm(system_prompt, user_prompt, label="StrayNodeAgent")
    result = parse_and_validate(raw, StrayNodeResult, label="StrayNodeAgent")
    # status is derived automatically by the model_validator on StrayNodeResult

    if result.status == "ontology_gap":
        print(f"  [StrayNodeAgent] ⚠ Ontology gap for {len(result.ontology_gaps)} node(s). Halting.")
    elif result.status == "resolved":
        print(f"  [StrayNodeAgent] Resolved {len(result.new_relationships)} relationship(s).")
    else:
        print("  [StrayNodeAgent] All stray nodes resolved — KG is clean.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. KG Completeness judge
# ─────────────────────────────────────────────────────────────────────────────

def kg_completeness_judge(
    kg: KnowledgeGraph,
    document_text: str,
) -> KGCompletenessResult:
    """Checks for missing nodes/edges and hallucinated nodes/edges."""

    system_prompt = (
        "You are a knowledge-graph completeness judge. "
        "Compare the knowledge graph against the source document. "
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
    {{"entity_type": "people|organisations|assets|transactions", "name": "...", "reasoning": "...", "evidence": "..."}}
  ],
  "missing_relationships": [
    {{"type": "<ONTOLOGY_TYPE>", "from": "<name or id>", "to": "<name or id>", "reasoning": "...", "evidence": "..."}}
  ],
  "hallucinated_entities": [
    {{"entity_id": "...", "reasoning": "...", "evidence": "..."}}
  ],
  "hallucinated_relationships": [
    {{"rel_id": "...", "reasoning": "...", "evidence": "..."}}
  ],
  "reasoning": "Overall assessment."
}}

Set status "complete" only when there are zero missing items AND zero hallucinations.
"""

    raw = call_llm(system_prompt, user_prompt, label="KGCompletenessJudge")
    result = parse_and_validate(raw, KGCompletenessResult, label="KGCompletenessJudge")
    # status is re-derived by the model_validator

    print(f"  [KGCompletenessJudge] status={result.status} | "
          f"missing_entities={len(result.missing_entities)} | "
          f"missing_relationships={len(result.missing_relationships)} | "
          f"hallucinated_entities={len(result.hallucinated_entities)} | "
          f"hallucinated_relationships={len(result.hallucinated_relationships)}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. Contradiction spotting agent
# ─────────────────────────────────────────────────────────────────────────────

def contradiction_spotting_agent(
    kg: KnowledgeGraph,
    document_text: str,
) -> ContradictionResult:
    """Identifies contradictions or inconsistencies in the client profile."""

    system_prompt = (
        "You are a financial intelligence analyst specialising in client profile consistency. "
        "Identify contradictions or suspicious inconsistencies. "
        "Return ONLY a valid JSON object — no markdown, no prose."
    )

    user_prompt = f"""Analyse the final knowledge graph and source document for contradictions.

KNOWLEDGE GRAPH:
{_kg_json(kg)}

SOURCE DOCUMENT:
{document_text}

Look for:
- Conflicting attribute values for the same entity
- Relationships that contradict each other
- Transactions inconsistent with stated roles or relationships
- Any other logical inconsistency in the client profile

Return:
{{
  "contradictions": [
    {{
      "description": "What the contradiction is",
      "entities_involved": ["entity_id_1", "entity_id_2"],
      "evidence": "Quotes or paraphrases from document or KG"
    }}
  ],
  "assessment": "Overall summary of findings."
}}

Return an empty list for contradictions if none are found.
"""

    raw = call_llm(system_prompt, user_prompt, label="ContradictionAgent")
    result = parse_and_validate(raw, ContradictionResult, label="ContradictionAgent")
    print(f"  [ContradictionAgent] found {len(result.contradictions)} contradiction(s)")
    return result
