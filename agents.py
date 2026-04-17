"""
agents.py — all sub-agent implementations.

Each agent is a plain function that takes structured inputs and returns
structured outputs (Python dicts / lists).  No side-effects; the
orchestrator owns all state.
"""

import json
from llm_utils import call_llm, extract_json


# ─────────────────────────────────────────────────────────────────────────────
# 1. Entity extraction agents
# ─────────────────────────────────────────────────────────────────────────────

def _entity_extraction_agent(
    agent_name: str,
    entity_types: list[str],
    document_text: str,
    entity_ontology: dict,
) -> dict:
    """Generic entity extractor; specialised agents call this."""

    ontology_subset = {k: v for k, v in entity_ontology.items() if k in entity_types}

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

Example structure:
{{
  "people": [{{"id": "person_1", "name": "...", "age": null, ...}}],
  "organisations": [{{"id": "org_1", "name": "...", ...}}]
}}
"""

    raw = call_llm(system_prompt, user_prompt, label=agent_name)
    result = extract_json(raw)
    print(f"  [{agent_name}] extracted: { {k: len(v) for k, v in result.items()} }")
    return result


def people_and_orgs_agent(document_text: str, entity_ontology: dict) -> dict:
    return _entity_extraction_agent(
        "PeopleOrgsAgent",
        ["people", "organisations"],
        document_text,
        entity_ontology,
    )


def assets_agent(document_text: str, entity_ontology: dict) -> dict:
    return _entity_extraction_agent(
        "AssetsAgent",
        ["assets"],
        document_text,
        entity_ontology,
    )


def transactions_agent(document_text: str, entity_ontology: dict) -> dict:
    return _entity_extraction_agent(
        "TransactionsAgent",
        ["transactions"],
        document_text,
        entity_ontology,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. KG Consolidation agent
# ─────────────────────────────────────────────────────────────────────────────

def kg_consolidation_agent(existing_kg: dict, new_data: dict) -> dict:
    """
    Merges new_data (entities and/or relationships) into existing_kg.
    Uses an LLM to de-duplicate and reconcile; returns the full merged KG.
    """
    system_prompt = (
        "You are a knowledge-graph consolidation engine. "
        "Merge the two JSON knowledge graphs below into one coherent graph. "
        "De-duplicate entities (same real-world thing → keep one entry, merge attributes). "
        "Preserve all unique relationships. "
        "Return ONLY a valid JSON object with keys 'entities' and 'relationships'. "
        "No markdown, no explanations."
    )

    user_prompt = f"""EXISTING KNOWLEDGE GRAPH:
{json.dumps(existing_kg, indent=2)}

NEW DATA TO MERGE:
{json.dumps(new_data, indent=2)}

Return the fully merged knowledge graph.
The 'entities' value must be a dict keyed by entity type (people, organisations, assets, transactions).
The 'relationships' value must be a list of relationship objects
with fields: id, type, from_id, to_id, and optionally evidence.
"""

    raw = call_llm(system_prompt, user_prompt, label="KGConsolidationAgent")
    merged = extract_json(raw)

    # Ensure the top-level keys always exist
    merged.setdefault("entities", {})
    merged.setdefault("relationships", [])

    entity_counts = {k: len(v) for k, v in merged["entities"].items()}
    print(f"  [KGConsolidationAgent] entities={entity_counts}  relationships={len(merged['relationships'])}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 3. Relationship extraction agent  (processes one page at a time)
# ─────────────────────────────────────────────────────────────────────────────

def relationship_extraction_agent(
    document_pages: list[str],
    relationship_ontology: dict,
    entities: dict,
) -> list[dict]:
    """
    Extracts relationships page-by-page.
    Returns a flat list of relationship dicts.
    """
    all_relationships: list[dict] = []
    rel_counter = 1

    for page_idx, page_text in enumerate(document_pages):
        system_prompt = (
            "You are a knowledge-graph relationship extraction engine. "
            "Return ONLY a valid JSON array of relationship objects — no markdown, no prose."
        )

        user_prompt = f"""Extract relationships from the document page below.

RELATIONSHIP ONTOLOGY (only use these relationship types):
{json.dumps(relationship_ontology, indent=2)}

ALREADY EXTRACTED ENTITIES (use their exact IDs):
{json.dumps(entities, indent=2)}

DOCUMENT PAGE {page_idx + 1}:
{page_text}

Return a JSON array. Each element must have:
  "id"       : unique string like "rel_1", "rel_2", ...
  "type"     : one of the relationship types from the ontology
  "from_id"  : entity id of the source
  "to_id"    : entity id of the target
  "evidence" : short quote or paraphrase from the text supporting this relationship

Example:
[
  {{"id": "rel_1", "type": "OWNS", "from_id": "person_1", "to_id": "asset_1", "evidence": "John holds a savings account..."}}
]
"""

        raw = call_llm(system_prompt, user_prompt, label=f"RelationshipAgent[page {page_idx + 1}]")
        rels = extract_json(raw)
        if isinstance(rels, dict):
            rels = rels.get("relationships", [])

        # Re-index IDs to avoid collisions across pages
        for rel in rels:
            rel["id"] = f"rel_{rel_counter}"
            rel_counter += 1
        all_relationships.extend(rels)
        print(f"  [RelationshipAgent] page {page_idx + 1}: extracted {len(rels)} relationships")

    return all_relationships


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stray node detection agent
# ─────────────────────────────────────────────────────────────────────────────

def stray_node_agent(
    kg: dict,
    relationship_ontology: dict,
    document_text: str,
) -> dict:
    """
    Detects entities with no relationships, attempts to find edges for them,
    and flags nodes whose relationships cannot be represented by the current ontology.

    Returns:
        {
          "status": "clean" | "resolved" | "ontology_gap",
          "new_relationships": [...],          # relationships found for previously stray nodes
          "ontology_gaps": [                   # nodes that cannot be represented
              {"entity_id": ..., "reasoning": ..., "evidence": ...}
          ]
        }
    """
    entities = kg.get("entities", {})
    relationships = kg.get("relationships", [])

    # Collect all entity IDs
    all_entity_ids: set[str] = set()
    for entity_list in entities.values():
        for entity in entity_list:
            all_entity_ids.add(entity["id"])

    # Collect entity IDs that appear in at least one relationship
    connected_ids: set[str] = set()
    for rel in relationships:
        connected_ids.add(rel.get("from_id", ""))
        connected_ids.add(rel.get("to_id", ""))

    stray_ids = all_entity_ids - connected_ids

    if not stray_ids:
        print("  [StrayNodeAgent] No stray nodes found — KG is clean.")
        return {"status": "clean", "new_relationships": [], "ontology_gaps": []}

    print(f"  [StrayNodeAgent] Found {len(stray_ids)} stray node(s): {stray_ids}")

    # Build a lookup of stray entities for the prompt
    stray_entities = []
    for entity_list in entities.values():
        for entity in entity_list:
            if entity["id"] in stray_ids:
                stray_entities.append(entity)

    system_prompt = (
        "You are a knowledge-graph quality agent. "
        "Your job is to find relationships for stray nodes (entities with no edges). "
        "Return ONLY a valid JSON object — no markdown, no prose."
    )

    user_prompt = f"""The following entities currently have NO relationships in the knowledge graph.

STRAY ENTITIES:
{json.dumps(stray_entities, indent=2)}

FULL KNOWLEDGE GRAPH:
{json.dumps(kg, indent=2)}

RELATIONSHIP ONTOLOGY:
{json.dumps(relationship_ontology, indent=2)}

DOCUMENT:
{document_text}

For each stray entity:
1. Try to find at least one relationship to another entity in the KG using only ontology relationship types.
2. If no valid relationship type exists, flag it as an ontology gap.

Return a JSON object with exactly these keys:
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
    result = extract_json(raw)
    result.setdefault("new_relationships", [])
    result.setdefault("ontology_gaps", [])

    has_gaps = len(result["ontology_gaps"]) > 0
    has_new_rels = len(result["new_relationships"]) > 0

    if has_gaps:
        result["status"] = "ontology_gap"
        print(f"  [StrayNodeAgent] ⚠ Ontology gap detected for {len(result['ontology_gaps'])} node(s). Halting.")
    elif has_new_rels:
        result["status"] = "resolved"
        print(f"  [StrayNodeAgent] Resolved {len(result['new_relationships'])} new relationship(s).")
    else:
        result["status"] = "clean"
        print("  [StrayNodeAgent] All stray nodes resolved — KG is clean.")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. KG Completeness judge
# ─────────────────────────────────────────────────────────────────────────────

def kg_completeness_judge(kg: dict, document_text: str) -> dict:
    """
    Checks for missing nodes/edges and hallucinated nodes/edges.

    Returns:
        {
          "status": "complete" | "needs_improvement",
          "missing_entities": [...],
          "missing_relationships": [...],
          "hallucinated_entities": [...],
          "hallucinated_relationships": [...],
          "reasoning": "..."
        }
    """
    system_prompt = (
        "You are a knowledge-graph completeness judge. "
        "Compare the knowledge graph against the source document and identify gaps and hallucinations. "
        "Return ONLY a valid JSON object — no markdown, no prose."
    )

    user_prompt = f"""Review the knowledge graph below against the source document.

KNOWLEDGE GRAPH:
{json.dumps(kg, indent=2)}

SOURCE DOCUMENT:
{document_text}

Return a JSON object with exactly these keys:
{{
  "status": "complete" or "needs_improvement",
  "missing_entities": [
    {{"entity_type": "...", "name": "...", "reasoning": "...", "evidence": "..."}}
  ],
  "missing_relationships": [
    {{"type": "...", "from": "...", "to": "...", "reasoning": "...", "evidence": "..."}}
  ],
  "hallucinated_entities": [
    {{"entity_id": "...", "reasoning": "...", "evidence": "..."}}
  ],
  "hallucinated_relationships": [
    {{"rel_id": "...", "reasoning": "...", "evidence": "..."}}
  ],
  "reasoning": "Overall assessment."
}}

Set status to "complete" only if there are no missing items and no hallucinations.
"""

    raw = call_llm(system_prompt, user_prompt, label="KGCompletenessJudge")
    result = extract_json(raw)
    result.setdefault("status", "needs_improvement")
    result.setdefault("missing_entities", [])
    result.setdefault("missing_relationships", [])
    result.setdefault("hallucinated_entities", [])
    result.setdefault("hallucinated_relationships", [])

    print(f"  [KGCompletenessJudge] status={result['status']} | "
          f"missing_entities={len(result['missing_entities'])} | "
          f"missing_relationships={len(result['missing_relationships'])} | "
          f"hallucinated_entities={len(result['hallucinated_entities'])} | "
          f"hallucinated_relationships={len(result['hallucinated_relationships'])}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. Contradiction spotting agent
# ─────────────────────────────────────────────────────────────────────────────

def contradiction_spotting_agent(kg: dict, document_text: str) -> dict:
    """
    Identifies contradictions or inconsistencies in the client profile
    based on the completed knowledge graph.

    Returns:
        {
          "contradictions": [
              {"description": "...", "entities_involved": [...], "evidence": "..."}
          ],
          "assessment": "..."
        }
    """
    system_prompt = (
        "You are a financial intelligence analyst specialising in client profile consistency. "
        "Identify contradictions or suspicious inconsistencies in the knowledge graph. "
        "Return ONLY a valid JSON object — no markdown, no prose."
    )

    user_prompt = f"""Analyse the final knowledge graph and source document for contradictions or inconsistencies.

KNOWLEDGE GRAPH:
{json.dumps(kg, indent=2)}

SOURCE DOCUMENT:
{document_text}

Look for things like:
- Conflicting attribute values for the same entity
- Relationships that contradict each other
- Transactions inconsistent with stated roles or relationships
- Any other logical inconsistency in the client profile

Return a JSON object:
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

If there are no contradictions return an empty list for contradictions.
"""

    raw = call_llm(system_prompt, user_prompt, label="ContradictionAgent")
    result = extract_json(raw)
    result.setdefault("contradictions", [])
    result.setdefault("assessment", "")

    print(f"  [ContradictionAgent] found {len(result['contradictions'])} contradiction(s)")
    return result
