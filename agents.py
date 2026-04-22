"""
agents.py — all sub-agent implementations.

All agents use the unified flat KG format throughout:
  entities as a flat list with "type", "label", "attributes" on each entity
  relationships with "source"/"target"/"type"/"attributes"

User prompts are loaded from prompts/ as Jinja2 templates via render().
System prompts and custom instructions are loaded from agent_config.yaml
via get_system_prompt() and get_instructions().
"""

from __future__ import annotations

import json
from llm_utils import call_llm, parse_and_validate
from prompt_loader import render, render_with_ontology, get_system_prompt, get_instructions
from ontology_utils import assign_ids, GlobalIDRegistry, get_id_prefix
from schemas import (
    KnowledgeGraph,
    EntityExtractionResult,
    RelationshipExtractionResult,
    StrayNodeResult,
    KGCuratorResult,
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
    agent_key: str,
    entity_types: list[str],
    document_text: str,
    entity_ontology: dict,
    existing_kg: KnowledgeGraph | None = None,
    judge_feedback: dict | None = None,
    custom_instructions: str | None = None,
    page_text: str | None = None,
    id_seed_kg: KnowledgeGraph | None = None,
    id_registry: "GlobalIDRegistry | None" = None,
) -> EntityExtractionResult:
    """
    Generic entity extractor.

    id_registry : GlobalIDRegistry — when provided IDs are assigned via the
                  global registry (unique across the entire pipeline run).
                  Falls back to assign_ids seeded from id_seed_kg or existing_kg.
    """
    ontology_subset = {k: v for k, v in entity_ontology.items() if k in entity_types}
    is_improvement  = existing_kg is not None and judge_feedback is not None

    existing_for_types = [
        e.to_dict() for e in existing_kg.entities
        if e.type in entity_types
    ] if existing_kg else []

    if is_improvement:
        system_prompt = get_system_prompt(agent_key)
        user_prompt = render_with_ontology(
            "entity_extraction_improvement.j2",
            entity_ontology=ontology_subset,
            relationship_ontology={},
            entity_types=entity_types,
            ontology_subset=ontology_subset,
            existing_for_types=existing_for_types,
            relevant_missing=[
                e for e in judge_feedback.get("missing_entities", [])
                if e.get("entity_type", "") in entity_types
            ],
            relevant_hallucinated=judge_feedback.get("hallucinated_entities", []),
            page_text=page_text if page_text is not None else document_text,
            custom_instructions=custom_instructions,
        )
    else:
        system_prompt = get_system_prompt(agent_key)
        user_prompt = render_with_ontology(
            "entity_extraction_initial.j2",
            entity_ontology=ontology_subset,
            relationship_ontology={},
            ontology_subset=ontology_subset,
            document_text=document_text,
            custom_instructions=custom_instructions,
        )

    raw    = call_llm(system_prompt, user_prompt, label=agent_name)
    result = parse_and_validate(raw, EntityExtractionResult, label=agent_name)

    # Assign deterministic IDs — LLM-provided IDs are discarded.
    # Use global registry when available (unique across entire pipeline run);
    # fall back to seed_kg-based assign_ids otherwise.
    if result.entities:
        if id_registry is not None:
            new_entities, id_map = id_registry.assign(result.entities)
        else:
            seed = id_seed_kg or existing_kg
            new_entities, id_map = assign_ids(result.entities, seed, entity_ontology)
        new_to_remove = [id_map.get(eid, eid) for eid in result.entities_to_remove]
        result = EntityExtractionResult(
            entities=new_entities,
            entities_to_remove=new_to_remove,
        )

    mode = "improvement" if is_improvement else "initial"
    print(f"  [{agent_name}] {mode} → entities={len(result.entities)}  to_remove={len(result.entities_to_remove)}")
    return result


# ── Type-classification keyword sets ─────────────────────────────────────────
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
    custom_instructions: str | None = None,
    id_seed_kg: KnowledgeGraph | None = None,
    id_registry: "GlobalIDRegistry | None" = None,
) -> EntityExtractionResult:
    types = _pick_types(entity_ontology, _PEOPLE_ORG_KW)
    if not types:
        types = [k for k in entity_ontology
                 if not _matches_any(k, _ASSET_KW) and not _matches_any(k, _TRANSACTION_KW)]
    return _entity_extraction_agent(
        "PeopleOrgsAgent", "people_orgs", types, document_text, entity_ontology,
        existing_kg, judge_feedback,
        custom_instructions=custom_instructions or get_instructions("people_orgs"),
        id_seed_kg=id_seed_kg,
        id_registry=id_registry,
    )


def assets_agent(
    document_text: str,
    entity_ontology: dict,
    existing_kg: KnowledgeGraph | None = None,
    judge_feedback: dict | None = None,
    custom_instructions: str | None = None,
    id_seed_kg: KnowledgeGraph | None = None,
    id_registry: "GlobalIDRegistry | None" = None,
) -> EntityExtractionResult:
    types = _pick_types(entity_ontology, _ASSET_KW)
    return _entity_extraction_agent(
        "AssetsAgent", "assets", types, document_text, entity_ontology,
        existing_kg, judge_feedback,
        custom_instructions=custom_instructions or get_instructions("assets"),
        id_seed_kg=id_seed_kg,
        id_registry=id_registry,
    )


def transactions_agent(
    document_text: str,
    entity_ontology: dict,
    existing_kg: KnowledgeGraph | None = None,
    judge_feedback: dict | None = None,
    custom_instructions: str | None = None,
    id_seed_kg: KnowledgeGraph | None = None,
    id_registry: "GlobalIDRegistry | None" = None,
) -> EntityExtractionResult:
    types = _pick_types(entity_ontology, _TRANSACTION_KW)
    return _entity_extraction_agent(
        "TransactionsAgent", "transactions", types, document_text, entity_ontology,
        existing_kg, judge_feedback,
        custom_instructions=custom_instructions or get_instructions("transactions"),
        id_seed_kg=id_seed_kg,
        id_registry=id_registry,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2a. Ontology compliance validator  (pure Python, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_ontology_compliance(
    entities: list,
    relationships: list,
    entity_ontology: dict,
    relationship_ontology: dict,
) -> tuple[list, list]:
    """
    Enforce ontology compliance on a fully merged entity + relationship list.

    Entity pass:
      - Unknown entity types (not in ontology) → logged and passed through unchanged
      - Attribute keys not defined in the ontology for the entity type → removed

    Relationship pass:
      - Unknown relationship types → removed entirely
      - source entity type not in ontology allowed "from" list → removed
      - target entity type not in ontology allowed "to" list → removed
      - Attribute keys not defined in ontology for the relationship type → removed
        (evidence is always preserved regardless)

    Uses id → type lookup built from the entity list, not ID prefix parsing.
    Only validates types that are present in their respective ontologies.
    """
    # ID existence check always runs (it is ontology-independent).
    # Attribute/type checks only run when the relevant ontology is provided.

    # ── Build id → type lookup ────────────────────────────────────────────────
    id_to_type: dict[str, str] = {e.id: e.type for e in entities}

    # ── Entity attribute compliance ───────────────────────────────────────────
    validated_entities = []
    for entity in entities:
        etype = entity.type
        if etype not in entity_ontology:
            validated_entities.append(entity)   # unknown type — pass through
            continue

        allowed_raw = entity_ontology[etype].get("attributes", {})
        allowed_keys = (
            set(allowed_raw.keys())
            if isinstance(allowed_raw, dict)
            else set(allowed_raw)
        )

        bad_keys = {k for k in entity.attributes if k not in allowed_keys}
        if bad_keys:
            print(f"  [OntologyCheck] Entity {entity.id} ({etype}): "
                  f"removed disallowed attributes {bad_keys}")
            clean_attrs = {k: v for k, v in entity.attributes.items()
                           if k not in bad_keys}
            entity = entity.model_copy(update={"attributes": clean_attrs})

        validated_entities.append(entity)

    # ── Relationship compliance ───────────────────────────────────────────────
    valid_entity_ids: set[str] = {e.id for e in validated_entities}
    validated_relationships = []
    for rel in relationships:
        rtype = rel.type

        # ── Check source and target IDs exist in the entity list ──────────────
        if rel.source not in valid_entity_ids:
            print(f"  [OntologyCheck] Relationship [{rtype}]: "
                  f"source id '{rel.source}' does not exist in KG, removed")
            continue
        if rel.target not in valid_entity_ids:
            print(f"  [OntologyCheck] Relationship [{rtype}]: "
                  f"target id '{rel.target}' does not exist in KG, removed")
            continue

        # ── Unknown relationship type ─────────────────────────────────────────
        if relationship_ontology and rtype not in relationship_ontology:
            print(f"  [OntologyCheck] Relationship {rel.source}→{rel.target} [{rtype}]: "
                  f"unknown type, removed")
            continue

        # ── Type direction checks (only when ontology is provided) ────────────
        if relationship_ontology:
            ont_rel      = relationship_ontology[rtype]
            allowed_from = ont_rel.get("from", [])
            allowed_to   = ont_rel.get("to",   [])

            if isinstance(allowed_from, str):
                allowed_from = [v.strip() for v in allowed_from.split("|") if v.strip()]
            if isinstance(allowed_to, str):
                allowed_to   = [v.strip() for v in allowed_to.split("|") if v.strip()]

            source_type = id_to_type.get(rel.source)
            target_type = id_to_type.get(rel.target)

            source_ok = not allowed_from or not source_type or source_type in allowed_from
            target_ok = not allowed_to   or not target_type or target_type in allowed_to

            if not source_ok or not target_ok:
                # Before dropping, check if inverting source↔target satisfies the ontology
                inverted_source_ok = not allowed_from or not target_type or target_type in allowed_from
                inverted_target_ok = not allowed_to   or not source_type or source_type in allowed_to

                if inverted_source_ok and inverted_target_ok:
                    # Invert the relationship direction — keep it
                    print(f"  [OntologyCheck] Relationship {rel.source}→{rel.target} [{rtype}]: "
                          f"inverted source↔target to satisfy ontology "
                          f"({source_type}→{target_type} became {target_type}→{source_type})")
                    rel = rel.model_copy(update={"source": rel.target, "target": rel.source})
                else:
                    if not source_ok:
                        print(f"  [OntologyCheck] Relationship {rel.source}→{rel.target} [{rtype}]: "
                              f"source type '{source_type}' not in allowed from {allowed_from}, "
                              f"inversion also invalid, removed")
                    else:
                        print(f"  [OntologyCheck] Relationship {rel.source}→{rel.target} [{rtype}]: "
                              f"target type '{target_type}' not in allowed to {allowed_to}, "
                              f"inversion also invalid, removed")
                    continue

            # ── Strip disallowed attributes (keep evidence always) ────────────
            allowed_raw  = ont_rel.get("attributes", {})
            allowed_keys = (
                set(allowed_raw.keys())
                if isinstance(allowed_raw, dict)
                else set(allowed_raw)
            ) | {"evidence"}

            bad_keys = {k for k in rel.attributes if k not in allowed_keys}
            if bad_keys:
                print(f"  [OntologyCheck] Relationship {rel.source}→{rel.target} [{rtype}]: "
                      f"removed disallowed attributes {bad_keys}")
                clean_attrs = {k: v for k, v in rel.attributes.items()
                               if k not in bad_keys}
                rel = rel.model_copy(update={"attributes": clean_attrs})

        validated_relationships.append(rel)

    dropped_e = len(entities)      - len(validated_entities)
    dropped_r = len(relationships) - len(validated_relationships)
    if dropped_e or dropped_r:
        print(f"  [OntologyCheck] Summary: dropped {dropped_e} entity(ies), "
              f"{dropped_r} relationship(s) for ontology violations")

    return validated_entities, validated_relationships


def validate_kg(
    kg: KnowledgeGraph,
    entity_ontology: dict,
    relationship_ontology: dict,
) -> KnowledgeGraph:
    """
    Public wrapper around _validate_ontology_compliance.
    Returns a new KnowledgeGraph with all ontology violations corrected:
      - relationships referencing non-existent entity IDs → dropped
      - relationships with unknown types or wrong source/target types → dropped
      - disallowed attributes on entities or relationships → stripped
    Call this after any operation that may introduce non-compliant data.
    """
    validated_entities, validated_relationships = _validate_ontology_compliance(
        list(kg.entities),
        list(kg.relationships),
        entity_ontology,
        relationship_ontology,
    )
    return KnowledgeGraph(
        entities=validated_entities,
        relationships=validated_relationships,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. KG Consolidation agent  (deterministic — no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def kg_consolidation_agent(
    existing_kg: KnowledgeGraph,
    new_entities: EntityExtractionResult | None = None,
    new_relationships: RelationshipExtractionResult | None = None,
    entities_to_remove: list[str] | None = None,
    relationships_to_remove: list[str] | None = None,
    relationship_keys_to_remove: list[tuple] | None = None,
    entities_to_update: list | None = None,
    relationships_to_update: list | None = None,
    entity_ontology: dict | None = None,
    relationship_ontology: dict | None = None,
) -> KnowledgeGraph:
    """
    Pure deterministic merge — no LLM call.

    Operations in order:
      1. Apply attribute patches to existing entities (update)
      2. Apply attribute patches to existing relationships (update)
      3. Append new entities (dedup by id)
      4. Append new relationships (dedup by source+target+type)
      5. Remove entities by ID + cascade to orphaned relationships
      6. Remove relationships by (source, target, type) key
      7. Ontology compliance validation (strip bad attributes, remove
         relationships with wrong source/target types or unknown types)
    """
    entities:      list = list(existing_kg.entities)
    relationships: list = list(existing_kg.relationships)

    # ── 1. Update existing entities ──────────────────────────────────────────
    if entities_to_update:
        patch_map = {u.entity_id: u.attributes_patch for u in entities_to_update}
        updated_entities = []
        for e in entities:
            if e.id in patch_map:
                # Overwrite attributes with the complete intended patch
                updated_entities.append(
                    e.model_copy(update={"attributes": patch_map[e.id]})
                )
                print(f"  [KGConsolidationAgent] updated entity {e.id}")
            else:
                updated_entities.append(e)
        entities = updated_entities

    # ── 2. Update existing relationships ─────────────────────────────────────
    if relationships_to_update:
        # Key: (source, target, type)
        rel_patch_map = {
            (u.source, u.target, u.type): u.attributes_patch
            for u in relationships_to_update
        }
        updated_rels = []
        for r in relationships:
            key = (r.source, r.target, r.type)
            if key in rel_patch_map:
                updated_rels.append(
                    r.model_copy(update={"attributes": rel_patch_map[key]})
                )
                print(f"  [KGConsolidationAgent] updated relationship {r.source}→{r.target} [{r.type}]")
            else:
                updated_rels.append(r)
        relationships = updated_rels

    # ── 3. Add new entities (dedup by id — skip if ID already exists) ────────
    if new_entities:
        existing_ids = {e.id for e in entities}
        for entity in new_entities.entities:
            if entity.id not in existing_ids:
                entities.append(entity)
                existing_ids.add(entity.id)
            # If the ID already exists, the entity is already in the KG
            # (e.g. a history entity correctly reused). Skip silently.

    # ── 4. Add new relationships (dedup by source+target+type) ───────────────
    if new_relationships:
        existing_rel_keys = {(r.source, r.target, r.type) for r in relationships}
        for rel in new_relationships.relationships:
            key = (rel.source, rel.target, rel.type)
            if key not in existing_rel_keys:
                relationships.append(rel)
                existing_rel_keys.add(key)

    # ── 5. Collect all IDs to remove ─────────────────────────────────────────
    remove_entity_ids: set[str] = set(entities_to_remove or [])
    if new_entities:
        remove_entity_ids |= set(new_entities.entities_to_remove)

    if remove_entity_ids:
        entities = [e for e in entities if e.id not in remove_entity_ids]
        relationships = [
            r for r in relationships
            if r.source not in remove_entity_ids and r.target not in remove_entity_ids
        ]

    # ── 6. Remove relationships by (source, target, type) key  [Fix 10] ──────
    if relationship_keys_to_remove:
        remove_rel_key_set = set(relationship_keys_to_remove)
        before = len(relationships)
        relationships = [
            r for r in relationships
            if (r.source, r.target, r.type) not in remove_rel_key_set
        ]
        removed_rels = before - len(relationships)
    else:
        removed_rels = 0

    # ── 7. Ontology compliance validation ─────────────────────────────────────
    if entity_ontology or relationship_ontology:
        entities, relationships = _validate_ontology_compliance(
            entities, relationships,
            entity_ontology       or {},
            relationship_ontology or {},
        )

    kg = KnowledgeGraph(entities=entities, relationships=relationships)
    print(f"  [KGConsolidationAgent] entities={len(kg.entities)}  relationships={len(kg.relationships)}"
          + (f"  removed_e={len(remove_entity_ids)}" if remove_entity_ids else "")
          + (f"  removed_r={removed_rels}" if removed_rels else ""))
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
    is_improvement = judge_feedback is not None
    all_relationships: list = []
    all_to_remove:     list[str] = []

    # Fix 6: build a slim entity index (id + type + label) to pass alongside
    # every page so the model can reference cross-page entities by ID even when
    # those entities were introduced on a different page.
    entity_index = [
        {"id": e.id, "type": e.type, "label": e.label}
        for e in kg.entities
    ]

    for page_idx, page_text in enumerate(document_pages):
        label = f"RelationshipAgent[page {page_idx + 1}, {'improvement' if is_improvement else 'initial'}]"

        if is_improvement:
            system_prompt = get_system_prompt("relationship_extraction")
            user_prompt = render_with_ontology(
                "relationship_extraction_improvement.j2",
                entity_ontology={},
                relationship_ontology=relationship_ontology,
                existing_kg=kg,
                kg=kg.to_serialisable(),
                entity_index=entity_index,
                missing_relationships=judge_feedback.get("missing_relationships", []),
                hallucinated_relationships=judge_feedback.get("hallucinated_relationships", []),
                page_number=page_idx + 1,
                page_text=page_text,
            )
        else:
            system_prompt = get_system_prompt("relationship_extraction")
            user_prompt = render_with_ontology(
                "relationship_extraction_initial.j2",
                entity_ontology={},
                relationship_ontology=relationship_ontology,
                existing_kg=kg,
                kg=kg.to_serialisable(),
                entity_index=entity_index,
                page_number=page_idx + 1,
                page_text=page_text,
            )

        raw         = call_llm(system_prompt, user_prompt, label=label)
        page_result = parse_and_validate(raw, RelationshipExtractionResult, label=label)

        all_relationships.extend(page_result.relationships)
        all_to_remove.extend(page_result.relationships_to_remove)
        print(f"  [RelationshipAgent] page {page_idx + 1}: "
              f"add={len(page_result.relationships)}  remove={len(page_result.relationships_to_remove)}")

    return RelationshipExtractionResult(
        relationships=all_relationships,
        relationships_to_remove=all_to_remove,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Subgraph connectivity utilities + agent
# ─────────────────────────────────────────────────────────────────────────────

def find_connected_components(kg: KnowledgeGraph) -> list[set[str]]:
    """
    Return a list of connected components (each a set of entity IDs).
    Uses union-find on the relationship graph — purely deterministic, no LLM.
    Isolated entities (no relationships) each form their own singleton component.
    """
    parent: dict[str, str] = {e.id: e.id for e in kg.entities}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for rel in kg.relationships:
        if rel.source in parent and rel.target in parent:
            _union(rel.source, rel.target)

    # Group entity IDs by root
    groups: dict[str, set[str]] = {}
    for eid in parent:
        root = _find(eid)
        groups.setdefault(root, set()).add(eid)

    return list(groups.values())


def subgraph_connector_agent(
    kg: KnowledgeGraph,
    main_entity_ids: set[str],
    isolated_entity_ids: set[str],
    relationship_ontology: dict,
    document_text: str,
) -> RelationshipExtractionResult:
    """
    Ask the LLM to find relationships connecting an isolated subgraph to the
    main connected component.

    Returns a RelationshipExtractionResult with the bridging relationships.
    """
    main_entities     = [e.to_dict() for e in kg.entities if e.id in main_entity_ids]
    isolated_entities = [e.to_dict() for e in kg.entities if e.id in isolated_entity_ids]

    system_prompt = get_system_prompt("subgraph_connector")
    user_prompt   = render_with_ontology(
        "subgraph_connector.j2",
        entity_ontology={},
        relationship_ontology=relationship_ontology,
        main_entities=main_entities,
        isolated_entities=isolated_entities,
        document_text=document_text,
    )

    raw    = call_llm(system_prompt, user_prompt, label="SubgraphConnector")
    result = parse_and_validate(raw, RelationshipExtractionResult, label="SubgraphConnector")

    # Keep only relationships that actually bridge the two subgraphs
    bridging = [
        r for r in result.relationships
        if (r.source in main_entity_ids and r.target in isolated_entity_ids)
        or (r.source in isolated_entity_ids and r.target in main_entity_ids)
    ]
    if len(bridging) < len(result.relationships):
        print(f"  [SubgraphConnector] Filtered to {len(bridging)} bridging "
              f"relationships (dropped {len(result.relationships) - len(bridging)} non-bridging)")

    return RelationshipExtractionResult(relationships=bridging)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stray node detection agent
# ─────────────────────────────────────────────────────────────────────────────

def stray_node_agent(
    kg: KnowledgeGraph,
    relationship_ontology: dict,
    document_text: str,
) -> StrayNodeResult:
    all_ids   = kg.entity_ids()
    connected = {r.source for r in kg.relationships} | {r.target for r in kg.relationships}
    stray_ids = all_ids - connected

    if not stray_ids:
        print("  [StrayNodeAgent] No stray nodes — KG is clean.")
        return StrayNodeResult(status="clean")

    print(f"  [StrayNodeAgent] Found {len(stray_ids)} stray node(s): {stray_ids}")

    stray_entities = [e.to_dict() for e in kg.entities if e.id in stray_ids]

    system_prompt = get_system_prompt("stray_node")
    user_prompt = render(
        "stray_node.j2",
        stray_entities=stray_entities,
        kg=kg.to_serialisable(),
        relationship_ontology=relationship_ontology,
        document_text=document_text,
    )

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
# 5. KG Curator agent
# ─────────────────────────────────────────────────────────────────────────────

def kg_curator_agent(
    kg: KnowledgeGraph,
    document_pages: list[str],
    entity_ontology: dict | None = None,
    relationship_ontology: dict | None = None,
) -> KGCuratorResult:
    """
    Curates the KG against the source document.
    Returns three action lists: what to add, remove, and update.
    Accepts optional ontology dicts so the curator can enforce schema compliance.
    """
    numbered_pages = "\n\n".join(
        f"--- PAGE {i} ---\n{page}" for i, page in enumerate(document_pages)
    )

    system_prompt = get_system_prompt("kg_curator")
    user_prompt = render_with_ontology(
        "kg_curator.j2",
        entity_ontology=entity_ontology or {},
        relationship_ontology=relationship_ontology or {},
        existing_kg=kg,
        kg=kg.to_serialisable(),
        numbered_pages=numbered_pages,
    )

    raw    = call_llm(system_prompt, user_prompt, label="KGCuratorAgent")
    result = parse_and_validate(raw, KGCuratorResult, label="KGCuratorAgent")

    print(f"  [KGCuratorAgent] status={result.status} | "
          f"add_e={len(result.add_entities)} add_r={len(result.add_relationships)} | "
          f"remove_e={len(result.remove_entities)} remove_r={len(result.remove_relationships)} | "
          f"update_e={len(result.update_entities)} update_r={len(result.update_relationships)}")
    return result


# Backward-compatible alias
kg_completeness_judge = kg_curator_agent


# ─────────────────────────────────────────────────────────────────────────────
# 6. Contradiction spotting agent
# ─────────────────────────────────────────────────────────────────────────────

def contradiction_spotting_agent(
    kg: KnowledgeGraph,
    document_text: str,
) -> ContradictionResult:
    system_prompt = get_system_prompt("contradiction_spotting")
    user_prompt = render(
        "contradiction_spotting.j2",
        kg=kg.to_serialisable(),
        document_text=document_text,
    )

    raw    = call_llm(system_prompt, user_prompt, label="ContradictionAgent")
    result = parse_and_validate(raw, ContradictionResult, label="ContradictionAgent")
    print(f"  [ContradictionAgent] {len(result.contradictions)} contradiction(s)")
    return result
