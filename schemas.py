"""
schemas.py — Pydantic v2 models for all agent inputs and outputs.

Design note on entity types
────────────────────────────
The mock ontology uses fixed lowercase keys: "people", "organisations",
"assets", "transactions".

The real ontology uses arbitrary PascalCase keys: "Person", "Organisation",
"Transaction", etc. — whatever the user defines.

To handle both without duplicating logic, entities are stored as a
dict[str, list[Entity]] rather than as named fields.  The Entity model
itself is a flexible bag-of-attributes (a dict of str→Any) with only
"id" required.  This lets every agent work the same way regardless of
ontology source.

The KnowledgeGraph model therefore looks like:
    {
      "entities": {
        "Person":       [{"id": "person_1", "name": "John", ...}],
        "Organisation": [{"id": "org_1",    "name": "Alpine Bank", ...}]
      },
      "relationships": [...]
    }
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Core entity / relationship primitives
# ─────────────────────────────────────────────────────────────────────────────

class Entity(BaseModel):
    """
    A flexible entity node.  Only 'id' is required; all other attributes
    are stored in 'attributes' as a free-form dict so the model works for
    any ontology without schema changes.
    """
    id: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def absorb_extra_fields(cls, data: Any) -> Any:
        """
        LLMs return flat dicts like {"id": "p1", "name": "John", "age": 45}.
        We hoist 'id' to the top level and pack everything else into
        'attributes' so the schema stays clean.
        """
        if not isinstance(data, dict):
            return data
        known = {"id", "attributes"}
        entity_id = data.get("id", "")
        attrs = dict(data.get("attributes", {}))
        for k, v in data.items():
            if k not in known:
                attrs[k] = v
        return {"id": entity_id, "attributes": attrs}

    def to_flat_dict(self) -> dict:
        """Return {"id": ..., <attr_key>: <attr_value>, ...} for LLM prompts."""
        return {"id": self.id, **self.attributes}


class Relationship(BaseModel):
    id: str
    type: str
    from_id: str
    to_id: str
    evidence: Optional[str] = None
    reasoning: Optional[str] = None

    @field_validator("type", mode="before")
    @classmethod
    def upper_type(cls, v: Any) -> str:
        return str(v).upper().strip()


# ─────────────────────────────────────────────────────────────────────────────
# KG container
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeGraph(BaseModel):
    # entities: {"Person": [Entity, ...], "Organisation": [...], ...}
    entities: dict[str, list[Entity]] = Field(default_factory=dict)
    relationships: list[Relationship] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_entity_lists(cls, data: Any) -> Any:
        """
        Ensure every value in 'entities' is a list of Entity-compatible dicts,
        even if the LLM returned a bare dict or a list of flat attribute dicts.
        """
        if not isinstance(data, dict):
            return data
        raw_entities = data.get("entities", {})
        if isinstance(raw_entities, dict):
            coerced: dict[str, list] = {}
            for etype, elist in raw_entities.items():
                if isinstance(elist, list):
                    coerced[etype] = elist
                elif isinstance(elist, dict):
                    # Single entity returned as a dict — wrap it
                    coerced[etype] = [elist]
                else:
                    coerced[etype] = []
            data = {**data, "entities": coerced}
        return data

    def entity_ids(self) -> set[str]:
        ids: set[str] = set()
        for elist in self.entities.values():
            for e in elist:
                ids.add(e.id)
        return ids

    def all_entities_flat(self) -> list[tuple[str, Entity]]:
        """Yield (entity_type, entity) pairs across all types."""
        result = []
        for etype, elist in self.entities.items():
            for e in elist:
                result.append((etype, e))
        return result

    def to_serialisable(self) -> dict:
        """
        Return a plain dict for json.dumps / LLM prompts.
        Entities are serialised as flat dicts (id + attributes spread out).
        """
        return {
            "entities": {
                etype: [e.to_flat_dict() for e in elist]
                for etype, elist in self.entities.items()
            },
            "relationships": [
                r.model_dump(exclude_none=True)
                for r in self.relationships
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction agent output
# ─────────────────────────────────────────────────────────────────────────────

class EntityExtractionResult(BaseModel):
    """
    Output of any entity extraction agent (initial or improvement run).
    'entities' mirrors the KG format: dict keyed by entity type.
    """
    entities: dict[str, list[Entity]] = Field(default_factory=dict)
    entities_to_remove: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalise(cls, data: Any) -> Any:
        """
        Accept either:
          {"entities": {"Person": [...]}, "entities_to_remove": [...]}
          {"Person": [...], "Organisation": [...]}   ← bare entity map
        """
        if not isinstance(data, dict):
            return data

        known_top = {"entities", "entities_to_remove"}
        # If there's no "entities" key, assume the whole dict IS the entity map
        if "entities" not in data:
            entity_map = {k: v for k, v in data.items() if k not in {"entities_to_remove"}}
            return {
                "entities": entity_map,
                "entities_to_remove": data.get("entities_to_remove", []),
            }
        return data


# ─────────────────────────────────────────────────────────────────────────────
# Relationship extraction agent output
# ─────────────────────────────────────────────────────────────────────────────

class RelationshipExtractionResult(BaseModel):
    relationships: list[Relationship] = Field(default_factory=list)
    relationships_to_remove: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_bare_list(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"relationships": data, "relationships_to_remove": []}
        return data


# ─────────────────────────────────────────────────────────────────────────────
# KG Consolidation agent output  (same shape as KnowledgeGraph)
# ─────────────────────────────────────────────────────────────────────────────

class KGConsolidationResult(BaseModel):
    entities: dict[str, list[Entity]] = Field(default_factory=dict)
    relationships: list[Relationship] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_entity_lists(cls, data: Any) -> Any:
        # Reuse same logic as KnowledgeGraph
        if not isinstance(data, dict):
            return data
        raw_entities = data.get("entities", {})
        if isinstance(raw_entities, dict):
            coerced: dict[str, list] = {}
            for etype, elist in raw_entities.items():
                if isinstance(elist, list):
                    coerced[etype] = elist
                elif isinstance(elist, dict):
                    coerced[etype] = [elist]
                else:
                    coerced[etype] = []
            data = {**data, "entities": coerced}
        return data

    def to_knowledge_graph(self) -> KnowledgeGraph:
        return KnowledgeGraph(entities=self.entities, relationships=self.relationships)


# ─────────────────────────────────────────────────────────────────────────────
# Stray node agent output
# ─────────────────────────────────────────────────────────────────────────────

class OntologyGap(BaseModel):
    entity_id: str
    reasoning: str
    evidence: Optional[str] = None


class StrayNodeResult(BaseModel):
    status: Literal["clean", "resolved", "ontology_gap"] = "clean"
    new_relationships: list[Relationship] = Field(default_factory=list)
    ontology_gaps: list[OntologyGap] = Field(default_factory=list)

    @model_validator(mode="after")
    def infer_status(self) -> "StrayNodeResult":
        if self.ontology_gaps:
            self.status = "ontology_gap"
        elif self.new_relationships:
            self.status = "resolved"
        else:
            self.status = "clean"
        return self


# ─────────────────────────────────────────────────────────────────────────────
# KG Completeness judge output
# ─────────────────────────────────────────────────────────────────────────────

class MissingEntity(BaseModel):
    entity_type: str
    name: str
    reasoning: str
    evidence: Optional[str] = None
    page_number: Optional[int] = None   # 0-based page index where the entity appears


class MissingRelationship(BaseModel):
    type: str
    from_: str = Field(alias="from")
    to: str
    reasoning: str
    evidence: Optional[str] = None
    page_number: Optional[int] = None   # 0-based page index where the relationship appears

    model_config = {"populate_by_name": True}

    @field_validator("type", mode="before")
    @classmethod
    def upper_type(cls, v: Any) -> str:
        return str(v).upper().strip()


class HallucinatedEntity(BaseModel):
    entity_id: str
    reasoning: str
    evidence: Optional[str] = None


class HallucinatedRelationship(BaseModel):
    rel_id: str
    reasoning: str
    evidence: Optional[str] = None


class KGCompletenessResult(BaseModel):
    status: Literal["complete", "needs_improvement"] = "needs_improvement"
    missing_entities: list[MissingEntity] = Field(default_factory=list)
    missing_relationships: list[MissingRelationship] = Field(default_factory=list)
    hallucinated_entities: list[HallucinatedEntity] = Field(default_factory=list)
    hallucinated_relationships: list[HallucinatedRelationship] = Field(default_factory=list)
    reasoning: str = ""

    @model_validator(mode="after")
    def derive_status(self) -> "KGCompletenessResult":
        has_issues = any([
            self.missing_entities,
            self.missing_relationships,
            self.hallucinated_entities,
            self.hallucinated_relationships,
        ])
        if has_issues:
            self.status = "needs_improvement"
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Contradiction spotting agent output
# ─────────────────────────────────────────────────────────────────────────────

class Contradiction(BaseModel):
    description: str
    entities_involved: list[str] = Field(default_factory=list)
    evidence: str = ""


class ContradictionResult(BaseModel):
    contradictions: list[Contradiction] = Field(default_factory=list)
    assessment: str = ""
