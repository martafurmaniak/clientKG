"""
schemas.py — Pydantic v2 models for all agent inputs and outputs.

Single format throughout
────────────────────────
All agents, prompts, and the final output share the same KG structure:

  {
    "entities": [
      {"id": "person_1", "type": "Person", "label": "John Smith",
       "attributes": {"fullName": "John Smith", "age": 45}}
    ],
    "relationships": [
      {"source": "person_1", "target": "asset_1", "type": "OWNS",
       "attributes": {"since": "2019", "evidence": "..."}}
    ]
  }

This means:
  - LLM responses use the same format as the final output file
  - No translation layer needed — to_serialisable() IS the output format
  - to_output_format() is kept as an alias for backward compatibility
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Core entity / relationship primitives
# ─────────────────────────────────────────────────────────────────────────────

class Entity(BaseModel):
    id: str
    type: str = ""
    label: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def absorb_extra_fields(cls, data: Any) -> Any:
        """
        Accept both the canonical format:
            {"id": "p1", "type": "Person", "label": "John", "attributes": {...}}
        and the flat LLM format where attributes are top-level:
            {"id": "p1", "type": "Person", "label": "John", "fullName": "John", "age": 45}
        Extra keys beyond id/type/label are packed into attributes.
        """
        if not isinstance(data, dict):
            return data
        known = {"id", "type", "label", "attributes"}
        attrs = dict(data.get("attributes", {}))
        for k, v in data.items():
            if k not in known:
                attrs[k] = v
        return {
            "id":         data.get("id", ""),
            "type":       data.get("type", ""),
            "label":      data.get("label", ""),
            "attributes": attrs,
        }

    def to_dict(self) -> dict:
        """Canonical serialisable form."""
        return {
            "id":         self.id,
            "type":       self.type,
            "label":      self.label,
            "attributes": self.attributes,
        }


class Relationship(BaseModel):
    source: str
    target: str
    type: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type", mode="before")
    @classmethod
    def upper_type(cls, v: Any) -> str:
        return str(v).upper().strip()

    @model_validator(mode="before")
    @classmethod
    def absorb_extra_and_aliases(cls, data: Any) -> Any:
        """
        Accept:
          - canonical:  {"source": ..., "target": ..., "type": ..., "attributes": {...}}
          - legacy:     {"from_id": ..., "to_id": ..., "type": ..., ...}
          - flat attrs: any extra keys beyond source/target/type packed into attributes
        Also absorbs evidence/reasoning as attributes if present at top level.
        """
        if not isinstance(data, dict):
            return data
        # Normalise source/target aliases
        source = data.get("source") or data.get("from_id") or data.get("from") or ""
        target = data.get("target") or data.get("to_id") or data.get("to") or ""
        known  = {"source", "target", "type", "attributes",
                  "from_id", "to_id", "from", "to", "id"}
        attrs  = dict(data.get("attributes", {}))
        for k, v in data.items():
            if k not in known:
                attrs[k] = v   # captures evidence, reasoning, and all ontology attrs
        return {
            "source":     source,
            "target":     target,
            "type":       data.get("type", ""),
            "attributes": attrs,
        }

    def to_dict(self) -> dict:
        return {
            "source":     self.source,
            "target":     self.target,
            "type":       self.type,
            "attributes": self.attributes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# KG container
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeGraph(BaseModel):
    entities:      list[Entity]       = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_inputs(cls, data: Any) -> Any:
        """
        Accept both the canonical flat-list format and the legacy
        dict-keyed format {"Person": [...], "Organisation": [...]}
        so the pipeline is backward compatible during the transition.
        """
        if not isinstance(data, dict):
            return data

        raw_entities = data.get("entities", [])

        # Legacy dict-keyed format → flat list
        if isinstance(raw_entities, dict):
            flat: list[dict] = []
            for etype, elist in raw_entities.items():
                if isinstance(elist, list):
                    for e in elist:
                        if isinstance(e, dict):
                            flat.append({**e, "type": e.get("type") or etype})
                        else:
                            flat.append(e)
            data = {**data, "entities": flat}

        return data

    def entity_ids(self) -> set[str]:
        return {e.id for e in self.entities}

    def all_entities_flat(self) -> list[tuple[str, Entity]]:
        """Returns (entity_type, entity) pairs — kept for agent compatibility."""
        return [(e.type, e) for e in self.entities]

    def to_serialisable(self) -> dict:
        """
        Canonical serialisable form — this IS the output format.
        Used for LLM prompts and the final output file.
        """
        return {
            "entities":      [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
        }

    def to_output_format(self) -> dict:
        """Alias for to_serialisable() — same format throughout."""
        return self.to_serialisable()


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction agent output
# ─────────────────────────────────────────────────────────────────────────────

class EntityExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    entities_to_remove: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalise(cls, data: Any) -> Any:
        """
        Accept three formats the LLM might return:

        1. Canonical:
           {"entities": [...], "entities_to_remove": [...]}

        2. Flat list (bare):
           [{"id": ..., "type": ..., "label": ..., ...}, ...]

        3. Legacy dict-keyed (old prompt format):
           {"Person": [...], "Organisation": [...], "entities_to_remove": [...]}
        """
        if isinstance(data, list):
            return {"entities": data, "entities_to_remove": []}

        if not isinstance(data, dict):
            return data

        if "entities" in data:
            # Already canonical — but entities might be dict-keyed inside
            raw = data["entities"]
            if isinstance(raw, dict):
                flat = []
                for etype, elist in raw.items():
                    if isinstance(elist, list):
                        for e in elist:
                            if isinstance(e, dict):
                                flat.append({**e, "type": e.get("type") or etype})
                return {"entities": flat,
                        "entities_to_remove": data.get("entities_to_remove", [])}
            return data

        # Legacy dict-keyed — every key that isn't entities_to_remove is a type bucket
        non_type_keys = {"entities_to_remove"}
        flat = []
        for k, v in data.items():
            if k in non_type_keys:
                continue
            if isinstance(v, list):
                for e in v:
                    if isinstance(e, dict):
                        flat.append({**e, "type": e.get("type") or k})
        return {
            "entities":           flat,
            "entities_to_remove": data.get("entities_to_remove", []),
        }


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
    page_number: Optional[int] = None


class MissingRelationship(BaseModel):
    type: str
    from_: str = Field(alias="from")
    to: str
    reasoning: str
    evidence: Optional[str] = None
    page_number: Optional[int] = None

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
        if any([self.missing_entities, self.missing_relationships,
                self.hallucinated_entities, self.hallucinated_relationships]):
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
