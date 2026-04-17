"""
schemas.py — Pydantic v2 models for every agent input and output.

All agent functions accept and return these models (or plain dicts that are
validated into them).  This gives us:
  • runtime validation of LLM outputs
  • clear contracts between agents
  • automatic coercion of minor type mismatches (e.g. int age as string)
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Shared entity types
# ─────────────────────────────────────────────────────────────────────────────

class Person(BaseModel):
    id: str
    name: str
    age: Optional[int] = None
    role: Optional[str] = None
    relationship_to_client: Optional[str] = None

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, v: Any) -> Optional[int]:
        if v is None or v == "" or v == "null":
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None


class Organisation(BaseModel):
    id: str
    name: str
    type: Optional[str] = None
    founded_year: Optional[int] = None
    industry: Optional[str] = None

    @field_validator("founded_year", mode="before")
    @classmethod
    def coerce_year(cls, v: Any) -> Optional[int]:
        if v is None or v == "" or v == "null":
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            return None


class Asset(BaseModel):
    id: str
    asset_id: Optional[str] = None
    asset_type: Optional[str] = None
    value_chf: Optional[float] = None
    owner: Optional[str] = None

    @field_validator("value_chf", mode="before")
    @classmethod
    def coerce_value(cls, v: Any) -> Optional[float]:
        if v is None or v == "" or v == "null":
            return None
        if isinstance(v, str):
            v = v.replace(",", "").replace("CHF", "").strip()
        try:
            return float(v)
        except (ValueError, TypeError):
            return None


class Transaction(BaseModel):
    id: str
    transaction_id: Optional[str] = None
    date: Optional[str] = None
    amount_chf: Optional[float] = None
    from_account: Optional[str] = None
    to_party: Optional[str] = None
    description: Optional[str] = None

    @field_validator("amount_chf", mode="before")
    @classmethod
    def coerce_amount(cls, v: Any) -> Optional[float]:
        if v is None or v == "" or v == "null":
            return None
        if isinstance(v, str):
            v = v.replace(",", "").replace("CHF", "").strip()
        try:
            return float(v)
        except (ValueError, TypeError):
            return None


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

class Entities(BaseModel):
    people: list[Person] = Field(default_factory=list)
    organisations: list[Organisation] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    transactions: list[Transaction] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalise_keys(cls, data: Any) -> Any:
        """Accept common LLM variations like 'organization', 'asset', etc."""
        if not isinstance(data, dict):
            return data
        aliases = {
            "person": "people",
            "organization": "organisations",
            "organization": "organisations",
            "asset": "assets",
            "transaction": "transactions",
        }
        return {aliases.get(k, k): v for k, v in data.items()}


class KnowledgeGraph(BaseModel):
    entities: Entities = Field(default_factory=Entities)
    relationships: list[Relationship] = Field(default_factory=list)

    def entity_ids(self) -> set[str]:
        ids: set[str] = set()
        for lst in [
            self.entities.people,
            self.entities.organisations,
            self.entities.assets,
            self.entities.transactions,
        ]:
            for e in lst:
                ids.add(e.id)
        return ids

    def to_serialisable(self) -> dict:
        """Return a plain dict safe for json.dumps and LLM prompts."""
        return self.model_dump(exclude_none=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction agent output
# ─────────────────────────────────────────────────────────────────────────────

class EntityExtractionResult(BaseModel):
    """Output of any entity extraction agent (initial or improvement run)."""
    people: list[Person] = Field(default_factory=list)
    organisations: list[Organisation] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    transactions: list[Transaction] = Field(default_factory=list)

    # IDs of entities the agent recommends removing (improvement runs only)
    entities_to_remove: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalise_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        aliases = {
            "person": "people",
            "organization": "organisations",
            "asset": "assets",
            "transaction": "transactions",
        }
        return {aliases.get(k, k): v for k, v in data.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Relationship extraction agent output
# ─────────────────────────────────────────────────────────────────────────────

class RelationshipExtractionResult(BaseModel):
    """Output of the relationship extraction agent."""
    relationships: list[Relationship] = Field(default_factory=list)
    relationships_to_remove: list[str] = Field(default_factory=list)  # rel IDs to drop

    @model_validator(mode="before")
    @classmethod
    def accept_bare_list(cls, data: Any) -> Any:
        """LLM sometimes returns a bare list instead of a wrapped object."""
        if isinstance(data, list):
            return {"relationships": data, "relationships_to_remove": []}
        return data


# ─────────────────────────────────────────────────────────────────────────────
# KG Consolidation agent output
# ─────────────────────────────────────────────────────────────────────────────

class KGConsolidationResult(BaseModel):
    entities: Entities = Field(default_factory=Entities)
    relationships: list[Relationship] = Field(default_factory=list)

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

    @field_validator("entity_type", mode="before")
    @classmethod
    def lower_type(cls, v: Any) -> str:
        return str(v).lower().strip()


class MissingRelationship(BaseModel):
    type: str
    from_: str = Field(alias="from")
    to: str
    reasoning: str
    evidence: Optional[str] = None

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
