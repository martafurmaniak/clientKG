"""
ontology_utils.py — ontology-aware utilities used across the pipeline.

Responsibilities
────────────────
1. Stable ID prefix mapping
   Each entity type gets a short, consistent prefix so IDs are human-readable
   and stable across runs.  Derived automatically from the ontology type name
   with a small set of well-known overrides for common banking types.

   Person           → P
   Organization     → O
   Client Profile   → CP
   Asset            → AS
   Account          → A
   Transaction      → T
   CorporateEvent   → CE
   Any other type   → first two uppercase letters of the type name

2. Full ontology serialisation for prompts
   build_entity_prompt_ontology() and build_relationship_prompt_ontology()
   produce rich dicts that include attribute names, their types/descriptions,
   and the allowed relationship directions — giving the LLM everything it
   needs to produce schema-compliant output.

3. ID counter state
   IDCounter is a simple counter that hands out the next stable ID for a
   given prefix.  It is seeded from the existing KG so new IDs never clash
   with already-assigned ones.
"""

from __future__ import annotations

import re
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Stable ID prefix mapping
# ─────────────────────────────────────────────────────────────────────────────

# Well-known overrides for common banking/KYC ontology types
_KNOWN_PREFIXES: dict[str, str] = {
    "person":          "P",
    "naturalPerson":   "P",
    "organisation":    "O",
    "organization":    "O",
    "legalEntity":     "O",
    "company":         "O",
    "clientprofile":   "CP",
    "clientProfile":   "CP",
    "asset":           "AS",
    "account":         "A",
    "bankaccount":     "A",
    "transaction":     "T",
    "corporateevent":  "CE",
    "corporateEvent":  "CE",
    "fund":            "F",
    "trust":           "TR",
}


def get_id_prefix(entity_type: str) -> str:
    """
    Return the stable ID prefix for an entity type.

    Lookup order:
      1. Well-known override table (case-insensitive)
      2. First uppercase letter(s) from the type name
         - CamelCase → initials  e.g. "BeneficialOwner" → "BO"
         - Single word → first two uppercase chars e.g. "person" → "PE"
    """
    key = entity_type.replace(" ", "").lower()
    if key in _KNOWN_PREFIXES:
        return _KNOWN_PREFIXES[key]

    # CamelCase initials
    initials = re.sub(r"[^A-Z]", "", entity_type)
    if len(initials) >= 2:
        return initials

    # Fallback: first two chars uppercased
    return entity_type[:2].upper()


# ─────────────────────────────────────────────────────────────────────────────
# ID counter — seeds from existing KG to avoid clashes
# ─────────────────────────────────────────────────────────────────────────────

class IDCounter:
    """
    Hands out the next stable ID for each entity type prefix.
    Pre-seeded from an existing KG so new IDs never collide.

    Usage:
        counter = IDCounter.from_kg(existing_kg, entity_ontology)
        new_id  = counter.next("Person")   # → "P3" if P1, P2 already exist
    """

    def __init__(self, counts: dict[str, int] | None = None) -> None:
        self._counts: dict[str, int] = defaultdict(int, counts or {})

    @classmethod
    def from_kg(cls, kg, entity_ontology: dict) -> "IDCounter":
        """Seed counter from IDs already present in a KnowledgeGraph."""
        counts: dict[str, int] = defaultdict(int)
        for entity in kg.entities:
            prefix = get_id_prefix(entity.type)
            # Extract the numeric suffix from the ID  e.g. "P3" → 3
            m = re.search(r"(\d+)$", entity.id)
            if m:
                n = int(m.group(1))
                if n > counts[prefix]:
                    counts[prefix] = n
        return cls(counts)

    def next(self, entity_type: str) -> str:
        prefix = get_id_prefix(entity_type)
        self._counts[prefix] += 1
        return f"{prefix}{self._counts[prefix]}"


# ─────────────────────────────────────────────────────────────────────────────
# Next-ID map for prompt injection
# ─────────────────────────────────────────────────────────────────────────────

def get_next_id_map(entity_ontology: dict, kg=None) -> dict[str, str]:
    """
    Return a dict mapping each entity type to the next ID the LLM should use.

    If a KG is provided, seeds from the highest existing numeric suffix so
    new IDs are guaranteed not to clash with already-assigned ones.

    Example (with P1, P2 already in KG):
        {"Person": "P3", "Organisation": "O1", "Account": "A1", ...}

    This dict is injected into every extraction prompt so the LLM always
    knows exactly which ID to start from for each type.
    """
    import re
    from collections import defaultdict

    # Find current max numeric suffix per prefix from the existing KG
    max_counts: dict[str, int] = defaultdict(int)
    if kg is not None:
        for entity in kg.entities:
            prefix = get_id_prefix(entity.type)
            m = re.search(r"(\d+)$", entity.id)
            if m:
                n = int(m.group(1))
                if n > max_counts[prefix]:
                    max_counts[prefix] = n

    result: dict[str, str] = {}
    for etype in entity_ontology:
        prefix = get_id_prefix(etype)
        result[etype] = f"{prefix}{max_counts[prefix] + 1}"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full ontology serialisation for prompts
# ─────────────────────────────────────────────────────────────────────────────

def build_entity_prompt_ontology(
    entity_ontology: dict,
    next_id_map: dict[str, str] | None = None,
) -> list[dict]:
    """
    Convert the internal entity ontology into a rich list suitable for
    injection into LLM prompts.

    Each entry includes:
      - type         : entity type name
      - description  : human-readable description
      - id_prefix    : the stable ID prefix  e.g. "P", "A", "AS"
      - next_id      : the exact next ID the LLM should assign  e.g. "P3"
                       (derived from the current KG state — no clashes)
      - attributes   : attribute definitions

    Example output (with P1, P2 already in KG):
      [
        {
          "type": "Person",
          "description": "A natural person ...",
          "id_prefix": "P",
          "next_id": "P3",
          "attributes": {"fullName": "string", "dateOfBirth": "date", ...}
        },
        ...
      ]
    """
    label_priority = [
        "fullName", "legalName", "name", "companyName", "accountId",
        "transactionId", "title", "identifier",
    ]

    result = []
    for etype, body in entity_ontology.items():
        prefix  = get_id_prefix(etype)
        next_id = (next_id_map or {}).get(etype, f"{prefix}1")

        # Determine which attribute should be used as the entity label
        attrs_raw = body.get("attributes", {})
        attr_names = list(attrs_raw.keys()) if isinstance(attrs_raw, dict) else list(attrs_raw)
        label_from = next(
            (a for a in label_priority if a in attr_names),
            attr_names[0] if attr_names else "id"
        )

        result.append({
            "type":        etype,
            "description": body.get("description", ""),
            "id_prefix":   prefix,
            "next_id":     next_id,
            "label_from":  label_from,   # which attribute to use as the label
            "attributes":  attrs_raw,    # full dict (name → type) or list
        })
    return result


def build_relationship_prompt_ontology(relationship_ontology: dict) -> list[dict]:
    """
    Convert the internal relationship ontology into a rich list for prompts.

    Each entry includes:
      - type        : relationship type name  e.g. "EMPLOYED_BY"
      - description : human-readable description
      - from        : allowed source entity types
      - to          : allowed target entity types
      - attributes  : attribute names (with types/descriptions if available)

    Example output:
      [
        {
          "type": "EMPLOYED_BY",
          "description": "Employment relationship",
          "from": ["Person"],
          "to": ["Person", "Organization"],
          "attributes": ["jobTitle", "startDate", "endDate"]
        },
        ...
      ]
    """
    result = []
    for rtype, body in relationship_ontology.items():
        from_raw = body.get("from", [])
        to_raw   = body.get("to",   [])
        result.append({
            "type":        rtype,
            "description": body.get("description", ""),
            "from":        from_raw if isinstance(from_raw, list) else [from_raw],
            "to":          to_raw   if isinstance(to_raw,   list) else [to_raw],
            "attributes":  body.get("attributes", []),
        })
    return result
