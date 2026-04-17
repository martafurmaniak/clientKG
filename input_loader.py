"""
input_loader.py — loads and normalises pipeline inputs from two sources:

  MODE "mock"
  ───────────
  Uses the built-in mock_data constants directly.
  Entity/relationship ontologies are in the internal flat format already
  used by the agents.

  MODE "real"
  ───────────
  Reads:
    • an ontology JSON file whose schema is:
        {
          "entities": {
            "Person": { "description": "...", "attributes": { "fullName": {...}, ... } },
            ...
          },
          "relationships": {
            "BENEFICIAL_OWNER_OF": { "from": [...], "to": [...], "attributes": {...} },
            ...
          }
        }

    • a document JSON file that is the output of a document intelligence
      pipeline, shaped as:
        [
          { "page_number": 0, "offset": 0, "page_text": "<p>...</p><p>..." },
          { "page_number": 1, "offset": 0, "page_text": "<p>...</p>..." },
          ...
        ]

  The loader converts both external formats into the internal formats expected
  by the pipeline:

    entity_ontology      — dict[str, {"description": str, "attributes": list[str]}]
    relationship_ontology — dict[str, {"description": str, "from": list, "to": list}]
    document_text        — single concatenated plain-text string (HTML tags stripped)
    document_pages       — list[str], one entry per page, each with a
                           "lookahead" consisting of the first <p>…</p> paragraph
                           of the following page appended at the end.

Page lookahead rationale
────────────────────────
When the relationship extraction agent processes one page at a time, a sentence
split across a page boundary would otherwise be invisible to both pages.
Appending the first paragraph of page N+1 to the text of page N ensures that
boundary context is always present in at least one processing window.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

from mock_data import (
    DOCUMENT_TEXT   as MOCK_DOCUMENT_TEXT,
    DOCUMENT_PAGES  as MOCK_DOCUMENT_PAGES,
    ENTITY_ONTOLOGY as MOCK_ENTITY_ONTOLOGY,
    RELATIONSHIP_ONTOLOGY as MOCK_RELATIONSHIP_ONTOLOGY,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public container
# ─────────────────────────────────────────────────────────────────────────────

class PipelineInputs(NamedTuple):
    document_text:         str          # full doc as plain text
    document_pages:        list[str]    # per-page text with lookahead
    entity_ontology:       dict         # internal flat format
    relationship_ontology: dict         # internal flat format
    source_mode:           str          # "mock" | "real"
    page_count:            int


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace to plain text."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _first_paragraph(html: str) -> str:
    """
    Extract the content of the first <p>…</p> block from an HTML string.
    Returns an empty string if no paragraph is found.
    """
    match = re.search(r"<p[^>]*>(.*?)</p>", html, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _strip_html(match.group(0))


# ─────────────────────────────────────────────────────────────────────────────
# Ontology conversion  (external → internal)
# ─────────────────────────────────────────────────────────────────────────────

def _convert_entity_ontology(raw: dict) -> dict:
    """
    Convert the external entity ontology format to the internal one.

    External:
        {
          "Person": {
            "description": "A natural person",
            "attributes": { "fullName": { ... }, "dateOfBirth": { ... } }
          }
        }

    Internal:
        {
          "Person": {
            "description": "A natural person",
            "attributes": ["fullName", "dateOfBirth"]
          }
        }

    Key normalisation: entity type names are kept as-is (the agents receive
    them verbatim in the ontology and use them as KG keys).
    """
    converted: dict = {}
    for entity_type, body in raw.items():
        attrs_raw = body.get("attributes", {})
        if isinstance(attrs_raw, dict):
            attrs = list(attrs_raw.keys())
        elif isinstance(attrs_raw, list):
            # Already a list — accept as-is
            attrs = [str(a) for a in attrs_raw]
        else:
            attrs = []

        converted[entity_type] = {
            "description": body.get("description", ""),
            "attributes": attrs,
        }
    return converted


def _convert_relationship_ontology(raw: dict) -> dict:
    """
    Convert the external relationship ontology format to the internal one.

    External:
        {
          "BENEFICIAL_OWNER_OF": {
            "from": ["Person"],
            "to":   ["Organisation"],
            "attributes": { ... }
          }
        }

    Internal:
        {
          "BENEFICIAL_OWNER_OF": {
            "description": "",
            "from": ["Person"],
            "to":   ["Organisation"]
          }
        }
    """
    converted: dict = {}
    for rel_type, body in raw.items():
        converted[rel_type] = {
            "description": body.get("description", ""),
            "from": body.get("from", []),
            "to":   body.get("to",   []),
        }
    return converted


# ─────────────────────────────────────────────────────────────────────────────
# Document page processing  (external → internal, with lookahead)
# ─────────────────────────────────────────────────────────────────────────────

def _process_pages(raw_pages: list[dict]) -> tuple[str, list[str]]:
    """
    Convert the external paged-HTML document into:
      • document_text  — full document as concatenated plain text
      • document_pages — list of per-page plain-text strings, each with
                         the first paragraph of the following page appended
                         as lookahead context.

    The external page objects are sorted by page_number before processing
    so that out-of-order arrays are handled safely.
    """
    # Sort by page_number (field may be spelled "page_number" or "page_numer")
    def _page_num(p: dict) -> int:
        return int(p.get("page_number", p.get("page_numer", 0)))

    sorted_pages = sorted(raw_pages, key=_page_num)

    plain_pages: list[str] = [_strip_html(p.get("page_text", "")) for p in sorted_pages]
    html_pages:  list[str] = [p.get("page_text", "") for p in sorted_pages]

    document_text = "\n\n".join(plain_pages)

    document_pages_with_lookahead: list[str] = []
    for i, plain in enumerate(plain_pages):
        if i < len(html_pages) - 1:
            lookahead = _first_paragraph(html_pages[i + 1])
            if lookahead:
                page_with_context = (
                    plain
                    + "\n\n[LOOKAHEAD — first paragraph of next page]\n"
                    + lookahead
                )
            else:
                page_with_context = plain
        else:
            page_with_context = plain  # last page: no lookahead
        document_pages_with_lookahead.append(page_with_context)

    return document_text, document_pages_with_lookahead


# ─────────────────────────────────────────────────────────────────────────────
# Public loader functions
# ─────────────────────────────────────────────────────────────────────────────

def load_mock() -> PipelineInputs:
    """Return the built-in mock inputs unchanged."""
    return PipelineInputs(
        document_text         = MOCK_DOCUMENT_TEXT,
        document_pages        = MOCK_DOCUMENT_PAGES,
        entity_ontology       = MOCK_ENTITY_ONTOLOGY,
        relationship_ontology = MOCK_RELATIONSHIP_ONTOLOGY,
        source_mode           = "mock",
        page_count            = len(MOCK_DOCUMENT_PAGES),
    )


def load_real(ontology_path: str | Path, document_path: str | Path) -> PipelineInputs:
    """
    Load and convert real external inputs.

    Parameters
    ──────────
    ontology_path  : path to the ontology JSON file
    document_path  : path to the paged-HTML document JSON file

    Returns a fully normalised PipelineInputs ready to pass to run_pipeline().
    """
    ontology_path  = Path(ontology_path)
    document_path  = Path(document_path)

    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
    if not document_path.exists():
        raise FileNotFoundError(f"Document file not found: {document_path}")

    raw_ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    raw_document = json.loads(document_path.read_text(encoding="utf-8"))

    # Validate top-level structure
    if "entities" not in raw_ontology or "relationships" not in raw_ontology:
        raise ValueError(
            "Ontology JSON must have top-level keys 'entities' and 'relationships'."
        )
    if not isinstance(raw_document, list):
        raise ValueError(
            "Document JSON must be a list of page objects with 'page_text' fields."
        )

    entity_ontology       = _convert_entity_ontology(raw_ontology["entities"])
    relationship_ontology = _convert_relationship_ontology(raw_ontology["relationships"])
    document_text, document_pages = _process_pages(raw_document)

    return PipelineInputs(
        document_text         = document_text,
        document_pages        = document_pages,
        entity_ontology       = entity_ontology,
        relationship_ontology = relationship_ontology,
        source_mode           = "real",
        page_count            = len(document_pages),
    )
