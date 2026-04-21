"""
prompt_loader.py — Jinja2 template loader + agent config reader.

Structure
─────────
  prompts/          — Jinja2 user prompt templates (have dynamic variables)
  agent_config.yaml — system prompts + custom instructions for every agent

Public API
──────────
  render(template_name, **variables)  → rendered user prompt string
  get_system_prompt(agent_key)        → system prompt string for an agent
  get_instructions(agent_key)         → custom instructions string or None
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from ontology_utils import build_entity_prompt_ontology, build_relationship_prompt_ontology, get_next_id_map

_BASE_DIR   = Path(__file__).parent
_PROMPTS_DIR = _BASE_DIR / "prompts"
_CONFIG_PATH = _BASE_DIR / "agent_config.yaml"

# ── Jinja2 environment ────────────────────────────────────────────────────────
_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=StrictUndefined,   # raise immediately on missing variables
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)

def _tojson(value: object, indent: int = None) -> str:
    return json.dumps(value, indent=indent, ensure_ascii=False, default=str)

_env.filters["tojson"] = _tojson

# ── Agent config (loaded once at import time) ─────────────────────────────────
_config: dict = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))


# ── Public API ────────────────────────────────────────────────────────────────

def render(template_name: str, **variables) -> str:
    """
    Load and render a user prompt template from prompts/.

    Parameters
    ──────────
    template_name : filename e.g. "entity_extraction_initial.j2"
    **variables   : template variables passed to Jinja2
    """
    return _env.get_template(template_name).render(**variables).strip()


def get_system_prompt(agent_key: str) -> str:
    """
    Return the system prompt for an agent from agent_config.yaml.

    Parameters
    ──────────
    agent_key : top-level key in agent_config.yaml e.g. "people_orgs"
    """
    try:
        return _config[agent_key]["system_prompt"].strip()
    except KeyError:
        raise KeyError(
            f"Agent key '{agent_key}' not found in agent_config.yaml. "
            f"Available keys: {list(_config.keys())}"
        )


def get_instructions(agent_key: str) -> str | None:
    """
    Return custom instructions for an agent from agent_config.yaml,
    or None if the value is empty/absent.

    Parameters
    ──────────
    agent_key : top-level key in agent_config.yaml e.g. "people_orgs"
    """
    val = _config.get(agent_key, {}).get("custom_instructions")
    if not val:
        return None
    stripped = str(val).strip()
    return stripped if stripped else None


def render_with_ontology(
    template_name: str,
    entity_ontology: dict,
    relationship_ontology: dict,
    existing_kg=None,
    **variables,
) -> str:
    """
    Like render(), but automatically converts the raw ontology dicts into
    the rich prompt-ready format and injects current ID state so the LLM
    always knows the exact next ID to assign for each entity type.

    Parameters
    ──────────
    entity_ontology       : internal entity ontology dict
    relationship_ontology : internal relationship ontology dict
    existing_kg           : optional KnowledgeGraph — used to compute next_id
                            per entity type so the LLM never reuses an existing ID

    Templates receive:
      entity_ontology_rich       — list[dict] with type/description/id_prefix/
                                   next_id/attributes
      relationship_ontology_rich — list[dict] with type/description/from/to/attributes
    """
    next_ids = get_next_id_map(entity_ontology, kg=existing_kg)
    return render(
        template_name,
        entity_ontology_rich=build_entity_prompt_ontology(entity_ontology, next_id_map=next_ids),
        relationship_ontology_rich=build_relationship_prompt_ontology(relationship_ontology),
        entity_ontology=entity_ontology,
        relationship_ontology=relationship_ontology,
        **variables,
    )
