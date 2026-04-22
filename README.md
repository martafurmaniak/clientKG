# KG Extraction Pipeline

A multi-agent system that extracts structured Knowledge Graphs (KGs) from financial client history documents and corroboration sources, then reconciles the two sets of graphs to assess evidence quality.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Configuration](#configuration)
4. [Pipeline Phases](#pipeline-phases)
   - [Phase 1 — Client History KG](#phase-1--client-history-kg)
   - [Phase 2 — Corroboration KGs](#phase-2--corroboration-kgs)
   - [Phase 3 — Reconciliation](#phase-3--reconciliation)
5. [KG Data Format](#kg-data-format)
6. [Ontology Format](#ontology-format)
7. [ID Assignment](#id-assignment)
8. [Prompt Management](#prompt-management)
9. [Agent Reference](#agent-reference)
10. [Refinement Loop](#refinement-loop)
11. [Outputs](#outputs)
12. [Dependencies](#dependencies)

---

## Overview

The pipeline takes a client profile document and a set of corroboration documents (bank statements, commercial register extracts, etc.) and produces:

- A structured KG from the client history document
- A structured KG for each corroboration document
- A reconciled KG annotating each relationship in the history graph with its corroboration status

All three phases use the same shared ontology, ensuring entity types, relationship types, attribute names, and ID prefixes are consistent across every graph.

---

## Directory Structure

```
kg_system/
├── main.py                      Entry point — configures and runs all phases
├── orchestrator.py              Phase 1 pipeline driver (client history)
├── agents.py                    All LLM agent functions + deterministic helpers
├── schemas.py                   Pydantic v2 models for all data structures
├── llm_utils.py                 LLM backend abstraction (Azure OpenAI / Ollama)
├── input_loader.py              Loads and normalises ontology + document inputs
├── prompt_loader.py             Jinja2 template loader + agent config reader
├── ontology_utils.py            Stable ID registry, prefix mapping, ontology builders
├── kg_refinement_loop.py        Shared curator + stray-node + compliance loop
├── corroboration_loader.py      Discovers and loads *_summarized.json files
├── corroboration_pipeline.py    Phase 2 pipeline driver (corroboration documents)
├── reconciliation.py            Phase 3 — reconciles history KG vs corroboration KGs
├── mock_data.py                 Built-in test data (John Smith / Alpine Bank)
├── agent_config.yaml            System prompts + custom instructions per agent
└── prompts/
    ├── entity_extraction_initial.j2
    ├── entity_extraction_improvement.j2
    ├── relationship_extraction_initial.j2
    ├── relationship_extraction_improvement.j2
    ├── kg_curator.j2
    ├── stray_node.j2
    ├── subgraph_connector.j2
    ├── corroboration_extraction.j2
    └── contradiction_spotting.j2
```

---

## Configuration

All user-facing settings live at the top of `main.py`:

```python
RUN_MODE           = "mock"    # "mock" | "real"
ONTOLOGY_PATH      = "data/ontology.json"
DOCUMENT_PATH      = "data/profiles/MsX/ocr_doc/client_history.json"
RUN_CORROBORATION  = True      # set False to run Phase 1 only
RUN_RECONCILIATION = True      # set False to skip Phase 3
```

LLM backend settings live in `llm_utils.py`:

```python
BACKEND           = "azure"    # "azure" | "ollama"
AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT")
OLLAMA_MODEL      = "qwen2.5:7b"
```

Custom extraction instructions per agent are configured in `agent_config.yaml` — fill in the `custom_instructions` field under the relevant agent key to append domain-specific guidance to every extraction call for that agent.

Reconciliation entailment rules (which multi-hop paths imply which direct relationships) are configured in `reconciliation.py`:

```python
ENTAILMENT_RULES = {
    ("OWNS", "ACCOUNT_AT"): "ACCOUNT_AT",
    ("LEADS", "BENEFICIAL_OWNER_OF"): "BENEFICIAL_OWNER_OF",
}
MAX_PATH_LENGTH = 3
```

---

## Pipeline Phases

### Phase 1 — Client History KG

**Input:** a paged JSON document + an ontology JSON file.

**Steps:**

```
1. Entity Extraction
   PeopleOrgsAgent ──┐
   AssetsAgent       ├──► KGConsolidationAgent ──► initial KG
   TransactionsAgent ─┘

2. Relationship Extraction
   RelationshipAgent (page by page, with entity index for cross-page awareness)
   ──► KGConsolidationAgent ──► KG with relationships

3–4. Refinement Loop  [up to MAX_COMPLETENESS_ITERATIONS = 3]
   Per iteration:
     a. KGCuratorAgent        — produces add / remove / update action lists
     b. Targeted improvement  — entity/relationship agents run only on pages
                                 where items are missing; update_entities go
                                 directly to consolidation (no agent re-run)
     c. KGConsolidationAgent  — merges actions into KG
     d. Stray-node + Compliance loop  [up to MAX_STRAY_NODE_ITERATIONS * 2]
         i.  StrayNodeAgent         — finds unconnected nodes, proposes relationships
         ii. OntologyComplianceCheck — strips disallowed attributes, removes
                                       relationships with bad source/target IDs
                                       or wrong entity types
         iii. SubgraphConnector     — detects disconnected components, connects
                                       smallest to largest iteratively
         Repeats until stray=clean AND compliance=no-changes AND single-component
     Early exit if curator iteration produces zero net change

5. ContradictionAgent  — identifies logical inconsistencies in the final KG
```

**Output:** `final_kg.json`, `contradiction_report.json`, `ontology_gap_report.json`

---

### Phase 2 — Corroboration KGs

**Input:** all `*_summarized.json` files under the `ocr_corr/` sibling directory of `DOCUMENT_PATH`. Each file has the structure:

```json
{
  "pages": [
    {
      "page_number": 0,
      "page_text": "<raw OCR>",
      "page_summary": "<LLM page summary>"
    }
  ],
  "document_summary": "<LLM document-level summary>",
  "meta": {
    "source_file": "bank_statement_q1.pdf",
    "summarized_at": "2024-04-01T10:00:00Z"
  }
}
```

**Per-document steps:**

```
For each page:
  1. Entity extraction using page_summary + document_summary as context
     - LLM returns { reused_ids: [...], new_entities: [...] }
     - reused_ids: existing history/prior-doc entity IDs the LLM recognised
       → fetched directly from cumulative_kg by ID (no label guessing)
     - new_entities: genuinely new entities
       → IDs assigned via GlobalIDRegistry (never clashes with any prior ID)
     - All entities (reused + new) included in doc_kg (self-contained graph)

  2. Relationship extraction using all available entity IDs as context

  3. KGConsolidationAgent (per page, with ontology validation)

After all pages:
  4. Same Refinement Loop as Phase 1 (shared via kg_refinement_loop.py)

Save per-document KG as <source_file_stem>_kg.json in ocr_corr/
```

**Cross-document entity continuity:** a `cumulative_kg` grows across all documents in processing order. Document N+1 receives the full cumulative entity pool so it can reuse entities from Document N by referencing their IDs. The `GlobalIDRegistry` ensures no ID is ever reused across the entire run.

---

### Phase 3 — Reconciliation

**Input:** history KG + all corroboration KGs.

**Entity alignment:** entities are matched across graphs by normalised label (case-insensitive, whitespace-stripped). IDs do not need to match.

**For each relationship in the history KG**, all corroboration documents are checked and the best result is kept:

| Status | Condition |
|--------|-----------|
| `directly_corroborated` | Exact relationship (same type, aligned endpoints) found in a corroboration graph |
| `path_corroborated` | A multi-hop path in the corroboration graph matches an `ENTAILMENT_RULES` entry that implies this relationship type |
| `partially_corroborated` | Both endpoints are aligned in the corroboration graph but no path match found |
| `uncorroborated` | At least one endpoint has no aligned entity in any corroboration graph |

**Path-corroborated relationships** surface any intermediate nodes that exist in the corroboration graph path but are absent from the history graph as `missing_intermediaries`.

**Output:** `reconciled_kg.json` (history KG with `status` and `corroborating_doc` added to every relationship's `attributes`) and `reconciliation_report.json` (summary counts + per-relationship detail + missing intermediaries list).

---

## KG Data Format

All graphs — history, corroboration, and reconciled — use the same flat structure:

```json
{
  "entities": [
    {
      "id": "P1",
      "type": "Person",
      "label": "John Smith",
      "attributes": {
        "fullName": "John Smith",
        "dateOfBirth": "1979-04-03"
      }
    }
  ],
  "relationships": [
    {
      "source": "P1",
      "target": "A1",
      "type": "OWNS",
      "attributes": {
        "evidence": "John holds savings account ACC-001",
        "ownershipType": "sole",
        "since": null,
        "status": "directly_corroborated",
        "corroborating_doc": "alpine_bank_statement_q1_2024.pdf"
      }
    }
  ]
}
```

`status` and `corroborating_doc` are only present on relationships in `reconciled_kg.json`.

---

## Ontology Format

The ontology JSON file has two top-level keys:

```json
{
  "entities": {
    "Person": {
      "description": "A natural person involved in the client relationship",
      "attributes": {
        "fullName": "string",
        "aliases": ["string"],
        "dateOfBirth": "date",
        "nationality": "string"
      }
    }
  },
  "relationships": {
    "EMPLOYED_BY": {
      "description": "Employment relationship",
      "from": "Person",
      "to": "Person|Organization",
      "attributes": {
        "jobTitle": "string",
        "startDate": "date",
        "employmentType": "Part-time|Full-time"
      }
    }
  }
}
```

Attribute values can be:
- A plain type string: `"date"`, `"string"`, `"number"`, `"integer"`
- A pipe-separated list of allowed values: `"Part-time|Full-time"`
- A nested object: `{"type": "date"}` or `{"type": "string", "description": "..."}`
- A list of allowed values: `["Part-time", "Full-time"]` (rendered as `Part-time|Full-time` in prompts)

The `from` / `to` fields accept a type name, a list `["Person", "Organisation"]`, or a pipe-separated string `"Person|Organisation"`.

---

## ID Assignment

Entity IDs are stable, human-readable, and globally unique across the entire run.

**Prefix mapping** (defined in `ontology_utils.py`):

| Entity type | Prefix | Example IDs |
|-------------|--------|-------------|
| Person | P | P1, P2, P3 |
| Organisation / Organization | O | O1, O2 |
| Account | A | A1, A2 |
| Transaction | T | T1, T2 |
| CorporateEvent | CE | CE1, CE2 |
| Asset | AS | AS1, AS2 |
| ClientProfile | CP | CP1 |
| Fund | F | F1 |
| Trust | TR | TR1 |
| Other (CamelCase) | initials | BeneficialOwner → BO1 |

The `GlobalIDRegistry` is created once in `main.py` before Phase 1, seeded with zero, and passed through every agent call in both phases. Every assigned ID is immediately registered so subsequent calls — in improvement runs, refinement loops, and corroboration documents — always pick up from the current maximum with no possibility of collision.

LLM-assigned IDs in extraction responses are **discarded** and replaced by the registry. The LLM's IDs serve only as internal placeholders within a single response.

---

## Prompt Management

All prompts are managed as Jinja2 templates under `prompts/` and system prompts + custom instructions in `agent_config.yaml`. No prompt strings exist in Python code.

**`prompt_loader.py`** exposes:
- `render(template_name, **vars)` — renders a user prompt template
- `render_with_ontology(template_name, entity_ontology, relationship_ontology, ...)` — renders with automatically-built rich ontology context injected as `entity_ontology_rich` and `relationship_ontology_rich`
- `get_system_prompt(agent_key)` — loads system prompt from `agent_config.yaml`
- `get_instructions(agent_key)` — loads optional custom instructions from `agent_config.yaml`

**To add custom extraction instructions** for an agent, edit `agent_config.yaml`:

```yaml
people_orgs:
  system_prompt: >
    You are a precise knowledge-graph entity extraction engine. ...
  custom_instructions: |
    - Always extract Client Profile separately from the underlying Person.
    - If a person appears only as a counterparty, still extract them.
```

---

## Agent Reference

All agents are defined in `agents.py`. Each LLM agent calls `call_llm()` and parses the response with `parse_and_validate()` against a Pydantic schema.

| Agent | Function | Schema | Notes |
|-------|----------|--------|-------|
| PeopleOrgsAgent | `people_and_orgs_agent()` | `EntityExtractionResult` | Extracts Person, Organisation, ClientProfile types |
| AssetsAgent | `assets_agent()` | `EntityExtractionResult` | Extracts Account, Asset types |
| TransactionsAgent | `transactions_agent()` | `EntityExtractionResult` | Extracts Transaction, CorporateEvent types |
| RelationshipAgent | `relationship_extraction_agent()` | `RelationshipExtractionResult` | Page-by-page; carries cross-page entity index |
| KGCuratorAgent | `kg_curator_agent()` | `KGCuratorResult` | Returns add / remove / update action lists |
| StrayNodeAgent | `stray_node_agent()` | `StrayNodeResult` | Finds unconnected nodes; proposes bridging relationships |
| SubgraphConnector | `subgraph_connector_agent()` | `RelationshipExtractionResult` | Bridges disconnected components |
| ContradictionAgent | `contradiction_spotting_agent()` | `ContradictionResult` | Final consistency check |
| KGConsolidation | `kg_consolidation_agent()` | — | **Deterministic** — no LLM; pure Python merge with ontology validation |

**Deterministic helpers** (no LLM):
- `validate_kg()` — strips disallowed attributes, removes bad relationships
- `find_connected_components()` — union-find on relationship graph
- `_validate_ontology_compliance()` — enforces ontology on entities and relationships

---

## Refinement Loop

The shared refinement loop (`kg_refinement_loop.py`) is used by both Phase 1 and Phase 2. It receives a KG and document context and returns a refined KG.

**Outer loop** (`MAX_COMPLETENESS_ITERATIONS = 3`):

```
iteration N:
  1. KGCuratorAgent
     → add_entities      → targeted entity extraction agents (per page)
     → add_relationships → RelationshipAgent (per page)
     → remove_entities   → direct to consolidation
     → remove_relationships → direct to consolidation
     → update_entities   → direct to consolidation (no agent re-run)
     → update_relationships → direct to consolidation
  2. KGConsolidationAgent (merges all actions + runs ontology validation)
  3. Early exit if zero net change

  _run_stray_compliance_loop():
    inner loop (MAX_STRAY_NODE_ITERATIONS * 2):
      a. StrayNodeAgent  — resolve unconnected nodes
      b. validate_kg     — enforce ontology compliance
      c. find_connected_components
         → for each isolated component (smallest first):
              SubgraphConnector → bridge to main component
      repeat until clean + compliant + single component
```

**After the outer loop exits** (whether by completion or hitting max iterations), one final `_run_stray_compliance_loop()` pass runs unconditionally to catch any nodes introduced on the last iteration.

---

## Outputs

All outputs are saved to `output/<RUN_MODE>/`:

| File | Description |
|------|-------------|
| `final_kg.json` | Client history KG |
| `contradiction_report.json` | Contradictions found in history KG |
| `ontology_gap_report.json` | Entities with no valid relationship type available |
| `reconciled_kg.json` | History KG with corroboration status on every relationship |
| `reconciliation_report.json` | Summary counts + per-relationship detail + missing intermediaries |

Corroboration document KGs are saved in the `ocr_corr/` folder alongside their source files:

```
data/profiles/MsX/ocr_corr/
├── bank_statement_summarized.json     ← input
├── bank_statement_kg.json             ← output
├── commercial_register_summarized.json
└── commercial_register_kg.json
```

---

## Dependencies

```
pydantic>=2.0
jinja2
pyyaml
openai          # for Azure backend
ollama          # for local Ollama backend
```

Install:
```bash
pip install pydantic jinja2 pyyaml openai ollama
```

Run:
```bash
cd kg_system
python main.py
```
