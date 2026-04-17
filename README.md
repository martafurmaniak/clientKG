# Knowledge Graph Extraction — Multi-Agent System

A fully local, end-to-end multi-agent pipeline that extracts a structured
knowledge graph from a client history document using **qwen2.5:7b** via Ollama.

---

## File Structure

```
kg_system/
├── main.py            ← entry point — run this
├── orchestrator.py    ← drives the full pipeline and phase transitions
├── agents.py          ← all 7 sub-agent implementations
├── llm_utils.py       ← thin Ollama wrapper + JSON extractor
├── mock_data.py       ← sample document, entity ontology, relationship ontology
└── README.md
```

---

## Prerequisites

1. **Ollama** installed and running — https://ollama.com
2. **qwen2.5:7b** pulled:
   ```bash
   ollama pull qwen2.5:7b
   ```
3. Python 3.11+ with the Ollama client:
   ```bash
   pip install ollama
   ```

---

## Run

```bash
cd kg_system
python main.py
```

Expected runtime: ~5–15 minutes on a laptop (CPU inference, depends on RAM).

---

## Pipeline Phases

```
Phase 1 · Entity Extraction
  PeopleOrgsAgent ──┐
  AssetsAgent       ├──► KGConsolidationAgent ──► KG (entities only)
  TransactionsAgent ┘

Phase 2 · Relationship Extraction
  RelationshipAgent (per page) ──► KGConsolidationAgent ──► KG (+ relationships)

Phase 3 · Stray Node Feedback Loop  [repeats until clean or ontology gap]
  StrayNodeAgent ──► (resolved) ──► KGConsolidationAgent ──► back to StrayNodeAgent
                └──► (ontology_gap) ──► HALT — user must refine ontology

Phase 4 · KG Completeness Loop  [repeats until complete, max 3 iterations]
  KGCompletenessJudge
    ├── complete      ──► Phase 5
    └── needs_improvement ──► re-run entity + relationship agents ──► KGConsolidation
                              ──► remove hallucinations ──► back to Phase 3

Phase 5 · Contradiction Spotting
  ContradictionAgent ──► final report
```

---

## Outputs

| File | Description |
|------|-------------|
| `final_kg.json` | Full knowledge graph with `entities` and `relationships` |
| `contradiction_report.json` | Contradictions found in the client profile |
| `ontology_gap_report.json` | Only created if pipeline halts due to an ontology gap |

---

## Customising

- **Document**: replace `DOCUMENT_TEXT` and `DOCUMENT_PAGES` in `mock_data.py`
- **Ontologies**: extend `ENTITY_ONTOLOGY` or `RELATIONSHIP_ONTOLOGY` in `mock_data.py`
- **Model**: change `MODEL` in `llm_utils.py` (any Ollama-hosted model works)
- **Loop limits**: adjust `MAX_STRAY_NODE_ITERATIONS` and `MAX_COMPLETENESS_ITERATIONS` in `orchestrator.py`
