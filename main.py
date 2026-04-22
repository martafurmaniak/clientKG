"""
main.py — entry point for the two-phase KG extraction pipeline.

Phase 1 — Client History KG
    Extracts a full KG from the client history document using the multi-agent
    pipeline (entity extraction → relationships → stray node → curator loop →
    contradiction spotting).

Phase 2 — Corroboration KG (optional)
    For each *_summarized.json found in the sibling ocr_corr/ folder, extracts
    a per-document KG using page summaries + document summary as context.
    Each document KG is saved alongside the source file as <name>_kg.json.

────────────────────────────────────────────────────────────────
CHOOSE YOUR RUN MODE  (edit the block below)
────────────────────────────────────────────────────────────────

  RUN_MODE = "mock"
      Uses the built-in John Smith example.  No extra files needed.

  RUN_MODE = "real"
      Reads two JSON files you supply:
        ONTOLOGY_PATH  — path to your ontology JSON
        DOCUMENT_PATH  — path to your paged-HTML client history JSON
                         (corroboration docs are discovered automatically
                          from the sibling ocr_corr/ folder)

────────────────────────────────────────────────────────────────
"""

import json
from pathlib import Path

from input_loader import load_mock, load_real, PipelineInputs
from orchestrator import run_pipeline
from schemas import KnowledgeGraph
from corroboration_loader import (
    load_corroboration_docs_real,
    load_corroboration_docs_mock,
)
from corroboration_pipeline import run_corroboration_phase
from ontology_utils import GlobalIDRegistry
from mock_data import MOCK_CORROBORATION_DOCS

# ══════════════════════════════════════════════════════════════
#  ▶  CONFIGURE HERE
# ══════════════════════════════════════════════════════════════

RUN_MODE = "mock"          # "mock"  |  "real"

# Only used when RUN_MODE = "real"
ONTOLOGY_PATH = "data/ontology.json"
DOCUMENT_PATH = "data/profiles/MsX/ocr_doc/client_history.json"

# Set to False to skip Phase 2 (useful for quick testing of Phase 1 only)
RUN_CORROBORATION = True

# ══════════════════════════════════════════════════════════════


def _load_inputs() -> PipelineInputs:
    if RUN_MODE == "mock":
        print("  Mode      : MOCK (built-in John Smith example)")
        return load_mock()
    elif RUN_MODE == "real":
        print("  Mode      : REAL")
        print(f"  Ontology  : {ONTOLOGY_PATH}")
        print(f"  Document  : {DOCUMENT_PATH}")
        return load_real(ONTOLOGY_PATH, DOCUMENT_PATH)
    else:
        raise ValueError(f"Unknown RUN_MODE '{RUN_MODE}'. Use 'mock' or 'real'.")


def _output_dir() -> Path:
    """Returns the output directory and creates it if needed."""
    base = Path(__file__).parent / "output" / RUN_MODE
    base.mkdir(parents=True, exist_ok=True)
    return base


def _save_phase1_outputs(result: dict, output_dir: Path) -> None:
    kg_path            = output_dir / "final_kg.json"
    contradiction_path = output_dir / "contradiction_report.json"
    gap_path           = output_dir / "ontology_gap_report.json"

    kg_path.write_text(json.dumps(result["kg"], indent=2))
    contradiction_path.write_text(json.dumps(result["contradiction_report"], indent=2))
    gap_path.write_text(json.dumps(result.get("ontology_gap_report", {}), indent=2))

    print(f"\n✅  Phase 1 outputs saved:")
    print(f"   Knowledge graph      → {kg_path}")
    print(f"   Contradiction report → {contradiction_path}")
    print(f"   Ontology gap report  → {gap_path}")

    _print_contradiction_summary(result["contradiction_report"])
    _print_gap_summary(result.get("ontology_gap_report", {}))


def _print_page_preview(pages: list[str]) -> None:
    print("\n  Page previews (first 120 chars each):")
    for i, page in enumerate(pages):
        preview = page.replace("\n", " ")[:120]
        tag = " +lookahead" if "[LOOKAHEAD" in page else ""
        print(f"    Page {i:>3}: {preview!r}{tag}")


def _print_contradiction_summary(cr: dict) -> None:
    print("\n" + "═" * 70)
    print("  CONTRADICTION REPORT SUMMARY")
    print("═" * 70)
    print(f"\nAssessment: {cr.get('assessment', 'N/A')}")
    contradictions = cr.get("contradictions", [])
    if contradictions:
        print(f"\nContradictions ({len(contradictions)}):")
        for i, c in enumerate(contradictions, 1):
            print(f"  {i}. {c.get('description', '')}")
            print(f"     Evidence : {c.get('evidence', '')}")
    else:
        print("\nNo contradictions found.")


def _print_gap_summary(gap_report: dict) -> None:
    gaps = gap_report.get("gaps", [])
    if not gaps:
        return
    print("\n" + "═" * 70)
    print("  ONTOLOGY GAP REPORT")
    print("═" * 70)
    print(f"\n{gap_report.get('recommendation', '')}")
    print(f"\nGaps found ({len(gaps)}):")
    for i, g in enumerate(gaps, 1):
        print(f"  {i}. Entity: {g.get('entity_id', 'unknown')}")
        print(f"     Reasoning : {g.get('reasoning', '')}")
        if g.get("evidence"):
            print(f"     Evidence  : {g['evidence']}")


def main() -> None:
    print("\n🚀  KG Extraction Pipeline — Phase 1: Client History")
    print("─" * 60)

    # ── Phase 1: Client history KG ───────────────────────────────────────────
    inputs = _load_inputs()

    print(f"  Pages     : {inputs.page_count}")
    print(f"  Doc chars : {len(inputs.document_text)}")
    print(f"  Entities  : {list(inputs.entity_ontology.keys())}")
    print(f"  Rel types : {list(inputs.relationship_ontology.keys())}")

    if RUN_MODE == "real":
        _print_page_preview(inputs.document_pages)

    # Single global registry for the entire run — survives across phases
    id_registry = GlobalIDRegistry()

    phase1_result = run_pipeline(
        document_text         = inputs.document_text,
        document_pages        = inputs.document_pages,
        entity_ontology       = inputs.entity_ontology,
        relationship_ontology = inputs.relationship_ontology,
        id_registry           = id_registry,
    )
    # Registry is returned populated with all history KG entity IDs
    id_registry = phase1_result.get("id_registry", id_registry)

    output_dir = _output_dir()
    _save_phase1_outputs(phase1_result, output_dir)

    if not RUN_CORROBORATION:
        print("\n  (Corroboration phase skipped — RUN_CORROBORATION=False)")
        return

    # ── Phase 2: Corroboration KGs ───────────────────────────────────────────
    print("\n\n🚀  KG Extraction Pipeline — Phase 2: Corroboration Documents")
    print("─" * 60)

    # Reconstruct the history KG as a KnowledgeGraph object for passing as context
    history_kg = KnowledgeGraph.model_validate(phase1_result["kg"])

    # Load corroboration documents
    if RUN_MODE == "mock":
        corr_docs = load_corroboration_docs_mock(MOCK_CORROBORATION_DOCS)
    else:
        corr_docs = load_corroboration_docs_real(DOCUMENT_PATH)

    if not corr_docs:
        print("  No corroboration documents found — Phase 2 complete.")
        return

    print(f"  Found {len(corr_docs)} corroboration document(s).")

    corr_results = run_corroboration_phase(
        docs                  = corr_docs,
        history_kg            = history_kg,
        entity_ontology       = inputs.entity_ontology,
        relationship_ontology = inputs.relationship_ontology,
        id_registry           = id_registry,   # continues from history phase
    )

    print(f"\n✅  Phase 2 complete — {len(corr_results)} corroboration KG(s) saved.")
    for source_file, doc_kg in corr_results.items():
        print(f"   {source_file}: {len(doc_kg.entities)} entities, "
              f"{len(doc_kg.relationships)} relationships")


if __name__ == "__main__":
    main()
