"""
main.py — entry point for the KG extraction pipeline.

────────────────────────────────────────────────────────────────
CHOOSE YOUR RUN MODE  (edit the block below)
────────────────────────────────────────────────────────────────

  RUN_MODE = "mock"
      Uses the built-in John Smith example.  No extra files needed.

  RUN_MODE = "real"
      Reads two JSON files you supply:
        ONTOLOGY_PATH  — path to your ontology JSON
        DOCUMENT_PATH  — path to your paged-HTML document JSON

      File formats are described in input_loader.py.

────────────────────────────────────────────────────────────────
"""

import json
import sys
from pathlib import Path

from input_loader import load_mock, load_real, PipelineInputs
from orchestrator import run_pipeline

# ══════════════════════════════════════════════════════════════
#  ▶  CONFIGURE HERE
# ══════════════════════════════════════════════════════════════

RUN_MODE = "mock"          # "mock"  |  "real"

# Only used when RUN_MODE = "real"
ONTOLOGY_PATH = "data/ontology.json"
DOCUMENT_PATH = "data/document.json"

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


def main() -> None:
    print("\n🚀  KG Extraction Pipeline")
    print("─" * 40)

    inputs = _load_inputs()

    print(f"  Pages     : {inputs.page_count}")
    print(f"  Doc chars : {len(inputs.document_text)}")
    print(f"  Entities  : {list(inputs.entity_ontology.keys())}")
    print(f"  Rel types : {list(inputs.relationship_ontology.keys())}")

    if RUN_MODE == "real":
        _print_page_preview(inputs.document_pages)

    result = run_pipeline(
        document_text         = inputs.document_text,
        document_pages        = inputs.document_pages,
        entity_ontology       = inputs.entity_ontology,
        relationship_ontology = inputs.relationship_ontology,
    )

    output_dir = Path(__file__).parent
    _save_outputs(result, output_dir)


def _print_page_preview(pages: list[str]) -> None:
    """Print a short preview of each page so the user can verify loading."""
    print("\n  Page previews (first 120 chars each):")
    for i, page in enumerate(pages):
        preview = page.replace("\n", " ")[:120]
        has_lookahead = "[LOOKAHEAD" in page
        tag = " +lookahead" if has_lookahead else ""
        print(f"    Page {i:>3}: {preview!r}{tag}")


def _save_outputs(result: dict, output_dir: Path) -> None:
    kg_path            = output_dir / "final_kg.json"
    contradiction_path = output_dir / "contradiction_report.json"
    gap_path           = output_dir / "ontology_gap_report.json"

    kg_path.write_text(json.dumps(result["kg"], indent=2))
    contradiction_path.write_text(json.dumps(result["contradiction_report"], indent=2))
    gap_path.write_text(json.dumps(result.get("ontology_gap_report", {}), indent=2))

    print(f"\n✅  Outputs saved:")
    print(f"   Knowledge graph      → {kg_path}")
    print(f"   Contradiction report → {contradiction_path}")
    print(f"   Ontology gap report  → {gap_path}")

    _print_contradiction_summary(result["contradiction_report"])
    _print_gap_summary(result.get("ontology_gap_report", {}))


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


if __name__ == "__main__":
    main()
