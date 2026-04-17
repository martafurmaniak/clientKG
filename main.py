"""
main.py — entry point.

Run:
    python main.py

Outputs:
    final_kg.json              — the completed knowledge graph
    contradiction_report.json  — contradiction / inconsistency findings
"""

import json
import sys
from pathlib import Path

from mock_data import DOCUMENT_TEXT, DOCUMENT_PAGES, ENTITY_ONTOLOGY, RELATIONSHIP_ONTOLOGY
from orchestrator import run_pipeline


def main() -> None:
    print("\n🚀  Starting KG extraction pipeline...")
    print(f"   Document length : {len(DOCUMENT_TEXT)} characters")
    print(f"   Document pages  : {len(DOCUMENT_PAGES)}")
    print(f"   Entity types    : {list(ENTITY_ONTOLOGY.keys())}")
    print(f"   Relationship types: {list(RELATIONSHIP_ONTOLOGY.keys())}")

    result = run_pipeline(
        document_text=DOCUMENT_TEXT,
        document_pages=DOCUMENT_PAGES,
        entity_ontology=ENTITY_ONTOLOGY,
        relationship_ontology=RELATIONSHIP_ONTOLOGY,
    )

    output_dir = Path(__file__).parent

    if result["status"] == "halted_ontology_gap":
        print("\n⛔  Pipeline halted due to ontology gap. See output above.")
        gap_path = output_dir / "ontology_gap_report.json"
        gap_path.write_text(json.dumps(result["stray_node_report"], indent=2))
        print(f"   Gap report saved → {gap_path}")
        sys.exit(1)

    # Save final outputs
    kg_path = output_dir / "final_kg.json"
    contradiction_path = output_dir / "contradiction_report.json"

    kg_path.write_text(json.dumps(result["kg"], indent=2))
    contradiction_path.write_text(json.dumps(result["contradiction_report"], indent=2))

    print(f"\n✅  Outputs saved:")
    print(f"   Knowledge graph      → {kg_path}")
    print(f"   Contradiction report → {contradiction_path}")

    # Pretty-print a summary
    print("\n" + "═" * 70)
    print("  CONTRADICTION REPORT SUMMARY")
    print("═" * 70)
    cr = result["contradiction_report"]
    print(f"\nAssessment: {cr.get('assessment', 'N/A')}")
    if cr["contradictions"]:
        print(f"\nContradictions ({len(cr['contradictions'])}):")
        for i, c in enumerate(cr["contradictions"], 1):
            print(f"  {i}. {c.get('description', '')}")
            print(f"     Evidence : {c.get('evidence', '')}")
    else:
        print("\nNo contradictions found.")


if __name__ == "__main__":
    main()
