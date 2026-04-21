"""
corroboration_loader.py — discovers and loads summarized corroboration documents.

Expected directory layout (real mode):
    data/profiles/MsX/
        ocr_doc/
            client_history.json          ← DOCUMENT_PATH points here
        ocr_corr/
            bank_statement_summarized.json
            tax_return_summarized.json
            ...

Each *_summarized.json has the structure:
    {
      "pages": [
        {
          "page_number": 0,
          "offset": 0,
          "page_text": "<raw OCR text>",
          "page_summary": "<LLM summary of the page>"
        },
        ...
      ],
      "document_summary": "<LLM summary of the full document>",
      "meta": {
        "summarized_at": "<ISO timestamp>",
        "source_file": "<original filename>"
      }
    }

For the mock mode, a small set of inline corroboration documents is provided
in mock_data.py (MOCK_CORROBORATION_DOCS).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CorroborationDoc:
    """A single loaded and parsed corroboration document."""
    source_file:      str        # from meta.source_file
    document_summary: str        # full-document LLM summary
    pages:            list[dict] # sorted page objects with page_number/page_text/page_summary
    output_path:      Path       # where the per-doc KG should be written


def _sort_pages(pages: list[dict]) -> list[dict]:
    return sorted(pages, key=lambda p: int(p.get("page_number", 0)))


def load_corroboration_docs_real(document_path: str | Path) -> list[CorroborationDoc]:
    """
    Given the DOCUMENT_PATH (client history), derive the corroboration folder
    (sibling ocr_corr/) and discover all *_summarized.json files there.

    Returns a list of CorroborationDoc objects ready for processing.
    """
    doc_path = Path(document_path)
    # Derive profile root: .../profiles/MsX/ocr_doc/file.json → .../profiles/MsX/
    corr_dir = doc_path.parent.parent / "ocr_corr"

    if not corr_dir.exists():
        print(f"  [CorrLoader] No corroboration directory found at {corr_dir} — skipping.")
        return []

    summarized_files = sorted(corr_dir.glob("*_summarized.json"))
    if not summarized_files:
        print(f"  [CorrLoader] No *_summarized.json files found in {corr_dir} — skipping.")
        return []

    docs: list[CorroborationDoc] = []
    for fpath in summarized_files:
        try:
            raw = json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [CorrLoader] ⚠ Failed to read {fpath.name}: {e} — skipping.")
            continue

        pages = raw.get("pages", [])
        if not pages:
            print(f"  [CorrLoader] ⚠ {fpath.name} has no pages — skipping.")
            continue

        meta        = raw.get("meta", {})
        source_file = meta.get("source_file", fpath.stem)
        doc_summary = raw.get("document_summary", "")

        # Output KG path: same folder, original source name + "_kg.json"
        stem        = Path(source_file).stem if source_file else fpath.stem.replace("_summarized", "")
        output_path = corr_dir / f"{stem}_kg.json"

        docs.append(CorroborationDoc(
            source_file      = source_file,
            document_summary = doc_summary,
            pages            = _sort_pages(pages),
            output_path      = output_path,
        ))
        print(f"  [CorrLoader] Loaded {fpath.name} → {len(pages)} pages, output → {output_path.name}")

    return docs


def load_corroboration_docs_mock(mock_docs: list[dict]) -> list[CorroborationDoc]:
    """
    Build CorroborationDoc objects from the inline MOCK_CORROBORATION_DOCS list.
    Output paths are relative to the current working directory under mock_output/.
    """
    output_dir = Path("mock_output") / "ocr_corr"
    output_dir.mkdir(parents=True, exist_ok=True)

    docs: list[CorroborationDoc] = []
    for raw in mock_docs:
        source_file = raw.get("meta", {}).get("source_file", "mock_doc")
        stem        = Path(source_file).stem
        docs.append(CorroborationDoc(
            source_file      = source_file,
            document_summary = raw.get("document_summary", ""),
            pages            = _sort_pages(raw.get("pages", [])),
            output_path      = output_dir / f"{stem}_kg.json",
        ))
    return docs
