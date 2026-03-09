from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.utils.citations import build_citation_address


class ClauseLocator(BaseModel):
    id_field: str = "clause_id"
    title_field: str = "title"
    text_field: str = "text"
    pointer_field: str = "pointer"


class DocumentRegistryEntry(BaseModel):
    id: str
    title: str
    standard: str
    year_version: str
    file_path: str
    clauses_key: str = "clauses"
    coverage_notes: str
    clause_locator: ClauseLocator


class ClauseRecord(BaseModel):
    doc_id: str
    doc_title: str
    standard: str
    clause_id: str
    clause_title: str
    text: str
    keywords: list[str] = Field(default_factory=list)
    pointer: str

    @property
    def citation_address(self) -> str:
        return build_citation_address(self.doc_id, self.clause_id, self.pointer)


def load_document_registry(path: Path) -> list[DocumentRegistryEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Document registry must be a list.")
    return [DocumentRegistryEntry.model_validate(item) for item in payload]


def _resolve_data_path(project_root: Path, relative_path: str) -> Path:
    candidate = (project_root / relative_path).resolve()
    root = project_root.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError(f"Unsafe document path outside project root: {relative_path}")
    return candidate


def _extract_clause_rows(payload: Any, entry_id: str, clauses_key: str = "clauses") -> list[dict[str, Any]]:
    # Supported shapes:
    # 1) {"clauses":[{...}, ...]} or {"sections":[{...}, ...]} (key is configurable)
    # 2) [{...}, {...}, ...]
    # 3) [[{...}, ...], [{...}, ...], ...]  (common OCR export layout)
    if isinstance(payload, dict):
        clauses_raw = payload.get(clauses_key, [])
        if not isinstance(clauses_raw, list):
            raise ValueError(f"Document {entry_id} has invalid '{clauses_key}' payload.")
        rows: list[dict[str, Any]] = []
        for item in clauses_raw:
            if isinstance(item, dict):
                rows.append(item)
        return rows

    if isinstance(payload, list):
        rows = []
        for item in payload:
            if isinstance(item, dict):
                rows.append(item)
                continue
            if isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, dict):
                        rows.append(sub_item)
        return rows

    raise ValueError(f"Document {entry_id} has unsupported JSON root type: {type(payload).__name__}")


def _render_table_text(table: dict[str, Any]) -> str:
    """Render a table dict (headers + rows + footnotes) into readable text."""
    parts: list[str] = []
    headers = table.get("headers", [])
    rows = table.get("rows", [])
    if headers:
        parts.append(" | ".join(str(h) for h in headers))
        parts.append("-" * min(len(parts[0]), 120))
    for row in rows:
        if isinstance(row, list):
            parts.append(" | ".join(str(cell) for cell in row))
    for fn in table.get("footnotes", []):
        parts.append(f"NOTE: {fn}")
    return "\n".join(parts)


def _base_table_id(table_id: str) -> str:
    """Strip sheet/continuation suffixes to get the base table identifier.

    'Table 5.2 (sheet 1 of 3)'          -> 'Table 5.2'
    'Table 5.2 (sheet 2 of 3) - Angles' -> 'Table 5.2'
    'Table 3.1 (continued)'             -> 'Table 3.1'
    'Table A.1 (continued)'             -> 'Table A.1'
    """
    import re
    m = re.match(r"(Table\s+[\w.]+)", table_id)
    return m.group(1) if m else table_id


def _extract_tables(payload: Any, entry: DocumentRegistryEntry) -> list[ClauseRecord]:
    """Extract tables from the data file and convert to ClauseRecord entries.

    Multi-sheet and continued tables are consolidated under the base table_id
    so that a lookup for "Table 5.2" finds the combined content.
    """
    if not isinstance(payload, dict):
        return []
    tables_raw = payload.get("tables", [])
    if not isinstance(tables_raw, list):
        return []

    # Group table sheets by base table_id
    grouped: dict[str, list[dict[str, Any]]] = {}
    for table in tables_raw:
        if not isinstance(table, dict):
            continue
        raw_id = str(table.get("table_id", ""))
        if not raw_id:
            continue
        base_id = _base_table_id(raw_id)
        grouped.setdefault(base_id, []).append(table)

    filename = entry.file_path.rsplit("/", 1)[-1]
    records: list[ClauseRecord] = []
    for base_id, sheets in grouped.items():
        title = str(sheets[0].get("title", base_id))
        text_parts = [_render_table_text(s) for s in sheets]
        text = "\n\n".join(text_parts)
        records.append(
            ClauseRecord(
                doc_id=entry.id,
                doc_title=entry.title,
                standard=entry.standard,
                clause_id=base_id,
                clause_title=title,
                text=text,
                keywords=[],
                pointer=f"{filename}#{base_id}",
            )
        )
    return records


def load_clauses_for_entry(project_root: Path, entry: DocumentRegistryEntry) -> list[ClauseRecord]:
    file_path = _resolve_data_path(project_root, entry.file_path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    clauses_raw = _extract_clause_rows(payload, entry.id, entry.clauses_key)

    locator = entry.clause_locator
    clauses: list[ClauseRecord] = []
    for idx, row in enumerate(clauses_raw):
        clause_id = str(row.get(locator.id_field, f"unknown-{idx}"))
        title = str(row.get(locator.title_field, "Untitled clause"))
        text = str(row.get(locator.text_field, ""))
        pointer = str(row.get(locator.pointer_field, f"{entry.file_path}#clauses[{idx}]"))
        keywords = row.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []

        clauses.append(
            ClauseRecord(
                doc_id=entry.id,
                doc_title=entry.title,
                standard=entry.standard,
                clause_id=clause_id,
                clause_title=title,
                text=text,
                keywords=[str(k) for k in keywords],
                pointer=pointer,
            )
        )

    clauses.extend(_extract_tables(payload, entry))

    return clauses


def load_all_clauses(
    project_root: Path, registry_entries: list[DocumentRegistryEntry]
) -> list[ClauseRecord]:
    all_clauses: list[ClauseRecord] = []
    for entry in registry_entries:
        all_clauses.extend(load_clauses_for_entry(project_root, entry))
    return all_clauses
