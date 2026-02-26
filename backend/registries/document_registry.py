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


def _extract_clause_rows(payload: Any, entry_id: str) -> list[dict[str, Any]]:
    # Supported shapes:
    # 1) {"clauses":[{...}, ...]}
    # 2) [{...}, {...}, ...]
    # 3) [[{...}, ...], [{...}, ...], ...]  (common OCR export layout)
    if isinstance(payload, dict):
        clauses_raw = payload.get("clauses", [])
        if not isinstance(clauses_raw, list):
            raise ValueError(f"Document {entry_id} has invalid 'clauses' payload.")
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


def load_clauses_for_entry(project_root: Path, entry: DocumentRegistryEntry) -> list[ClauseRecord]:
    file_path = _resolve_data_path(project_root, entry.file_path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    clauses_raw = _extract_clause_rows(payload, entry.id)

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

    return clauses


def load_all_clauses(
    project_root: Path, registry_entries: list[DocumentRegistryEntry]
) -> list[ClauseRecord]:
    all_clauses: list[ClauseRecord] = []
    for entry in registry_entries:
        all_clauses.extend(load_clauses_for_entry(project_root, entry))
    return all_clauses
