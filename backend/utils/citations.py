from __future__ import annotations


def build_citation_address(doc_id: str, clause_id: str, pointer: str) -> str:
    return f"CITE::{doc_id}::{clause_id}::{pointer}"
