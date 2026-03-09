from __future__ import annotations

import json
from pathlib import Path

_REGISTRY: dict[str, str] | None = None


def _load_registry() -> dict[str, str]:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    registry_path = Path(__file__).resolve().parents[2] / "data" / "document_registry.json"
    entries = json.loads(registry_path.read_text(encoding="utf-8"))
    _REGISTRY = {entry["id"]: entry["file_path"] for entry in entries}
    return _REGISTRY


def _clause_fragment(clause_id: str) -> str:
    """Strip parenthetical sub-paragraph markers for use as a URL fragment.

    '6.2.1(5)' -> '6.2.1'
    'Table 3.4' -> 'Table 3.4'
    """
    idx = clause_id.find("(")
    return clause_id[:idx].strip() if idx > 0 else clause_id.strip()


def clause_ref(
    doc_id: str,
    clause_id: str,
    title: str,
    *,
    pointer: str | None = None,
) -> dict[str, str]:
    """Build a clause_references entry with registry-resolved pointer.

    For docs in the registry, the pointer is auto-built from the registered
    file_path.  For docs not in the registry (e.g. structural_mechanics),
    pass ``pointer=`` explicitly.
    """
    if pointer is None:
        registry = _load_registry()
        file_path = registry.get(doc_id)
        if file_path is not None:
            filename = file_path.rsplit("/", 1)[-1]
            pointer = f"{filename}#{_clause_fragment(clause_id)}"
        else:
            pointer = f"{doc_id}#{clause_id}"

    return {
        "doc_id": doc_id,
        "clause_id": clause_id,
        "title": title,
        "pointer": pointer,
    }
