from pathlib import Path

from backend.registries.document_registry import load_all_clauses, load_document_registry


def test_document_registry_loads_and_resolves_files() -> None:
    root = Path(__file__).resolve().parents[1]
    registry_path = root / "data" / "document_registry.json"

    entries = load_document_registry(registry_path)
    assert entries, "Document registry should not be empty"
    assert any(entry.id == "ec3.en1993-1-1.2005" for entry in entries)

    clauses = load_all_clauses(root, entries)
    assert len(clauses) >= 50, f"Expected 50+ structured clauses, got {len(clauses)}"
    assert any(
        clause.clause_id == "6.2.5" and "bending" in clause.clause_title.lower()
        for clause in clauses
    ), "Should find clause 6.2.5 Bending moment"
    assert any(
        clause.clause_id == "5.5.2" and "classification" in clause.clause_title.lower()
        for clause in clauses
    ), "Should find clause 5.5.2 Classification"
