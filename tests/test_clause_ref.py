from tools.mcp.clause_ref import clause_ref, _clause_fragment


def test_auto_resolution_from_registry() -> None:
    result = clause_ref("ec3.en1993-1-1.2005", "6.2.1(5)", "Von Mises")
    assert result["doc_id"] == "ec3.en1993-1-1.2005"
    assert result["clause_id"] == "6.2.1(5)"
    assert result["title"] == "Von Mises"
    assert result["pointer"] == "EN_1993-1-1-2005.json#6.2.1"


def test_en1993_1_8_resolution() -> None:
    result = clause_ref("ec3.en1993-1-8.2005", "Table 3.4", "Bolt shear")
    assert result["pointer"] == "EN_1993-1-8-2005.json#Table 3.4"


def test_fragment_stripping() -> None:
    assert _clause_fragment("6.2.1(5)") == "6.2.1"
    assert _clause_fragment("Table 3.4") == "Table 3.4"
    assert _clause_fragment("BB.2.1") == "BB.2.1"
    assert _clause_fragment("B.1") == "B.1"


def test_explicit_pointer_override() -> None:
    result = clause_ref(
        "structural_mechanics", "cantilever_beam", "Cantilever",
        pointer="structural_mechanics#cantilever",
    )
    assert result["pointer"] == "structural_mechanics#cantilever"


def test_unknown_doc_id_fallback() -> None:
    result = clause_ref("ec0.en1990.2002", "A1.4.3", "Deflections")
    assert result["pointer"] == "ec0.en1990.2002#A1.4.3"


def test_return_dict_shape() -> None:
    result = clause_ref("ec3.en1993-1-1.2005", "6.3.1", "Buckling")
    assert set(result.keys()) == {"doc_id", "clause_id", "title", "pointer"}
