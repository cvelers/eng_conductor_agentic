from pathlib import Path

from backend.registries.tool_registry import load_tool_registry


def test_tool_registry_contains_expected_tools() -> None:
    root = Path(__file__).resolve().parents[1]
    registry_path = root / "tools" / "tool_registry.json"

    tools = load_tool_registry(registry_path)
    names = {tool.tool_name for tool in tools}

    assert "section_classification_ec3" in names
    assert "member_resistance_ec3" in names
    assert "interaction_check_ec3" in names
    assert "ipe_moment_resistance_ec3" in names
