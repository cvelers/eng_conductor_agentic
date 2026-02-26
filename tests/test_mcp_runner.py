from pathlib import Path

from backend.registries.tool_registry import load_tool_registry
from backend.tools.runner import MCPToolRunner


def test_mcp_runner_executes_registered_module_tool() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = load_tool_registry(root / "tools" / "tool_registry.json")
    runner = MCPToolRunner(project_root=root, registry=registry)

    payload = runner.run(
        "section_classification_ec3",
        {
            "section_name": "IPE300",
            "steel_grade": "S355",
        },
    )

    assert payload["status"] == "ok"
    assert payload["result"]["outputs"]["governing_class"] >= 1
