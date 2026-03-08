import pytest
from pathlib import Path

from backend.registries.tool_registry import load_tool_registry
from backend.tools.runner import MCPToolRunner


@pytest.fixture(scope="module")
def runner():
    root = Path(__file__).resolve().parents[1]
    registry = load_tool_registry(root / "tools" / "tool_registry.json")
    r = MCPToolRunner(project_root=root, registry=registry)
    yield r
    r.shutdown()


def test_mcp_runner_executes_registered_module_tool(runner: MCPToolRunner) -> None:
    payload = runner.run(
        "section_classification_ec3",
        {
            "section_name": "IPE300",
            "steel_grade": "S355",
        },
    )

    assert payload["status"] == "ok"
    assert payload["result"]["outputs"]["governing_class"] >= 1


def test_mcp_runner_rejects_unknown_tool(runner: MCPToolRunner) -> None:
    with pytest.raises(ValueError, match="not registered"):
        runner.run("nonexistent_tool", {})
