"""MCP protocol server exposing all engineering calculation tools.

Reads the tool registry, dynamically imports each tool module, and
serves them over stdio transport using the MCP protocol (JSON-RPC).
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Tool discovery ──────────────────────────────────────────────────

def _parse_run_cli_call(script_path: Path) -> tuple[str | None, str | None]:
    """Parse AST to extract input_model and handler names from run_cli() call."""
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "run_cli":
            input_model_name = None
            handler_name = None
            for kw in node.keywords:
                if kw.arg == "input_model" and isinstance(kw.value, ast.Name):
                    input_model_name = kw.value.id
                elif kw.arg == "handler" and isinstance(kw.value, ast.Name):
                    handler_name = kw.value.id
            return input_model_name, handler_name
    return None, None


def _load_tools(
    registry_path: Path,
) -> tuple[list[Tool], dict[str, Any]]:
    """Load all tools from the registry and return MCP Tool definitions + handlers.

    Returns:
        (mcp_tools, dispatch_map) where dispatch_map maps
        tool_name -> {"input_model": PydanticModel, "handler": callable}
    """
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    mcp_tools: list[Tool] = []
    dispatch_map: dict[str, Any] = {}

    for entry in registry:
        tool_name: str = entry["tool_name"]
        script_path = (PROJECT_ROOT / entry["script_path"]).resolve()

        # Discover input_model and handler names via AST
        input_model_name, handler_name = _parse_run_cli_call(script_path)
        if not input_model_name or not handler_name:
            logger.warning("Skipping %s: could not discover handler", tool_name)
            continue

        # Import the module
        rel = script_path.relative_to(PROJECT_ROOT)
        module_path = ".".join(rel.with_suffix("").parts)
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            logger.warning("Skipping %s: import failed", tool_name, exc_info=True)
            continue

        input_model = getattr(mod, input_model_name, None)
        handler = getattr(mod, handler_name, None)
        if input_model is None or handler is None:
            logger.warning(
                "Skipping %s: could not resolve %s / %s",
                tool_name, input_model_name, handler_name,
            )
            continue

        # Register MCP tool definition using the registry's input schema
        mcp_tools.append(
            Tool(
                name=tool_name,
                description=entry.get("description", ""),
                inputSchema=entry.get("input_schema", {}),
            )
        )
        dispatch_map[tool_name] = {
            "input_model": input_model,
            "handler": handler,
        }

    return mcp_tools, dispatch_map


# ── Server setup ────────────────────────────────────────────────────

def create_server() -> tuple[Server, list[Tool], dict[str, Any]]:
    """Create and configure the MCP server with all tools."""
    registry_path = PROJECT_ROOT / "tools" / "tool_registry.json"
    mcp_tools, dispatch_map = _load_tools(registry_path)

    app = Server("eng_conductor_tools")

    @app.list_tools()
    async def list_tools() -> list[Tool]:  # noqa: ARG001
        return mcp_tools

    @app.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:  # noqa: ARG001
        spec = dispatch_map.get(name)
        if spec is None:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "tool": name,
                        "status": "error",
                        "error": {"message": f"Unknown tool: {name}"},
                    }),
                )
            ]

        input_model = spec["input_model"]
        handler = spec["handler"]
        args = arguments or {}

        try:
            validated = input_model.model_validate(args)
        except Exception as exc:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "tool": name,
                        "status": "error",
                        "error": {
                            "message": "Input validation failed.",
                            "details": str(exc),
                        },
                    }),
                )
            ]

        try:
            result = handler(validated)
        except Exception as exc:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "tool": name,
                        "status": "error",
                        "error": {
                            "message": f"Tool execution failed: {exc}",
                        },
                    }),
                )
            ]

        return [
            TextContent(
                type="text",
                text=json.dumps({
                    "tool": name,
                    "status": "ok",
                    "result": result,
                }),
            )
        ]

    return app, mcp_tools, dispatch_map


async def main() -> None:
    app, mcp_tools, _ = create_server()
    logger.info("MCP server ready with %d tools", len(mcp_tools))
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    # Ensure project root is on sys.path for tool imports
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    asyncio.run(main())
