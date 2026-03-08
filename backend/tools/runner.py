"""MCP tool runner — calls tools via MCP protocol over stdio transport.

Starts a single MCP server subprocess at initialisation and keeps the
connection alive.  The synchronous ``run()`` method bridges into the
async MCP client session running on a background thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from backend.registries.tool_registry import ToolRegistryEntry

logger = logging.getLogger(__name__)

_CALL_TIMEOUT = 20  # seconds


class MCPToolRunner:
    """Run engineering tools via the MCP protocol (stdio transport)."""

    def __init__(
        self,
        *,
        project_root: Path,
        registry: list[ToolRegistryEntry],
    ) -> None:
        self.project_root = project_root
        self.registry = {entry.tool_name: entry for entry in registry}

        # Background event-loop for the async MCP client
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._session: ClientSession | None = None
        self._ready = threading.Event()
        self._shutdown_event: asyncio.Event | None = None

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="mcp-tool-runner",
        )
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise RuntimeError("MCP server failed to start within 30 s")

    # ── Background event-loop ───────────────────────────────────────

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())

    async def _connect(self) -> None:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = (
            str(self.project_root)
            if not existing
            else f"{self.project_root}{os.pathsep}{existing}"
        )

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "tools.mcp_server"],
            cwd=str(self.project_root),
            env=env,
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                self._session = session
                self._shutdown_event = asyncio.Event()
                self._ready.set()
                logger.info("MCP tool runner connected")
                # Keep alive until shutdown is requested
                await self._shutdown_event.wait()

    # ── Public API ──────────────────────────────────────────────────

    def run(self, tool_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Call a tool via MCP protocol.  Synchronous interface."""
        if tool_name not in self.registry:
            raise ValueError(f"Tool '{tool_name}' is not registered.")
        if self._session is None:
            raise RuntimeError("MCP server not connected.")

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(tool_name, inputs),
            self._loop,
        )
        return future.result(timeout=_CALL_TIMEOUT)

    async def _call_tool(
        self,
        tool_name: str,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        assert self._session is not None  # noqa: S101
        result = await self._session.call_tool(tool_name, arguments=inputs)

        # MCP returns content blocks; parse the first text block as JSON
        for block in result.content:
            if block.type == "text":
                payload: dict[str, Any] = json.loads(block.text)
                if payload.get("status") == "error":
                    error_info = payload.get("error", {})
                    msg = (
                        error_info.get("message")
                        if isinstance(error_info, dict)
                        else str(error_info)
                    )
                    logger.warning(
                        "tool_call_failed",
                        extra={"tool": tool_name, "error": msg},
                    )
                    raise RuntimeError(f"{tool_name} failed: {msg}")

                logger.info(
                    "tool_call_ok",
                    extra={"tool": tool_name, "status": payload.get("status", "ok")},
                )
                return payload

        raise RuntimeError(f"No text content in MCP response for {tool_name}")

    # ── Lifecycle ───────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Signal the background loop to exit and wait for cleanup."""
        if self._shutdown_event is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._shutdown_event.set)
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        if not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
