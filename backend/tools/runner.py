from __future__ import annotations

import json
import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Any

from backend.registries.tool_registry import ToolRegistryEntry

logger = logging.getLogger(__name__)


class MCPToolRunner:
    def __init__(self, *, project_root: Path, registry: list[ToolRegistryEntry]) -> None:
        self.project_root = project_root
        self.registry = {entry.tool_name: entry for entry in registry}
        self.tools_root = (project_root / "tools" / "mcp").resolve()

    def run(self, tool_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        entry = self.registry.get(tool_name)
        if entry is None:
            raise ValueError(f"Tool '{tool_name}' is not registered.")

        script_path = (self.project_root / entry.script_path).resolve()
        if self.tools_root not in script_path.parents:
            raise ValueError(
                f"Unsafe tool script path for {tool_name}. Expected under {self.tools_root}"
            )

        rel_script = script_path.relative_to(self.project_root)
        module_name = ".".join(rel_script.with_suffix("").parts)
        cmd = [sys.executable, "-m", module_name, "--input-json", json.dumps(inputs)]
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = (
            str(self.project_root)
            if not existing
            else f"{self.project_root}{os.pathsep}{existing}"
        )
        proc = subprocess.run(
            cmd,
            cwd=self.project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0:
            if stdout:
                try:
                    payload = json.loads(stdout)
                    error_msg = payload.get("error", {}).get("message") or payload.get("error")
                except Exception:  # noqa: BLE001
                    error_msg = stdout
            else:
                error_msg = stderr or "unknown tool failure"

            logger.warning(
                "tool_call_failed",
                extra={"tool": tool_name, "returncode": proc.returncode, "error": error_msg},
            )
            raise RuntimeError(f"{tool_name} failed: {error_msg}")

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON output from {tool_name}: {stdout}") from exc

        logger.info(
            "tool_call_ok",
            extra={"tool": tool_name, "status": payload.get("status", "ok")},
        )

        return payload
