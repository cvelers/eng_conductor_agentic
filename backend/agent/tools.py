"""Tool definitions (OpenAI function-calling format) and dispatcher.

Each tool is a dict in OpenAI's ``tools`` array format and a plain
Python handler function.  No MCP subprocess — tools are called directly.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Tool definitions (OpenAI format) ─────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    # ── Engineering ───────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "eurocode_search",
            "description": (
                "Search the Eurocode standards database by topic. Returns relevant "
                "clauses with full text, formulas, and tables. Use this to find design "
                "rules, formulas, material properties, or any Eurocode requirement."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query describing what you need from the Eurocodes. "
                            "Be specific: 'lateral torsional buckling resistance IPE beam EN 1993-1-1' "
                            "not just 'buckling'."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of clauses to return (default 8, max 20).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_clause",
            "description": (
                "Read a specific Eurocode clause by its ID. Use when you know the exact "
                "clause number (e.g. '6.3.2.3' or 'Table 6.2')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clause_id": {
                        "type": "string",
                        "description": "Clause ID, e.g. '6.3.2.3', 'Table 3.1', '3.2.6'.",
                    },
                    "standard": {
                        "type": "string",
                        "description": "Optional standard filter, e.g. 'EN 1993-1-1'.",
                    },
                },
                "required": ["clause_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "math_calculator",
            "description": (
                "Evaluate engineering math expressions safely. Supports sequential "
                "equations where later equations reference earlier results. "
                "Supports: +, -, *, /, **, sqrt(), min(), max(), abs(), round(), "
                "trig functions, pi, e, comparisons (<, >, ==), "
                "conditionals (value_a if cond else value_b), boolean (and, or)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "equations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Variable name for the result."},
                                "expression": {
                                    "type": "string",
                                    "description": (
                                        "Math expression. Can reference earlier names. "
                                        "Use Python ternary for conditionals: a if cond else b"
                                    ),
                                },
                                "unit": {"type": "string", "description": "Unit, e.g. 'mm²', 'kN'."},
                                "description": {"type": "string", "description": "What this computes."},
                            },
                            "required": ["name", "expression"],
                        },
                        "description": "Ordered list of equations. Each result is available to subsequent equations.",
                    },
                    "variables": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Input variables, e.g. {\"A\": 5380, \"fy\": 355}.",
                    },
                },
                "required": ["equations", "variables"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "section_lookup",
            "description": (
                "Look up geometric properties for a standard rolled steel profile. "
                "Supports IPE, HEA, HEB, HEM families. Returns area, moments of inertia, "
                "section moduli, and dimensions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_name": {
                        "type": "string",
                        "description": "Section designation, e.g. 'IPE300', 'HEA200', 'HEB300', 'HEM200'.",
                    },
                },
                "required": ["section_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "material_lookup",
            "description": (
                "Look up steel grade material properties per EN 10025 / EC3 Table 3.1. "
                "Returns fy, fu, E, epsilon for S235, S275, S355, S420, S460."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steel_grade": {
                        "type": "string",
                        "description": "Steel grade, e.g. 'S355', 'S235'.",
                    },
                    "thickness_mm": {
                        "type": "number",
                        "description": "Element thickness in mm (affects fy if > 40mm). Optional.",
                    },
                },
                "required": ["steel_grade"],
            },
        },
    },
    # ── General ───────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for engineering references, standards, or general information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch and extract text content from a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."},
                    "offset": {"type": "integer", "description": "Start line (0-indexed). Optional."},
                    "limit": {"type": "integer", "description": "Max lines to read. Optional."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or create a file on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write."},
                    "content": {"type": "string", "description": "File content."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Defaults to current directory."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files by name pattern or grep content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Filename glob pattern (e.g. '*.py') or text to search inside files.",
                    },
                    "path": {"type": "string", "description": "Directory to search in. Optional."},
                    "content_search": {
                        "type": "boolean",
                        "description": "If true, search file contents instead of filenames.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command. Use for running scripts, tests, or other CLI tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)."},
                },
                "required": ["command"],
            },
        },
    },
]

# Tools kept in code but hidden from the LLM (not sent in the tools array).
# Handlers remain in the dispatcher so they still work if called.
_HIDDEN_TOOLS = {"section_lookup", "material_lookup", "write_file", "run_command"}

TOOLS = [t for t in TOOLS if t["function"]["name"] not in _HIDDEN_TOOLS]


# ── Tool handlers ────────────────────────────────────────────────────


def _handle_eurocode_search(args: dict, retriever: Any) -> str:
    query = args.get("query", "")
    top_k = min(args.get("top_k", 8), 20)
    results, _trace = retriever.retrieve(query, top_k=top_k)
    clauses_out = []
    for r in results:
        clauses_out.append({
            "clause_id": r.clause.clause_id,
            "title": r.clause.clause_title,
            "standard": r.clause.standard,
            "text": r.clause.text[:3000],
            "score": round(r.score, 2),
            "pointer": r.clause.pointer,
        })
    return json.dumps({"clauses": clauses_out, "total_found": len(results)})


def _handle_read_clause(args: dict, clause_index: dict) -> str:
    clause_id = args.get("clause_id", "").strip()
    standard = args.get("standard", "").strip().lower()
    candidates = clause_index.get(clause_id.lower(), [])
    if standard:
        candidates = [c for c in candidates if standard in c.standard.lower()]
    if not candidates:
        # Try partial match
        for key, vals in clause_index.items():
            if clause_id.lower() in key:
                candidates.extend(vals)
        if standard:
            candidates = [c for c in candidates if standard in c.standard.lower()]
    if not candidates:
        return json.dumps({"error": f"Clause '{clause_id}' not found in database."})
    results = []
    for c in candidates[:5]:
        results.append({
            "clause_id": c.clause_id,
            "title": c.clause_title,
            "standard": c.standard,
            "text": c.text,
            "pointer": c.pointer,
        })
    return json.dumps({"clauses": results})


def _handle_math_calculator(args: dict) -> str:
    from tools.mcp.math_calculator import MathCalculatorInput, calculate
    inp = MathCalculatorInput(**args)
    result = calculate(inp)
    return json.dumps(result, default=str)


def _handle_section_lookup(args: dict) -> str:
    from tools.mcp.section_properties import SectionPropertiesInput, lookup
    inp = SectionPropertiesInput(section_name=args["section_name"])
    result = lookup(inp)
    return json.dumps(result, default=str)


def _handle_material_lookup(args: dict) -> str:
    from tools.mcp.steel_grade_properties import SteelGradeInput, lookup
    inp = SteelGradeInput(
        steel_grade=args["steel_grade"],
        thickness_mm=args.get("thickness_mm"),
    )
    result = lookup(inp)
    return json.dumps(result, default=str)


def _handle_web_search(args: dict) -> str:
    import httpx
    query = args.get("query", "")
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            # Simple extraction of result snippets
            import re
            results = []
            for m in re.finditer(r'class="result__snippet"[^>]*>(.*?)</a', resp.text, re.DOTALL):
                snippet = re.sub(r"<[^>]+>", "", m.group(1)).strip()
                if snippet:
                    results.append(snippet)
            if not results:
                return json.dumps({"results": [], "note": "No results found."})
            return json.dumps({"results": results[:5], "query": query})
    except Exception as e:
        return json.dumps({"error": f"Web search failed: {e}"})


def _handle_fetch_url(args: dict) -> str:
    import httpx
    url = args.get("url", "")
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            import re
            text = re.sub(r"<script[^>]*>.*?</script>", "", resp.text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:8000]
    except Exception as e:
        return json.dumps({"error": f"Fetch failed: {e}"})


def _handle_read_file(args: dict) -> str:
    path = Path(args.get("path", "")).expanduser()
    if not path.exists():
        return json.dumps({"error": f"File not found: {path}"})
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        offset = args.get("offset", 0)
        limit = args.get("limit", 500)
        selected = lines[offset : offset + limit]
        return "\n".join(f"{i + offset + 1}: {line}" for i, line in enumerate(selected))
    except Exception as e:
        return json.dumps({"error": f"Read failed: {e}"})


def _handle_write_file(args: dict) -> str:
    path = Path(args.get("path", "")).expanduser()
    content = args.get("content", "")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return json.dumps({"status": "ok", "path": str(path), "bytes": len(content.encode())})
    except Exception as e:
        return json.dumps({"error": f"Write failed: {e}"})


def _handle_list_directory(args: dict) -> str:
    path = Path(args.get("path", ".")).expanduser()
    if not path.is_dir():
        return json.dumps({"error": f"Not a directory: {path}"})
    entries = []
    try:
        for entry in sorted(path.iterdir()):
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
            })
    except PermissionError:
        return json.dumps({"error": "Permission denied"})
    return json.dumps({"path": str(path), "entries": entries[:100]})


def _handle_search_files(args: dict) -> str:
    import glob as glob_mod
    pattern = args.get("pattern", "")
    search_path = Path(args.get("path", ".")).expanduser()
    content_search = args.get("content_search", False)

    if content_search:
        matches = []
        try:
            for fpath in search_path.rglob("*"):
                if fpath.is_file() and fpath.stat().st_size < 500_000:
                    try:
                        text = fpath.read_text(encoding="utf-8", errors="replace")
                        for i, line in enumerate(text.splitlines()):
                            if pattern.lower() in line.lower():
                                matches.append({"file": str(fpath), "line": i + 1, "text": line.strip()[:200]})
                                if len(matches) >= 30:
                                    return json.dumps({"matches": matches, "truncated": True})
                    except Exception:
                        continue
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"matches": matches})
    else:
        results = sorted(str(p) for p in search_path.rglob(pattern))[:50]
        return json.dumps({"files": results})


def _handle_run_command(args: dict) -> str:
    command = args.get("command", "")
    timeout = min(args.get("timeout", 30), 60)
    # Basic safety
    dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){", "fork bomb"]
    for d in dangerous:
        if d in command.lower():
            return json.dumps({"error": f"Blocked dangerous command pattern: {d}"})
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=os.getcwd(),
        )
        return json.dumps({
            "exit_code": result.returncode,
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000],
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {timeout}s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Dispatcher factory ───────────────────────────────────────────────


def build_tool_dispatcher(
    retriever: Any,
    clauses: list,
) -> Callable[[str, dict], str]:
    """Build a closure that dispatches tool calls by name.

    Returns a ``(tool_name, args) -> result_str`` callable.
    """
    # Pre-build clause lookup index for read_clause
    clause_index: dict[str, list] = {}
    for c in clauses:
        clause_index.setdefault(c.clause_id.lower(), []).append(c)
        # Also index tables by their ID
        if c.clause_id.lower().startswith("table"):
            clause_index.setdefault(c.clause_id.lower(), []).append(c)

    _handlers: dict[str, Callable] = {
        "eurocode_search": lambda args: _handle_eurocode_search(args, retriever),
        "read_clause": lambda args: _handle_read_clause(args, clause_index),
        "math_calculator": _handle_math_calculator,
        "section_lookup": _handle_section_lookup,
        "material_lookup": _handle_material_lookup,
        "web_search": _handle_web_search,
        "fetch_url": _handle_fetch_url,
        "read_file": _handle_read_file,
        "write_file": _handle_write_file,
        "list_directory": _handle_list_directory,
        "search_files": _handle_search_files,
        "run_command": _handle_run_command,
    }

    def dispatch(tool_name: str, args: dict) -> str:
        handler = _handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return handler(args)
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return json.dumps({"error": f"{tool_name} failed: {e}"})

    return dispatch
