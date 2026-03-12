"""Tool definitions (OpenAI function-calling format) and dispatcher.

Each tool is a dict in OpenAI's ``tools`` array format and a plain
Python handler function.  No MCP subprocess — tools are called directly.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Regex for extracting cross-references from clause text
_CROSS_REF_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+)?(?:\([^)]+\))?)\b")
_TABLE_REF_RE = re.compile(r"\b[Tt]able\s+(\d+\.\d+(?:\.\d+)?)\b")

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
                        "description": (
                            "The Eurocode standard this clause belongs to, e.g. 'EN 1993-1-1', "
                            "'EN 1993-1-8'. ALWAYS specify this — the same clause ID (e.g. "
                            "'Table 3.1') exists in multiple standards and you must name the "
                            "correct one."
                        ),
                    },
                },
                "required": ["clause_id", "standard"],
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
    # ── Planning ──────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "todo_write",
            "description": (
                "Create or update your task plan. Call this FIRST before using any other tool. "
                "List the steps you intend to follow to answer the user's question. "
                "You can update the plan later to mark steps as completed or add new ones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Short unique step ID (e.g. 'search', 'calc', 'fetch_table').",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "Brief description of this step.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "done"],
                                    "description": "Step status. Use 'pending' for initial plan, update later.",
                                },
                            },
                            "required": ["id", "text", "status"],
                        },
                        "description": "Ordered list of plan steps.",
                    },
                },
                "required": ["todos"],
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
    top_k = min(args.get("top_k", 20), 30)
    results, trace = retriever.retrieve(query, top_k=top_k)

    clauses_out = []
    # Collect cross-referenced IDs across all results for hint generation
    all_referenced_ids: set[str] = set()
    returned_ids: set[str] = set()

    for r in results:
        returned_ids.add(r.clause.clause_id.lower())
        clause_data: dict[str, Any] = {
            "clause_id": r.clause.clause_id,
            "title": r.clause.clause_title,
            "standard": r.clause.standard,
            "text": r.clause.text,
            "score": round(r.score, 2),
            "pointer": r.clause.pointer,
            "selected": r.selected,
        }
        # Extract cross-references from clause text for progressive disclosure
        refs = _CROSS_REF_RE.findall(r.clause.text)
        table_refs = _TABLE_REF_RE.findall(r.clause.text)
        if refs or table_refs:
            all_refs = set(refs) | {f"Table {t}" for t in table_refs}
            clause_data["cross_references"] = sorted(all_refs)[:8]
            all_referenced_ids.update(r.lower() for r in refs)
            all_referenced_ids.update(f"table {t}".lower() for t in table_refs)
        clauses_out.append(clause_data)

    result: dict[str, Any] = {
        "clauses": clauses_out,
        "total_found": len(results),
    }

    # Surface unretrieved cross-references as a hint for the agent
    missing_refs = all_referenced_ids - returned_ids
    if missing_refs:
        # Filter to only plausible Eurocode-style references
        plausible = sorted(r for r in missing_refs if len(r) > 2)[:6]
        if plausible:
            result["_referenced_but_not_retrieved"] = plausible

    return json.dumps(result)


def _handle_read_clause(args: dict, clause_index: dict) -> str:
    clause_id = args.get("clause_id", "").strip()
    standard = args.get("standard", "").strip().lower()
    logger.info("read_clause called: clause_id=%r, standard=%r", clause_id, standard)

    # Normalize: "Table 6.2" → try both "table 6.2" and "6.2"
    lookup_keys = [clause_id.lower()]
    if clause_id.lower().startswith("table"):
        bare = clause_id.lower().replace("table", "").strip()
        lookup_keys.append(f"table {bare}")
        lookup_keys.append(bare)

    candidates = []
    for key in lookup_keys:
        candidates.extend(clause_index.get(key, []))

    if standard:
        candidates = [c for c in candidates if c.standard.lower() == standard]
    if not candidates:
        # Try partial match
        for key, vals in clause_index.items():
            if clause_id.lower() in key or key in clause_id.lower():
                candidates.extend(vals)
        if standard:
            candidates = [c for c in candidates if c.standard.lower() == standard]
    if not candidates:
        # Provide a helpful error with suggestions
        similar = []
        cid_lower = clause_id.lower()
        for key in clause_index:
            if cid_lower[:3] in key:  # Match first part of the ID
                similar.append(key)
                if len(similar) >= 5:
                    break
        error_data: dict[str, Any] = {
            "error": f"Clause '{clause_id}' not found in database.",
        }
        if similar:
            error_data["similar_ids"] = sorted(set(similar))[:5]
            error_data["_hint"] = "Try one of the similar IDs, or use eurocode_search to find it."
        return json.dumps(error_data)

    # If no standard was specified and multiple standards matched, REJECT — force retry
    matched_stds = sorted({c.standard for c in candidates})
    logger.info("read_clause: standard=%r, matched_standards=%r, count=%d", standard, matched_stds, len(candidates))
    if not standard:
        matched_standards = sorted({c.standard for c in candidates})
        if len(matched_standards) > 1:
            return json.dumps({
                "error": (
                    f"AMBIGUOUS: Clause '{clause_id}' exists in {len(matched_standards)} "
                    f"different standards: {matched_standards}. "
                    f"You MUST call read_clause again with the 'standard' parameter set "
                    f"to the correct standard (e.g. standard='EN 1993-1-1')."
                ),
                "matching_standards": matched_standards,
            })

    # Deduplicate
    seen: set[str] = set()
    results = []
    for c in candidates:
        key = f"{c.standard}:{c.clause_id}"
        if key in seen:
            continue
        seen.add(key)
        clause_data: dict[str, Any] = {
            "clause_id": c.clause_id,
            "title": c.clause_title,
            "standard": c.standard,
            "text": c.text,
            "pointer": c.pointer,
        }
        # Surface cross-references for progressive disclosure
        refs = _CROSS_REF_RE.findall(c.text)
        table_refs = _TABLE_REF_RE.findall(c.text)
        if refs or table_refs:
            all_refs = set(refs) | {f"Table {t}" for t in table_refs}
            clause_data["cross_references"] = sorted(all_refs)[:10]
        results.append(clause_data)
        if len(results) >= 5:
            break

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


def _handle_todo_write(args: dict) -> str:
    """No-op planning tool (Claude Code TodoWrite pattern).

    The tool simply echoes the plan back. Its value is forcing the LLM to
    articulate its approach before acting. The agent loop intercepts the call
    to emit plan/plan_update events for the frontend.
    """
    todos = args.get("todos", [])
    summary_lines = []
    for step in todos:
        icon = {"pending": "○", "in_progress": "▶", "done": "✓"}.get(step.get("status", "pending"), "○")
        summary_lines.append(f"  {icon} {step.get('id', '?')}: {step.get('text', '')}")
    return json.dumps({
        "status": "ok",
        "plan": todos,
        "summary": "\n".join(summary_lines),
    })


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
    # Multiple keys per clause for flexible lookup (Table 6.2, table 6.2, 6.2, etc.)
    clause_index: dict[str, list] = {}
    for c in clauses:
        cid = c.clause_id.lower().strip()
        clause_index.setdefault(cid, []).append(c)

        # Index tables under multiple keys for flexible lookup
        if cid.startswith("table"):
            bare = cid.replace("table", "").strip()
            clause_index.setdefault(f"table {bare}", []).append(c)
            clause_index.setdefault(bare, []).append(c)
        # Also index by title keywords for tables mentioned in title
        title_lower = c.clause_title.lower()
        if "table" in title_lower:
            import re as _re
            for m in _re.finditer(r"table\s+(\d+\.\d+(?:\.\d+)?)", title_lower):
                key = f"table {m.group(1)}"
                clause_index.setdefault(key, []).append(c)

    _handlers: dict[str, Callable] = {
        "eurocode_search": lambda args: _handle_eurocode_search(args, retriever),
        "read_clause": lambda args: _handle_read_clause(args, clause_index),
        "math_calculator": _handle_math_calculator,
        "section_lookup": _handle_section_lookup,
        "material_lookup": _handle_material_lookup,
        "todo_write": _handle_todo_write,
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
