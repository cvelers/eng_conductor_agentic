"""Core agent loop — think -> act -> observe -> repeat.

Uses the ``openai`` Python package for native tool calling with streaming.
Bridges sync OpenAI streaming into async iteration via a background thread.

Harness patterns implemented:
  - **System reminders** injected after every tool result (Claude Code pattern).
    Higher behavioral adherence than system-prompt-only instructions.
  - **Observation compression** for older tool results (SWE-Agent pattern).
    Only the last N tool results are kept at full fidelity; older ones are
    compressed to summaries, keeping the context window lean.
  - **Budget-aware guidance** — as tool budget depletes, the agent is
    coached to wrap up rather than hard-stopped without context.
  - **Novelty tracking** — detects when successive searches return the
    same clauses (circular retrieval) and breaks out early.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from typing import Any, AsyncIterator, Callable

from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_ROUNDS = 25
MAX_CONSECUTIVE_SAME_TOOL = 4
LOOP_BREAKER_HARD_LIMIT = 3

# Number of recent tool results to keep at full fidelity.
# Older ones are compressed to summaries (SWE-Agent pattern).
FULL_FIDELITY_TOOL_RESULTS = 4

# After this many total tool calls, inject budget guidance.
TOOL_BUDGET_WARN_THRESHOLD = 10

# ── System Reminders (Claude Code pattern) ─────────────────────────
# Injected after every tool result to keep the agent on track.
# These repeat key behavioral instructions at the exact moment the
# agent is deciding what to do next — the "high attention zone" at
# the end of the context window.

_TODO_NUDGE = (
    " Before your next tool call, call `todo_write` to mark this step "
    "'done' and set the next step to 'in_progress'."
)

_SYSTEM_REMINDERS: dict[str, str] = {
    "eurocode_search": (
        "\n\n<system-reminder>"
        "Review these search results. If they reference tables, clauses, or "
        "equations that are NOT in the results but you need, use `read_clause` "
        "to fetch them directly (e.g., read_clause with clause_id='Table 6.2'). "
        "If results don't cover what you need, search again with different terms."
        + _TODO_NUDGE +
        "</system-reminder>"
    ),
    "read_clause": (
        "\n\n<system-reminder>"
        "You have the full clause text. Check if it cross-references other "
        "clauses, tables, or formulas you still need. Fetch them if so."
        + _TODO_NUDGE +
        "</system-reminder>"
    ),
    "math_calculator": (
        "\n\n<system-reminder>"
        "Verify calculation inputs match the Eurocode requirements. "
        "If any input was assumed rather than looked up, state this clearly."
        + _TODO_NUDGE +
        "</system-reminder>"
    ),
    "search_engineering_tools": (
        "\n\n<system-reminder>"
        "Review the engineering tools found. To use one, call `engineering_calculator` "
        "with the exact tool_name and the required parameters from the schema. "
        "If no suitable tool was found, try different search terms or a different category."
        "</system-reminder>"
    ),
    "engineering_calculator": (
        "\n\n<system-reminder>"
        "Check the calculation result. Verify inputs match the design requirements. "
        "If additional calculations are needed (e.g., you computed capacity but still "
        "need to check utilization), search for more tools or use math_calculator."
        "</system-reminder>"
    ),
    "todo_write": (
        "\n\n<system-reminder>"
        "Plan updated. Now call the next tool for the step marked 'in_progress'. "
        "After it completes, call todo_write again to mark it done."
        "</system-reminder>"
    ),
}

# Default reminder for tools without a specific one
_DEFAULT_REMINDER = (
    "\n\n<system-reminder>"
    "Continue working toward answering the user's question. "
    "If you have enough information, provide your answer now."
    + _TODO_NUDGE +
    "</system-reminder>"
)


# ── Streaming bridge ─────────────────────────────────────────────────


async def _iter_stream_chunks(client: OpenAI, request_kwargs: dict):
    """Bridge sync OpenAI streaming into async iteration."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            stream_obj = client.chat.completions.create(**request_kwargs)
            try:
                iterator = iter(stream_obj)
            except TypeError:
                # Provider returned a non-streaming completion
                if hasattr(stream_obj, "choices"):
                    loop.call_soon_threadsafe(queue.put_nowait, ("completion", stream_obj))
                    return
                raise
            for chunk in iterator:
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", e))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        item_type, payload = await queue.get()
        if item_type == "done":
            break
        if item_type == "error":
            raise payload
        yield item_type, payload


# ── Think-tag handling ───────────────────────────────────────────────


def _consume_think_tags(text: str) -> tuple[str, str]:
    """Split text into (visible, reasoning) based on <think> tags."""
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    in_think = False
    i = 0
    while i < len(text):
        if text[i] != "<":
            if in_think:
                reasoning_parts.append(text[i])
            else:
                visible_parts.append(text[i])
            i += 1
            continue
        close_idx = text.find(">", i + 1)
        if close_idx == -1:
            break
        token = text[i : close_idx + 1]
        if re.match(r"(?is)^<\s*think(?:\s+[^>]*)?>$", token):
            in_think = True
        elif re.match(r"(?is)^<\s*/\s*think\s*>$", token):
            in_think = False
        else:
            if in_think:
                reasoning_parts.append(token)
            else:
                visible_parts.append(token)
        i = close_idx + 1
    # Anything after last tag
    if i < len(text):
        rest = text[i:]
        if in_think:
            reasoning_parts.append(rest)
        else:
            visible_parts.append(rest)
    return "".join(visible_parts), "".join(reasoning_parts)


# ── Tool call accumulation ───────────────────────────────────────────


def _accumulate_tool_call_delta(store: dict[int, dict], delta_tool_calls: Any):
    """Accumulate streaming tool call deltas into complete tool calls."""
    for tc in delta_tool_calls or []:
        idx = getattr(tc, "index", None)
        if idx is None:
            idx = max(store.keys(), default=-1) + 1
        entry = store.setdefault(
            int(idx),
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        tc_id = getattr(tc, "id", None)
        if tc_id:
            entry["id"] = tc_id
        fn = getattr(tc, "function", None)
        if fn:
            name_piece = getattr(fn, "name", None)
            args_piece = getattr(fn, "arguments", None)
            if isinstance(name_piece, str) and name_piece:
                entry["function"]["name"] = name_piece
            if isinstance(args_piece, str) and args_piece:
                entry["function"]["arguments"] += args_piece
        # Capture thought_signature (Gemini thinking models)
        extra = getattr(tc, "extra_content", None)
        if extra and "extra_content" not in entry:
            entry["extra_content"] = extra


def _parse_tool_calls(store: dict[int, dict], tool_round: int) -> list[dict]:
    """Parse accumulated tool call store into clean list, skipping invalid ones."""
    calls = []
    for idx in sorted(store.keys()):
        item = store[idx]
        fn = item.get("function", {})
        name = (fn.get("name") or "").strip()
        args_str = fn.get("arguments") or ""
        if not name or not args_str.strip():
            continue
        try:
            parsed_args = json.loads(args_str)
        except json.JSONDecodeError:
            logger.warning("Skipped tool call '%s' — invalid JSON args", name)
            continue
        tc_parsed: dict[str, Any] = {
            "id": item.get("id") or f"tool_{tool_round}_{idx}",
            "name": name,
            "args": parsed_args,
        }
        if "extra_content" in item:
            tc_parsed["extra_content"] = item["extra_content"]
        calls.append(tc_parsed)
    return calls


# ── Novelty tracking ─────────────────────────────────────────────────


def _extract_clause_ids_from_result(result_str: str) -> set[str]:
    """Extract clause IDs from a tool result for novelty tracking."""
    try:
        data = json.loads(result_str)
        if isinstance(data, dict) and "clauses" in data:
            return {
                c.get("clause_id", "") for c in data["clauses"]
                if isinstance(c, dict) and c.get("clause_id")
            }
    except (json.JSONDecodeError, TypeError):
        pass
    return set()


# ── Main agent loop ──────────────────────────────────────────────────


async def run_agent_loop(
    client: OpenAI,
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_dispatcher: Callable[[str, dict], str],
    *,
    max_rounds: int = MAX_ROUNDS,
    temperature: float = 0.2,
    max_tokens: int = 16_000,
    reasoning_effort: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Core agent loop. Yields event dicts for the stream adapter.

    Events:
      {"type": "thinking", "content": str}
      {"type": "delta", "content": str}
      {"type": "tool_start", "tool": str, "args": dict}
      {"type": "tool_result", "tool": str, "result": Any, "status": str, "summary": str}
      {"type": "plan", "steps": list[dict]}
      {"type": "plan_update", "step_id": str, "status": str}
      {"type": "done", "content": str}
      {"type": "error", "message": str}
    """
    full_response = ""
    tool_round = 0
    total_tool_calls = 0

    # Loop detection state
    consecutive_names: list[str] = []
    loop_breaker_count = 0

    # Novelty tracking — detect circular retrieval
    search_result_sets: list[set[str]] = []

    # Plan tracking — TodoWrite pattern (Claude Code "progress anchor")
    # Stores the latest plan steps so we can emit plan_update events
    plan_steps: list[dict[str, str]] = []
    plan_emitted = False

    all_messages = [{"role": "system", "content": system_prompt}] + messages

    while tool_round < max_rounds:
        # NOTE: No per-query compression here — tool results are kept at
        # full fidelity within a single query. Session-level compaction
        # (in context.py) handles long conversations via the context usage
        # indicator, using semantic compression when triggered.

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": all_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        # Add tools (unless hard-stopped)
        if loop_breaker_count >= LOOP_BREAKER_HARD_LIMIT:
            request_kwargs["tool_choice"] = "none"
        else:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort

        # ── Stream the LLM response ──────────────────────────────────
        assistant_content = ""
        stream_tool_calls: dict[int, dict] = {}
        got_response = False

        try:
            async for chunk_type, payload in _iter_stream_chunks(client, request_kwargs):
                if chunk_type == "completion":
                    # Non-streaming fallback
                    msg = payload.choices[0].message
                    text = getattr(msg, "content", "") or ""
                    if text:
                        got_response = True
                        visible, reasoning = _consume_think_tags(text)
                        if reasoning:
                            yield {"type": "thinking", "content": reasoning}
                        if visible:
                            assistant_content = visible
                            yield {"type": "delta", "content": visible}
                    for idx, tc in enumerate(getattr(msg, "tool_calls", None) or []):
                        got_response = True
                        tc_entry: dict[str, Any] = {
                            "id": tc.id or f"tool_{tool_round}_{idx}",
                            "function": {
                                "name": tc.function.name or "",
                                "arguments": tc.function.arguments or "",
                            },
                        }
                        # Preserve thought_signature (Gemini thinking models)
                        extra = getattr(tc, "extra_content", None)
                        if extra:
                            tc_entry["extra_content"] = extra
                        stream_tool_calls[idx] = tc_entry
                    continue

                # Streaming chunk
                chunk = payload
                choices = getattr(chunk, "choices", []) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if not delta:
                    continue

                # Text content
                text_piece = getattr(delta, "content", None) or ""
                if text_piece:
                    got_response = True
                    visible, reasoning = _consume_think_tags(text_piece)
                    if reasoning:
                        yield {"type": "thinking", "content": reasoning}
                    if visible:
                        assistant_content += visible
                        yield {"type": "delta", "content": visible}

                # Tool call deltas
                delta_tc = getattr(delta, "tool_calls", None)
                if delta_tc:
                    got_response = True
                    _accumulate_tool_call_delta(stream_tool_calls, delta_tc)

        except Exception as e:
            logger.exception("LLM stream error")
            yield {"type": "error", "message": f"LLM error: {e}"}
            break

        if not got_response:
            yield {"type": "error", "message": "No response from LLM"}
            break

        # ── Parse tool calls ─────────────────────────────────────────
        tool_calls = _parse_tool_calls(stream_tool_calls, tool_round)

        # Build assistant message for history
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if assistant_content:
            assistant_msg["content"] = assistant_content
        if tool_calls:
            tc_list = []
            for tc in tool_calls:
                tc_msg: dict[str, Any] = {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                # Preserve thought_signature for Gemini thinking models
                if "extra_content" in tc:
                    tc_msg["extra_content"] = tc["extra_content"]
                tc_list.append(tc_msg)
            assistant_msg["tool_calls"] = tc_list
        all_messages.append(assistant_msg)

        # Accumulate visible text across all rounds (matches what the
        # frontend displays from delta events).
        full_response += assistant_content

        if not tool_calls:
            break

        # ── Loop detection ───────────────────────────────────────────
        # Don't count todo_write toward tool budget — it's a planning no-op
        real_calls = [tc for tc in tool_calls if tc["name"] != "todo_write"]
        for tc in real_calls:
            consecutive_names.append(tc["name"])
        total_tool_calls += len(real_calls)

        force_stop = False
        force_reason = ""

        # Check consecutive same-tool calls
        if len(consecutive_names) >= MAX_CONSECUTIVE_SAME_TOOL + 1:
            recent = consecutive_names[-(MAX_CONSECUTIVE_SAME_TOOL + 1) :]
            if len(set(recent)) == 1:
                force_stop = True
                force_reason = f"Same tool '{recent[0]}' called {MAX_CONSECUTIVE_SAME_TOOL + 1}x consecutively"

        # Check total tool budget
        if total_tool_calls > 15:
            force_stop = True
            force_reason = f"Tool budget exceeded ({total_tool_calls} calls)"

        if force_stop:
            loop_breaker_count += 1
            if loop_breaker_count >= LOOP_BREAKER_HARD_LIMIT:
                all_messages.append({
                    "role": "user",
                    "content": (
                        "STOP CALLING TOOLS. Answer now using the information you already have. "
                        "Summarise your findings."
                    ),
                })
                tool_round += 1
                continue

        # ── Execute tool calls ───────────────────────────────────────
        for tc in tool_calls:
            yield {"type": "tool_start", "tool": tc["name"], "args": tc["args"]}
            t0 = time.time()
            try:
                result_str = tool_dispatcher(tc["name"], tc["args"])
                status = "ok"
            except Exception as e:
                result_str = json.dumps({"error": str(e)})
                status = "error"
            elapsed_ms = int((time.time() - t0) * 1000)

            # ── TodoWrite → plan events (Claude Code pattern) ─────
            if tc["name"] == "todo_write" and status == "ok":
                new_steps = tc["args"].get("todos", [])
                if not plan_emitted:
                    # First call → emit full plan card
                    yield {"type": "plan", "steps": new_steps}
                    plan_steps = new_steps
                    plan_emitted = True
                else:
                    # Subsequent calls → emit per-step updates for changed statuses
                    old_map = {s["id"]: s.get("status", "pending") for s in plan_steps}
                    for step in new_steps:
                        sid = step.get("id", "")
                        new_status = step.get("status", "pending")
                        if sid and old_map.get(sid) != new_status:
                            yield {"type": "plan_update", "step_id": sid, "status": new_status}
                    plan_steps = new_steps

            # Build summary for UI
            summary = _summarize_result(result_str, tc["name"], elapsed_ms)
            yield {
                "type": "tool_result",
                "tool": tc["name"],
                "result": result_str,
                "status": status,
                "summary": summary,
            }

            # ── Novelty tracking for search results ──────────────────
            if tc["name"] == "eurocode_search":
                clause_ids = _extract_clause_ids_from_result(result_str)
                if clause_ids:
                    # Check if this result set is mostly identical to a previous one
                    for prev_set in search_result_sets:
                        overlap = len(clause_ids & prev_set)
                        if prev_set and overlap / max(len(prev_set), 1) > 0.8:
                            # Circular retrieval detected — inject guidance
                            result_str += (
                                '\n\n{"_hint": "WARNING: This search returned very similar '
                                'results to a previous search. Try a significantly different '
                                'query, or use read_clause to fetch specific items directly."}'
                            )
                            break
                    search_result_sets.append(clause_ids)

            # ── System reminder injection (Claude Code pattern) ──────
            # Append a behavioral reminder to the tool result content.
            # This places guidance in the "high attention zone" at the
            # end of the context, achieving higher adherence than
            # system-prompt-only instructions.
            reminder = _SYSTEM_REMINDERS.get(tc["name"], _DEFAULT_REMINDER)

            # Budget-aware guidance: as budget depletes, nudge toward wrapping up
            budget_hint = ""
            remaining = max(0, 15 - total_tool_calls)
            if remaining <= 3 and remaining > 0:
                budget_hint = (
                    f"\n[Budget: {remaining} tool calls remaining. "
                    "Start composing your answer with available information.]"
                )
            elif remaining <= 0:
                budget_hint = (
                    "\n[Budget depleted. Provide your answer now with available information.]"
                )

            tool_content = result_str[:30_000] + reminder + budget_hint

            all_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_content,
            })

        tool_round += 1

    # Build a condensed tool context summary so future turns remember
    # what tools were called and key results from this turn.
    tool_summary = _build_tool_context(all_messages)
    if tool_summary:
        yield {"type": "_tool_context", "summary": tool_summary}

    # Emit token count of what the NEXT request's input will look like:
    # prior history + this turn's text + tool context summary.
    from backend.agent.context import estimate_messages_tokens, estimate_tokens
    # Approximate next-turn input: current messages + assistant response + tool context
    next_turn_msgs = list(messages) + [
        {"role": "assistant", "content": full_response + ("\n\n" + tool_summary if tool_summary else "")},
    ]
    session_tokens = estimate_messages_tokens(next_turn_msgs, system_prompt)
    yield {"type": "_session_tokens", "tokens": session_tokens}
    yield {"type": "done", "content": full_response}


def _build_tool_context(all_messages: list[dict]) -> str:
    """Build a full-fidelity record of tool calls and results for session memory.

    Preserves all tool arguments, all clause content, all calculation outputs
    and intermediate steps. The only data dropped are clauses with low
    relevance scores (< 5.0) from search results.
    """
    blocks: list[str] = []
    skip_tools = {"todo_write"}
    _MIN_CLAUSE_SCORE = 5.0

    # Map tool_call_id → tool name so we can label tool results
    tc_id_to_name: dict[str, str] = {}

    for msg in all_messages:
        # ── Assistant tool calls: capture full args ──────────────────
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                tc_id = tc.get("id", "")
                if tc_id:
                    tc_id_to_name[tc_id] = name
                if name in skip_tools:
                    continue
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                args_str = json.dumps(args, ensure_ascii=False)
                blocks.append(f"[tool_call] {name}({args_str})")

        # ── Tool results: preserve full data ─────────────────────────
        if msg.get("role") == "tool":
            tool_name = tc_id_to_name.get(msg.get("tool_call_id", ""), "")
            if tool_name in skip_tools:
                continue

            content = msg.get("content", "")
            # Strip system-reminder suffixes before parsing
            raw = content.split("<system-reminder>")[0].strip()

            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError, IndexError):
                # Non-JSON results (fetch_url, read_file): store as-is
                if raw:
                    blocks.append(f"[tool_result] {tool_name}:\n{raw}")
                continue

            if not isinstance(data, dict):
                blocks.append(f"[tool_result] {tool_name}: {raw}")
                continue

            # ── Clauses: keep all relevant ones with full text ───────
            if "clauses" in data:
                kept_clauses = []
                for c in data["clauses"]:
                    score = c.get("score", 10)
                    if isinstance(score, (int, float)) and score < _MIN_CLAUSE_SCORE:
                        continue  # drop irrelevant clauses
                    kept_clauses.append(c)
                data["clauses"] = kept_clauses

            # Remove empty/noise fields but keep everything else
            for drop_key in ("_referenced_but_not_retrieved",):
                data.pop(drop_key, None)

            blocks.append(
                f"[tool_result] {tool_name}:\n"
                + json.dumps(data, ensure_ascii=False, default=str)
            )

    if not blocks:
        return ""
    return "<tool-context>\n" + "\n".join(blocks) + "\n</tool-context>"


def _summarize_result(result_str: str, tool_name: str, elapsed_ms: int = 0) -> str:
    """Build a short summary of a tool result for the UI."""
    try:
        data = json.loads(result_str)
        if isinstance(data, dict):
            if "error" in data:
                return f"Error: {data['error'][:100]}"
            if "outputs" in data:
                outputs = data["outputs"]
                if isinstance(outputs, dict):
                    items = [f"{k}={v}" for k, v in list(outputs.items())[:4]]
                    return ", ".join(items)
            if "clauses" in data:
                n = len(data["clauses"])
                return f"Found {n} clause(s)"
            if "matches" in data:
                n = len(data["matches"])
                return f"Found {n} match(es)"
    except (json.JSONDecodeError, TypeError):
        pass
    length = len(result_str)
    return f"OK ({length} chars, {elapsed_ms}ms)"
