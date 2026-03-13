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
MAX_VALIDATION_RETRIES = 2

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
        "If no suitable tool was found, try different search terms or a different category. "
        "Remember: engineering calculators compute numeric results but do NOT retrieve "
        "Eurocode clause text. If your answer will cite specific clauses, also search "
        "for them with `eurocode_search` or `read_clause`."
        "</system-reminder>"
    ),
    "engineering_calculator": (
        "\n\n<system-reminder>"
        "Check the calculation result. Verify inputs match the design requirements. "
        "If additional calculations are needed (e.g., you computed capacity but still "
        "need to check utilization), search for more tools or use math_calculator. "
        "IMPORTANT: This calculator only provides numeric results — if you plan to "
        "cite Eurocode clauses in your answer (e.g. 'per Cl. 6.3.2'), you MUST "
        "fetch the actual clause text with `eurocode_search` or `read_clause` first. "
        "Calculator output alone does NOT ground a clause citation. Also preserve "
        "symbol meaning: never feed a resistance/capacity result such as M_Rd, "
        "Mc,Rd, Mb,Rd, N_Rd, or V_Rd back into a demand input such as M_Ed, N_Ed, "
        "or V_Ed."
        "</system-reminder>"
    ),
    "todo_write": (
        "\n\n<system-reminder>"
        "Plan updated. Now call the next tool for the step marked 'in_progress'. "
        "After it completes, call todo_write again to mark it done."
        "</system-reminder>"
    ),
    "ask_user": (
        "\n\n<system-reminder>"
        "The question has been shown to the user. STOP immediately — "
        "do NOT call any more tools or write an answer. Wait for the user's response."
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


# ── Grounding validation (independent LLM check) ────────────────────

_GROUNDING_VALIDATOR_PROMPT = """\
You are a grounding validator for an engineering assistant. Your ONLY job is to \
check whether the response below is fully supported by the available evidence. \
You are an independent auditor — the response author has no control over your verdict.

{conversation_history_section}\
## Tool Results From This Session
{tool_results}

## Response to Validate
{response}

## Instructions
Check EVERY factual claim in the response against the evidence above:
1. Every Eurocode clause cited (e.g. "Cl. 6.2.5") — is it in the tool results \
from explicit clause evidence returned by `eurocode_search` / `read_clause`, \
or in a previous validated response/session memory? Calculator metadata alone \
is NOT enough to ground a cited clause.
2. Every numeric value stated (fy, Wpl, dimensions, safety factors) — does it \
match a tool output or a value from a previous validated response? Check the \
actual numbers.
3. Every calculation result — was it produced by a calculator tool or stated in \
a previous validated response?
4. Any formula or rule — is it from a retrieved clause or previous response, \
or recited from memory?
5. If a calculator used assumed inputs, make sure those assumptions came from \
the user, previous validated context, or were explicitly stated as assumptions \
in the response. Do NOT accept silent assumptions just because the tool ran.
6. Preserve engineering semantics of symbols and variable roles. A prior value \
is only grounded for the SAME physical quantity. Flag demand/resistance swaps \
such as using `M_Rd`, `M_c,Rd`, or `M_b,Rd` as `M_Ed`, or reusing any resistance \
value as a load effect just because the number matches.
7. Inspect calculator CALL arguments as well as final prose. A calculator run does \
NOT make its inputs valid. If a tool input lacks evidence, contradicts prior context, \
or changes the meaning of an established symbol, flag it.

Values and clauses that appear in previously validated responses are considered \
grounded — they were already verified in an earlier turn. However, grounding does \
NOT transfer across different symbol meanings or variable roles. Do NOT treat a \
previously validated resistance as evidence for a demand value, or vice versa.

Respond with ONLY a JSON object (no markdown, no explanation):
- If fully grounded: {{"valid": true}}
- If issues found: {{"valid": false, "issues": ["issue1", "issue2", ...]}}\
"""

def _build_tool_results_for_validator(all_messages: list[dict]) -> str:
    """Extract tool call/result pairs from message history for the validator."""
    blocks: list[str] = []
    skip = {"todo_write", "ask_user"}
    tc_id_to_name: dict[str, str] = {}

    for msg in all_messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                tc_id = tc.get("id", "")
                if tc_id:
                    tc_id_to_name[tc_id] = name
                if name in skip:
                    continue
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                blocks.append(f"[CALL] {name}({json.dumps(args, ensure_ascii=False)})")

        if msg.get("role") == "tool":
            tool_name = tc_id_to_name.get(msg.get("tool_call_id", ""), "")
            if tool_name in skip:
                continue
            content = msg.get("content", "")
            raw = content.split("<system-reminder>")[0].strip()

            # Strip clause_references from engineering tool results.
            # These are static registry metadata (e.g. "EN 1993-1-1 §6.3.2")
            # about which clauses the tool *implements*, NOT evidence that the
            # clause was actually retrieved.  Leaving them in causes the
            # validator to wrongly treat cited clauses as grounded.
            if tool_name in ("engineering_calculator", "search_engineering_tools"):
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        data.pop("clause_references", None)
                        # Also strip from nested results array
                        for r in data.get("results", []):
                            if isinstance(r, dict):
                                r.pop("clause_references", None)
                        raw = json.dumps(data, ensure_ascii=False, default=str)
                except (json.JSONDecodeError, TypeError):
                    pass

            if len(raw) > 6000:
                raw = raw[:6000] + "\n... (truncated)"
            blocks.append(f"[RESULT] {tool_name}:\n{raw}")

    return "\n\n".join(blocks) if blocks else "(no tool calls in this session)"


def _build_conversation_history_for_validator(all_messages: list[dict]) -> str:
    """Extract previous validated responses from conversation history.

    Returns user questions and assistant answers from prior turns so the
    validator knows which values were already established and validated.
    """
    blocks: list[str] = []
    for msg in all_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        # Skip system messages and injected validation-failure messages
        if role == "system":
            continue
        if role == "user" and content.startswith("GROUNDING VALIDATION FAILED"):
            continue
        # Previous user questions
        if role == "user":
            blocks.append(f"[USER] {content[:500]}")
        # Previous assistant text responses (no tool_calls = final answer)
        elif role == "assistant" and not msg.get("tool_calls"):
            blocks.append(f"[ASSISTANT — previously validated] {content[:3000]}")
    return "\n\n".join(blocks) if blocks else ""


async def _validate_grounding(
    client: OpenAI,
    model: str,
    response_text: str,
    all_messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 2000,
    reasoning_effort: str | None = None,
) -> dict:
    """Call an independent LLM to validate grounding of the agent's response."""
    tool_results = _build_tool_results_for_validator(all_messages)
    conversation_history = _build_conversation_history_for_validator(all_messages)

    # Only include conversation history section if there are previous turns
    if conversation_history:
        history_section = (
            "## Conversation History (previously validated)\n"
            f"{conversation_history}\n\n"
        )
    else:
        history_section = ""

    prompt = _GROUNDING_VALIDATOR_PROMPT.format(
        conversation_history_section=history_section,
        tool_results=tool_results,
        response=response_text,
    )

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort

    loop = asyncio.get_running_loop()
    try:
        completion = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(**request_kwargs),
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Grounding validator returned non-JSON: %s", raw[:200] if raw else "")
        return {"valid": True}
    except Exception:
        logger.exception("Grounding validation LLM call failed")
        return {"valid": True}


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
    grounding_validation: bool = True,
    validator_client: OpenAI | None = None,
    validator_model: str | None = None,
    validator_temperature: float = 0.0,
    validator_max_tokens: int = 2000,
    validator_reasoning_effort: str | None = None,
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

    # Grounding validation retry tracking
    validation_retries = 0

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

        # Buffer text events — only flushed after we know the round's
        # outcome (tool calls → flush immediately; final answer → hold
        # for grounding validation, flush only if validation passes).
        pending_events: list[dict] = []

        # Should we buffer deltas? Yes when validation could run.
        may_validate = (
            grounding_validation
            and validator_client is not None
            and validator_model
        )

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
                            if may_validate:
                                pending_events.append({"type": "delta", "content": visible})
                            else:
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
                        if may_validate:
                            pending_events.append({"type": "delta", "content": visible})
                        else:
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
            # ── Grounding validation (independent LLM) ───────────────
            # Always validate the final response if there was tool usage.
            has_tool_results = any(m.get("role") == "tool" for m in all_messages)
            if (
                grounding_validation
                and has_tool_results
                and full_response.strip()
                and validator_client is not None
                and validator_model
            ):
                yield {"type": "tool_start", "tool": "grounding_validator", "args": {}}
                verdict = await _validate_grounding(
                    client=validator_client,
                    model=validator_model,
                    response_text=full_response,
                    all_messages=all_messages,
                    temperature=validator_temperature,
                    max_tokens=validator_max_tokens,
                    reasoning_effort=validator_reasoning_effort or None,
                )
                yield {
                    "type": "tool_result",
                    "tool": "grounding_validator",
                    "result": json.dumps(verdict),
                    "status": "ok" if verdict.get("valid") else "warning",
                    "summary": "Grounded" if verdict.get("valid") else f"{len(verdict.get('issues', []))} issue(s)",
                }

                if not verdict.get("valid") and validation_retries < MAX_VALIDATION_RETRIES:
                    issues = verdict.get("issues", [])
                    issues_text = "\n".join(f"- {i}" for i in issues)
                    validation_retries += 1

                    # Discard buffered events — user never sees the bad response
                    pending_events.clear()

                    # Inject issues as a message and let the agent try again
                    all_messages.append({
                        "role": "user",
                        "content": (
                            "GROUNDING VALIDATION FAILED. An independent validator found these issues "
                            "with your response:\n\n"
                            f"{issues_text}\n\n"
                            "Fix these issues: either fetch the missing data with the appropriate tool, "
                            "or remove the ungrounded claims. Then provide a corrected response."
                        ),
                    })
                    # Reset full_response — the agent will produce a new one
                    full_response = ""
                    tool_round += 1
                    continue

            # Validation passed (or not needed) — flush buffered deltas
            for evt in pending_events:
                yield evt
            pending_events.clear()
            break

        # ── Loop detection ───────────────────────────────────────────
        # Don't count meta-tools toward tool budget
        _META_TOOLS = {"todo_write", "ask_user"}
        real_calls = [tc for tc in tool_calls if tc["name"] not in _META_TOOLS]
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

        # ── Flush buffered events for intermediate rounds ─────────
        # When validation is enabled, hold ALL text until the final
        # validation passes — the user must not see any response text
        # before it is validated. When validation is off, flush
        # immediately so the user sees intermediate reasoning.
        if not may_validate:
            for evt in pending_events:
                yield evt
            pending_events.clear()

        # ── Execute tool calls ───────────────────────────────────────
        asked_user = False
        for tc_idx, tc in enumerate(tool_calls):
            yield {"type": "tool_start", "tool": tc["name"], "args": tc["args"]}
            t0 = time.time()
            try:
                result_str = tool_dispatcher(tc["name"], tc["args"])
                status = "ok"
            except Exception as e:
                result_str = json.dumps({"error": str(e)})
                status = "error"
            elapsed_ms = int((time.time() - t0) * 1000)


            # ── AskUser → structured question event ──────────────
            if tc["name"] == "ask_user" and status == "ok":
                asked_user = True
                try:
                    ask_data = json.loads(result_str)
                except (json.JSONDecodeError, TypeError):
                    ask_data = {}
                yield {
                    "type": "ask_user",
                    "question": ask_data.get("question", ""),
                    "options": ask_data.get("options", []),
                    "context": ask_data.get("context", ""),
                }

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

            if asked_user:
                logger.info(
                    "ask_user_hard_stop",
                    extra={"skipped_following_calls": len(tool_calls) - tc_idx - 1},
                )
                break

        # If ask_user was called, stop the loop — wait for user input
        if asked_user:
            # Flush any buffered text (the question preamble)
            for evt in pending_events:
                yield evt
            pending_events.clear()
            break

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
    """Build a curated record of tool calls and results for session memory.

    Design decisions:
      - **todo_write**: only the LAST call is kept (represents final plan
        state).  Earlier calls are superseded and would waste tokens.
      - **ask_user**: always included so the continuation agent knows what
        question was asked and can resume seamlessly.
      - **Search results (eurocode_search)**: only clauses with
        ``selected=True`` (LLM-marked as truly needed) are kept.  Falls back
        to score ≥ 5.0 when no clause has ``selected`` set (non-agentic mode).
      - **Engineering calculator**: ``clause_references`` metadata is stripped.
        These are static registry metadata about which clauses the tool
        *implements*, NOT evidence that a clause was actually retrieved/read.
        Keeping them would cause the grounding validator to wrongly treat them
        as retrieved evidence.
    """
    blocks: list[str] = []

    # Map tool_call_id → tool name so we can label tool results
    tc_id_to_name: dict[str, str] = {}

    # Track the last todo_write call+result to append at the end
    last_todo_call: str | None = None
    last_todo_result: str | None = None

    for msg in all_messages:
        # ── Assistant tool calls: capture full args ──────────────────
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                tc_id = tc.get("id", "")
                if tc_id:
                    tc_id_to_name[tc_id] = name

                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                args_str = json.dumps(args, ensure_ascii=False)

                if name == "todo_write":
                    # Only keep the last todo_write — it supersedes earlier ones
                    last_todo_call = f"[tool_call] {name}({args_str})"
                    last_todo_result = None  # reset until result arrives
                    continue

                blocks.append(f"[tool_call] {name}({args_str})")

        # ── Tool results: preserve full data ─────────────────────────
        if msg.get("role") == "tool":
            tool_name = tc_id_to_name.get(msg.get("tool_call_id", ""), "")

            content = msg.get("content", "")
            # Strip system-reminder suffixes before parsing
            raw = content.split("<system-reminder>")[0].strip()

            if tool_name == "todo_write":
                # Keep the last todo result alongside the last call
                last_todo_result = f"[tool_result] {tool_name}:\n{raw}"
                continue

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

            # ── Search results: keep only LLM-selected clauses ───────
            if "clauses" in data and "total_found" in data:
                clauses = data["clauses"]
                selected = [c for c in clauses
                            if isinstance(c, dict) and c.get("selected")]
                if selected:
                    data["clauses"] = selected
                else:
                    # Fallback for non-agentic mode: keep score ≥ 5.0
                    data["clauses"] = [
                        c for c in clauses
                        if isinstance(c.get("score", 10), (int, float))
                        and c.get("score", 10) >= 5.0
                    ]

            # ── Engineering tools: strip clause_references ────────────
            # These are static registry metadata about which clauses the
            # tool *implements*, NOT evidence that a clause was actually
            # retrieved.  Keeping them causes the grounding validator to
            # treat them as retrieved evidence, masking missing lookups.
            if tool_name in ("engineering_calculator", "search_engineering_tools"):
                data.pop("clause_references", None)
                for r in data.get("results", []):
                    if isinstance(r, dict):
                        r.pop("clause_references", None)

            # Remove noise fields
            for drop_key in ("_referenced_but_not_retrieved",):
                data.pop(drop_key, None)

            blocks.append(
                f"[tool_result] {tool_name}:\n"
                + json.dumps(data, ensure_ascii=False, default=str)
            )

    # Append the last todo_write (plan state) at the end — it's the
    # most important context for continuation after ask_user.
    if last_todo_call:
        blocks.append(last_todo_call)
    if last_todo_result:
        blocks.append(last_todo_result)

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
