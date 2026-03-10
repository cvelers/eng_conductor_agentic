"""Core agent loop — think → act → observe → repeat.

Uses the ``openai`` Python package for native tool calling with streaming.
Bridges sync OpenAI streaming into async iteration via a background thread.
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
      {"type": "done", "content": str}
      {"type": "error", "message": str}
    """
    full_response = ""
    tool_round = 0
    total_tool_calls = 0

    # Loop detection state
    consecutive_names: list[str] = []
    loop_breaker_count = 0

    all_messages = [{"role": "system", "content": system_prompt}] + messages

    while tool_round < max_rounds:
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

        if not tool_calls:
            full_response = assistant_content
            break

        # ── Loop detection ───────────────────────────────────────────
        for tc in tool_calls:
            consecutive_names.append(tc["name"])
        total_tool_calls += len(tool_calls)

        force_stop = False
        if len(consecutive_names) >= MAX_CONSECUTIVE_SAME_TOOL + 1:
            recent = consecutive_names[-(MAX_CONSECUTIVE_SAME_TOOL + 1) :]
            if len(set(recent)) == 1:
                force_stop = True
        if total_tool_calls > 15:
            force_stop = True

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

            # Build summary for UI
            summary = _summarize_result(result_str, tc["name"], elapsed_ms)
            yield {
                "type": "tool_result",
                "tool": tc["name"],
                "result": result_str[:20_000],
                "status": status,
                "summary": summary,
            }

            all_messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_str[:30_000],
            })

        tool_round += 1

    yield {"type": "done", "content": full_response}


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
