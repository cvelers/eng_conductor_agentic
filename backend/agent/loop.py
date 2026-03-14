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

_META_TOOLS = {"todo_write", "ask_user"}
_GROUNDING_EVIDENCE_TOOLS = {
    "eurocode_search",
    "read_clause",
    "engineering_calculator",
    "math_calculator",
    "web_search",
    "fetch_url",
}

# ── System Reminders (Claude Code pattern) ─────────────────────────
# Injected after every tool result to keep the agent on track.
# These repeat key behavioral instructions at the exact moment the
# agent is deciding what to do next — the "high attention zone" at
# the end of the context window.

_TODO_NUDGE = (
    " If the task is multi-step, the plan changed, or you are about to ask "
    "the user or finish, consider updating `todo_write` so the plan stays aligned."
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
        "If any input was assumed rather than looked up, state this clearly and include it "
        "in the final `## Assumptions` section."
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
        "or V_Ed. Any defaulted or idealized input must appear in the final "
        "`## Assumptions` section."
        "</system-reminder>"
    ),
    "todo_write": (
        "\n\n<system-reminder>"
        "Plan updated. Proceed with the next step. "
        "Update todo_write again when you complete a milestone, change approach, "
        "or are about to finish."
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


def _is_internal_harness_user_message(content: str) -> bool:
    text = str(content or "").strip()
    return (
        text.startswith("GROUNDING VALIDATION FAILED")
        or text.startswith("SELF-REVIEW FAILED")
        or text.startswith("STOP CALLING TOOLS.")
    )


_ASSUMPTIONS_SECTION_RE = re.compile(
    r"(?ims)(?:^|\n)\s{0,3}#{1,6}\s*assumptions\s*$\n?(?P<body>.*?)(?=(?:\n\s{0,3}#{1,6}\s+\S)|\Z)"
)
_MARKDOWN_LIST_PREFIX_RE = re.compile(r"^(?:[-*+]\s+|\d+\.\s+)")


def _extract_assumptions_from_response(response_text: str) -> tuple[list[str], bool]:
    """Return (assumptions, found_section) from a markdown response."""
    match = _ASSUMPTIONS_SECTION_RE.search(str(response_text or ""))
    if not match:
        return [], False

    body = match.group("body").strip()
    if not body:
        return [], True

    items: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = _MARKDOWN_LIST_PREFIX_RE.sub("", line).strip()
        if line:
            items.append(line)

    if not items and body:
        items = [re.sub(r"\s+", " ", body)]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped, True


def _strip_assumptions_section_from_response(response_text: str) -> str:
    """Remove the visible assumptions section after assumptions were extracted."""
    stripped = _ASSUMPTIONS_SECTION_RE.sub("\n", str(response_text or ""))
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


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
You are a grounding validator for a civil engineering assistant. Your ONLY job is to \
check whether the response below is fully supported by the available evidence. \
You are an independent auditor — the response author has no control over your verdict.

## Latest User Question
{latest_user_message}

{conversation_history_section}\
## Tool Results From This Session
{tool_results}

## Response to Validate
{response}

## Instructions
Check EVERY factual claim in the response against the evidence above:
1. First determine whether the latest user question is a civil engineering technical question \
or a direct follow-up to an earlier civil engineering answer in this thread. If it is NOT, then \
the only valid response is a brief refusal that says the assistant only helps with civil engineering \
technical questions.
2. For in-scope technical answers, every stated or implied assumption / default / modelling \
idealisation must be explicitly listed in a dedicated `Assumptions` section near the end of the \
response. If there were no assumptions, the response should say so explicitly (for example `- None.`). \
Do NOT accept silent assumptions just because a tool ran.
3. Every Eurocode clause cited (e.g. "Cl. 6.2.5") — is it in the tool results \
from explicit clause evidence returned by `eurocode_search` / `read_clause`, \
or in a previous validated response/session memory? Calculator metadata alone \
is NOT enough to ground a cited clause.
4. Every numeric value stated (fy, Wpl, dimensions, safety factors) — does it \
match a tool output or a value from a previous validated response? Check the \
actual numbers.
5. Every calculation result — was it produced by a calculator tool or stated in \
a previous validated response?
6. Any formula or rule — is it from a retrieved clause or previous response, \
or recited from memory?
7. If a calculator used assumed inputs, make sure those assumptions came from \
the user, previous validated context, or were explicitly stated in the response's \
`Assumptions` section.
8. Preserve engineering semantics of symbols and variable roles. A prior value \
is only grounded for the SAME physical quantity. Flag demand/resistance swaps \
such as using `M_Rd`, `M_c,Rd`, or `M_b,Rd` as `M_Ed`, or reusing any resistance \
value as a load effect just because the number matches.
9. Inspect calculator CALL arguments as well as final prose. A calculator run does \
NOT make its inputs valid. If a tool input lacks evidence, contradicts prior context, \
or changes the meaning of an established symbol, flag it.
10. Prior validated conversation/session memory only grounds facts that are EXPLICITLY \
present there. It does NOT license new technical explanations, new formulas, new clauses, \
new checks, or a new topic just because the same member, standard, or design context was \
discussed earlier.
11. Topic continuity is not evidence. If the response introduces a new subject (for example, \
shear after earlier turns only discussed bending/classification/LTB), that new subject must \
be explicitly supported by current tool results or explicit prior validated text about that \
same subject.
12. If current-turn tool results are empty or irrelevant to the response topic, and prior \
validated context does not explicitly contain the needed claim, the response is ungrounded.

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
            validator_content = msg.get("validator_content")
            if isinstance(validator_content, str):
                raw = validator_content.strip()
            else:
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
        # Keep structured continuation memory / compaction summaries as validator evidence.
        if role == "system":
            text = str(content).strip()
            if (
                text.startswith("Continuation memory:")
                or "Conversation memory (auto-compacted):" in text
                or text.startswith("<tool-context>")
            ):
                blocks.append(f"[SESSION MEMORY]\n{text}")
            continue
        if role == "user" and _is_internal_harness_user_message(str(content)):
            continue
        # Previous user questions
        if role == "user":
            blocks.append(f"[USER] {content}")
        # Previous assistant text responses (no tool_calls = final answer)
        elif role == "assistant" and not msg.get("tool_calls"):
            blocks.append(f"[ASSISTANT — previously validated] {content}")
    return "\n\n".join(blocks) if blocks else ""


_FINAL_ANSWER_REVIEW_PROMPT = """\
You are the SAME engineering agent performing a hidden end-of-turn self-review \
before a draft answer is shown to the user.

Your task is to decide:
1. What kind of answer this is.
2. Whether this draft answer needs grounding validation.
3. Whether the draft can be shown as-is, must be rewritten without tools, or requires \
more evidence from tools before answering.

## Latest User Question
{latest_user_message}

## Conversation History / Session Memory
{conversation_history}

## Tool Results Available In This Turn
{tool_results}

## Draft Answer
{response}

## Decision rules
- Choose exactly one `answer_type`:
  - `civil_engineering_technical`: civil engineering technical content, calculations, \
code interpretation, factual design guidance, or technical follow-up grounded in engineering context.
  - `conversation_meta`: questions about what was said earlier in the chat, what the user \
first asked, what you answered previously, or other conversation-management/meta content \
that only recalls established thread history.
  - `out_of_scope`: anything that is not a civil engineering technical request or a direct \
follow-up to an in-scope engineering discussion.
- Set `requires_validation=true` for `civil_engineering_technical`.
- Set `requires_validation=false` for `conversation_meta` and `out_of_scope`.
- Set `required_action` to exactly one of:
  - `answer_ok`: the draft can be shown as-is
  - `rewrite_without_tools`: the draft should be rewritten without calling tools
  - `gather_tools`: the draft needs more evidence from tools before answering
- Use `rewrite_without_tools` when:
  - the user is out of scope and the draft does NOT clearly refuse / redirect to civil engineering scope
  - the answer should stay within existing evidence but needs formatting repair, especially a final \
    `## Assumptions` section for a civil engineering technical answer
- Use `gather_tools` when the draft makes claims that are not adequately supported by the evidence above.
- For `conversation_meta`, never require tools just to restate chat history.
- Prior validated context can support the answer only when it explicitly contains the same \
fact being restated. Do not treat "same standard", "same member", or "same topic area" as \
enough evidence for new technical claims.
- For `civil_engineering_technical`, require a final `## Assumptions` section. If it is missing \
or incomplete but the answer otherwise stays within existing evidence, use `rewrite_without_tools`.
- For `out_of_scope`, `answer_ok` is only allowed when the draft briefly refuses and redirects \
the user back to civil engineering technical scope.

Respond with ONLY JSON:
{{"answer_type": "civil_engineering_technical", "requires_validation": true, "required_action": "answer_ok", "reason": "short reason"}}\
"""


async def _self_review_final_answer(
    client: OpenAI,
    model: str,
    response_text: str,
    all_messages: list[dict],
    *,
    max_tokens: int = 300,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Ask the same model whether the draft answer is technical and/or unsupported."""
    tool_results = _build_tool_results_for_validator(all_messages)
    conversation_history = _build_conversation_history_for_validator(all_messages) or "(none)"
    latest_user_message = ""
    for msg in reversed(all_messages):
        if msg.get("role") == "user":
            latest_user_message = str(msg.get("content", "") or "").strip()
            if latest_user_message and not _is_internal_harness_user_message(latest_user_message):
                break
    prompt = _FINAL_ANSWER_REVIEW_PROMPT.format(
        latest_user_message=latest_user_message or "(none)",
        conversation_history=conversation_history,
        tool_results=tool_results,
        response=response_text,
    )

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
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
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("self-review did not return a JSON object")
        answer_type = str(data.get("answer_type", "") or "").strip() or "civil_engineering_technical"
        required_action = str(data.get("required_action", "") or "").strip()
        if required_action not in {"answer_ok", "rewrite_without_tools", "gather_tools"}:
            required_action = (
                "gather_tools"
                if bool(data.get("requires_tools_before_answering"))
                else "answer_ok"
            )
        return {
            "answer_type": answer_type,
            "requires_validation": bool(data.get("requires_validation")),
            "required_action": required_action,
            "reason": str(data.get("reason", "") or ""),
        }
    except Exception:
        logger.exception("final_answer_self_review_failed")
        # Conservative fallback: validate the answer, but don't force tools solely
        # because the self-review channel failed.
        return {
            "answer_type": "civil_engineering_technical",
            "requires_validation": True,
            "required_action": "answer_ok",
            "reason": "self-review failed",
        }


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
    latest_user_message = ""
    for msg in reversed(all_messages):
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", "") or "").strip()
        if not content or _is_internal_harness_user_message(content):
            continue
        latest_user_message = content
        break

    # Only include conversation history section if there are previous turns
    if conversation_history:
        history_section = (
            "## Conversation History (previously validated)\n"
            f"{conversation_history}\n\n"
        )
    else:
        history_section = ""

    prompt = _GROUNDING_VALIDATOR_PROMPT.format(
        latest_user_message=latest_user_message or "(none)",
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
        return {
            "valid": False,
            "issues": ["Grounding validator returned malformed output; technical answer cannot be released unvalidated."],
        }
    except Exception:
        logger.exception("Grounding validation LLM call failed")
        return {
            "valid": False,
            "issues": ["Grounding validator failed; technical answer cannot be released without validation."],
        }


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
    last_ask_user_payload: dict[str, Any] | None = None

    # Grounding validation retry tracking
    validation_retries = 0
    force_tool_use = False
    force_no_tool_use = False
    used_grounding_tools_this_turn = False
    final_assumptions: list[str] = []

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
        if loop_breaker_count >= LOOP_BREAKER_HARD_LIMIT or force_no_tool_use:
            if tools:
                request_kwargs["tools"] = tools
                request_kwargs["tool_choice"] = "none"
        else:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "required" if force_tool_use and tools else "auto"

        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort

        # ── Stream the LLM response ──────────────────────────────────
        assistant_content = ""
        stream_tool_calls: dict[int, dict] = {}
        got_response = False

        # Buffer text events until the round is complete.
        # If the model emits tool calls, discard the text for that round:
        # tool-call rounds are internal working state, not user-facing output.
        # Only a no-tool final round is allowed to become visible answer text.
        pending_events: list[dict] = []
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
                            pending_events.append({"type": "delta", "content": visible})
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
                        pending_events.append({"type": "delta", "content": visible})

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
        # Never preserve hidden scratch prose from tool-call rounds.
        # The user never saw that text, and replaying it across turns causes
        # stale assumptions and half-formed reasoning to leak forward.
        if assistant_content and not tool_calls:
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
            # Only a no-tool round is a candidate final answer.
            full_response += assistant_content
            review = {
                "requires_validation": False,
                "required_action": "answer_ok",
                "reason": "",
            }
            if grounding_validation and full_response.strip():
                review = await _self_review_final_answer(
                    client=client,
                    model=model,
                    response_text=full_response,
                    all_messages=all_messages,
                    max_tokens=300,
                    reasoning_effort=reasoning_effort or None,
                )

            required_action = str(review.get("required_action", "") or "answer_ok")

            if required_action == "gather_tools" and tools:
                pending_events.clear()
                full_response = ""
                force_tool_use = True
                force_no_tool_use = False
                all_messages.pop()
                reason = str(review.get("reason", "") or "").strip()
                all_messages.append({
                    "role": "user",
                    "content": (
                        "SELF-REVIEW FAILED. Your draft answer is not sufficiently supported yet. "
                        + (f"Reason: {reason}\n\n" if reason else "")
                        + "Gather the missing evidence with the appropriate tools before answering."
                    ),
                })
                tool_round += 1
                continue

            if required_action == "rewrite_without_tools":
                pending_events.clear()
                full_response = ""
                force_tool_use = False
                force_no_tool_use = True
                all_messages.pop()
                reason = str(review.get("reason", "") or "").strip()
                all_messages.append({
                    "role": "user",
                    "content": (
                        "SELF-REVIEW FAILED. Rewrite your draft answer without calling tools. "
                        + (f"Reason: {reason}\n\n" if reason else "")
                        + "Do not add new facts. If the user is out of scope, briefly refuse and "
                        "redirect to civil engineering technical questions. If the answer is in scope, "
                        "keep it grounded and repair the final `## Assumptions` section."
                    ),
                })
                tool_round += 1
                continue

            # ── Grounding validation (independent LLM) ───────────────
            should_validate = (
                grounding_validation
                and validator_client is not None
                and validator_model
                and bool(review.get("requires_validation"))
            )
            if should_validate:
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
                    full_response = ""
                    had_evidence_this_turn = used_grounding_tools_this_turn
                    all_messages.pop()

                    if not had_evidence_this_turn and tools:
                        force_tool_use = True
                        all_messages.append({
                            "role": "user",
                            "content": (
                                "GROUNDING VALIDATION FAILED. Your draft answer was not supported by "
                                "retrieved evidence. An independent validator found these issues:\n\n"
                                f"{issues_text}\n\n"
                                "Do NOT answer from memory. First gather the missing evidence with the "
                                "appropriate search, clause, or calculation tools. Only then answer."
                            ),
                        })
                    else:
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
                    tool_round += 1
                    continue

            final_assumptions, _ = _extract_assumptions_from_response(full_response)
            full_response = _strip_assumptions_section_from_response(full_response)
            pending_events = (
                [{"type": "delta", "content": full_response}]
                if full_response
                else []
            )
            force_no_tool_use = False
            # Validation passed (or not needed) — flush buffered deltas
            for evt in pending_events:
                yield evt
            pending_events.clear()
            break

        # Tool-call rounds are internal. Never surface their prose to the user.
        pending_events.clear()
        if any(tc["name"] not in _META_TOOLS for tc in tool_calls):
            force_tool_use = False
        force_no_tool_use = False

        # ── Loop detection ───────────────────────────────────────────
        # Don't count meta-tools toward tool budget
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

        # ── Execute tool calls ───────────────────────────────────────
        # If ask_user is in the batch, execute it FIRST and skip the rest.
        # The agent shouldn't be calling other tools alongside ask_user.
        has_ask_user = any(tc["name"] == "ask_user" for tc in tool_calls)
        if has_ask_user:
            tool_calls = [tc for tc in tool_calls if tc["name"] == "ask_user"][:1]
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
                last_ask_user_payload = {
                    "question": ask_data.get("question", ""),
                    "options": ask_data.get("options", []),
                    "context": ask_data.get("context", ""),
                }
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

            if tc["name"] in _GROUNDING_EVIDENCE_TOOLS and status == "ok":
                used_grounding_tools_this_turn = True

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
                # Preserve full raw tool output for the grounding validator.
                # The orchestrator still receives the budgeted `content` form.
                "validator_content": result_str,
            })

            if asked_user:
                logger.info(
                    "ask_user_hard_stop",
                    extra={"skipped_following_calls": len(tool_calls) - tc_idx - 1},
                )
                break

        # If ask_user was called, stop the loop — wait for user input
        if asked_user:
            pending_events.clear()
            break

        tool_round += 1

    # Build a condensed tool context summary so future turns remember
    # what tools were called and key results from this turn.
    tool_summary = _build_tool_context(all_messages)
    if tool_summary:
        yield {"type": "_tool_context", "summary": tool_summary}

    session_memory = _build_session_memory(
        all_messages,
        plan_steps=plan_steps,
        full_response=full_response,
        assumptions=final_assumptions,
        ask_user_payload=last_ask_user_payload,
    )
    if session_memory:
        yield {"type": "_session_memory", "memory": session_memory}

    # Emit token count of what the NEXT request's input will look like:
    # prior history + this turn's text + tool context summary.
    from backend.agent.context import estimate_messages_tokens, estimate_tokens
    # Approximate next-turn input: current messages + assistant response + tool context
    next_turn_msgs = list(messages) + [
        {"role": "assistant", "content": full_response + ("\n\n" + tool_summary if tool_summary else "")},
    ]
    session_tokens = estimate_messages_tokens(next_turn_msgs, system_prompt)
    yield {"type": "_session_tokens", "tokens": session_tokens}

    yield {"type": "done", "content": full_response, "assumptions": final_assumptions}


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


def _extract_task_anchor(all_messages: list[dict]) -> str:
    """Return the latest user task, skipping validator and ask_user wrapper text."""
    for msg in reversed(all_messages):
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", "") or "").strip()
        if not content or _is_internal_harness_user_message(content):
            continue
        if content.startswith("[User's answer to your ask_user question]"):
            continue
        return re.sub(r"\s+", " ", content)[:280]
    return ""


def _extract_selected_clauses_for_memory(all_messages: list[dict]) -> list[dict[str, str]]:
    """Collect the clauses that should anchor future turns."""
    clauses: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    tc_id_to_name: dict[str, str] = {}

    for msg in all_messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                name = tc.get("function", {}).get("name", "")
                if tc_id:
                    tc_id_to_name[tc_id] = name

        if msg.get("role") != "tool":
            continue

        tool_name = tc_id_to_name.get(msg.get("tool_call_id", ""), "")
        if tool_name not in {"eurocode_search", "read_clause"}:
            continue

        raw = str(msg.get("validator_content") or msg.get("content") or "")
        raw = raw.split("<system-reminder>")[0].strip()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or "clauses" not in data:
            continue

        clause_rows = data.get("clauses", [])
        if not isinstance(clause_rows, list):
            continue

        if tool_name == "eurocode_search":
            selected = [
                c for c in clause_rows
                if isinstance(c, dict) and c.get("selected")
            ]
            if selected:
                clause_rows = selected
            else:
                clause_rows = [
                    c for c in clause_rows
                    if isinstance(c, dict)
                    and isinstance(c.get("score", 10), (int, float))
                    and c.get("score", 10) >= 5.0
                ]

        for clause in clause_rows:
            if not isinstance(clause, dict):
                continue
            standard = str(clause.get("standard", "") or "")
            clause_id = str(clause.get("clause_id", "") or "")
            title = str(clause.get("title") or clause.get("clause_title") or "")
            if not standard or not clause_id:
                continue
            key = (standard, clause_id)
            if key in seen:
                continue
            seen.add(key)
            clauses.append({
                "standard": standard,
                "clause_id": clause_id,
                "title": title,
            })
    return clauses[:6]


def _summarize_for_session_memory(tool_name: str, raw: str) -> str:
    """Build a compact continuation summary from a tool result."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        text = re.sub(r"\s+", " ", raw).strip()
        return text[:220]

    if not isinstance(data, dict):
        return str(data)[:220]

    if tool_name == "ask_user":
        question = str(data.get("question", "") or "").strip()
        return question[:220]

    outputs = data.get("outputs")
    if isinstance(outputs, dict) and outputs:
        items = [f"{k}={v}" for k, v in list(outputs.items())[:4]]
        return ", ".join(items)[:220]

    clauses = data.get("clauses")
    if isinstance(clauses, list) and clauses:
        refs: list[str] = []
        for clause in clauses[:3]:
            if not isinstance(clause, dict):
                continue
            standard = str(clause.get("standard", "") or "").strip()
            clause_id = str(clause.get("clause_id", "") or "").strip()
            if standard and clause_id:
                refs.append(f"{standard} {clause_id}")
        if refs:
            return "; ".join(refs)[:220]

    results = data.get("results")
    if isinstance(results, list) and results:
        names = []
        for row in results[:3]:
            if isinstance(row, dict):
                name = row.get("tool_name") or row.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
        if names:
            return ", ".join(names)[:220]

    if "error" in data:
        return f"error: {str(data['error'])[:180]}"

    return _summarize_result(raw, tool_name)[:220]


def _extract_recent_tool_results_for_memory(all_messages: list[dict]) -> list[dict[str, str]]:
    """Collect compact summaries of recent tool outputs for continuation."""
    tc_id_to_name: dict[str, str] = {}
    facts: list[dict[str, str]] = []
    skip = {"todo_write", "grounding_validator"}

    for msg in all_messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                name = tc.get("function", {}).get("name", "")
                if tc_id:
                    tc_id_to_name[tc_id] = name

        if msg.get("role") != "tool":
            continue

        tool_name = tc_id_to_name.get(msg.get("tool_call_id", ""), "")
        if not tool_name or tool_name in skip:
            continue

        raw = str(msg.get("validator_content") or msg.get("content") or "")
        raw = raw.split("<system-reminder>")[0].strip()
        summary = _summarize_for_session_memory(tool_name, raw)
        if not summary:
            continue
        facts.append({"tool": tool_name, "summary": summary})

    return facts[-4:]


def _build_session_memory(
    all_messages: list[dict],
    *,
    plan_steps: list[dict[str, str]],
    full_response: str,
    assumptions: list[str],
    ask_user_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build structured continuation memory for the next turn."""
    selected_clauses = _extract_selected_clauses_for_memory(all_messages)
    preferred_standard = ""
    if selected_clauses:
        counts: dict[str, int] = {}
        for clause in selected_clauses:
            standard = clause.get("standard", "")
            if standard:
                counts[standard] = counts.get(standard, 0) + 1
        preferred_standard = max(counts.items(), key=lambda item: item[1])[0] if counts else ""

    memory: dict[str, Any] = {
        "state": "waiting_for_user" if ask_user_payload else "final",
        "task_anchor": _extract_task_anchor(all_messages),
        "preferred_standard": preferred_standard,
        "selected_clauses": selected_clauses,
        "recent_tool_results": _extract_recent_tool_results_for_memory(all_messages),
    }

    if plan_steps:
        memory["plan"] = [{
            "id": step.get("id", ""),
            "text": step.get("text", ""),
            "status": step.get("status", "pending"),
        } for step in plan_steps[:8]]

    if assumptions:
        memory["assumptions"] = list(assumptions[:12])

    if ask_user_payload:
        memory["ask_user"] = ask_user_payload

    answer_summary = re.sub(r"\s+", " ", full_response or "").strip()
    if answer_summary:
        memory["answer_summary"] = answer_summary[:320]

    return memory


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
