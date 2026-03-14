"""FEA Analyst — agentic loop for finite element analysis.

The FEA analyst is a separate LLM-powered sub-agent that autonomously builds
FEA models through tool calls, triggers the client-side solver, interprets
results, and provides engineering reports.

The key architectural feature: the analyst runs on the backend (LLM), but the
solver runs on the frontend (browser). Communication uses a pause/resume
pattern via asyncio.Queue for result callbacks.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from uuid import uuid4

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.orchestrator.fea_prompts import FEA_ANALYST_SYSTEM, FEA_TOOLS
from backend.orchestrator.fea_tools import (
    FEAModelState,
    execute_fea_tool,
    get_result_query_coverage_key,
)
from backend.utils.json_utils import parse_json_loose

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 25
MAX_REAL_TOOL_CALLS = 30
_META_TOOLS = {"todo_write", "ask_user", "fea_ask_user", "fea_record_assumptions"}
_REQUIRED_RESULT_QUERIES = {"displacements", "reactions", "element_forces"}
_SEMANTIC_PATCH_TOOLS = {
    "fea_query_model",
    "fea_define_rectilinear_frame",
    "fea_patch_frame_geometry",
    "fea_patch_supports",
    "fea_patch_members",
    "fea_patch_loads",
}
_LOW_LEVEL_BUILD_TOOLS = {
    "fea_add_nodes",
    "fea_add_elements",
    "fea_assign_sections",
    "fea_assign_material",
    "fea_set_restraints",
    "fea_add_loads",
    "fea_set_analysis_type",
}
_SEMANTIC_SESSION_ALLOWED_TOOLS = {
    "todo_write",
    "fea_record_assumptions",
    "ask_user",
    "fea_clear",
    "fea_query_model",
    "fea_define_rectilinear_frame",
    "fea_patch_frame_geometry",
    "fea_patch_supports",
    "fea_patch_members",
    "fea_patch_loads",
    "fea_check_model",
    "fea_solve",
    "fea_get_results",
    "fea_set_view",
}


@dataclass
class FEASession:
    """In-memory state for an active FEA analysis session."""

    session_id: str
    model_state: FEAModelState = field(default_factory=FEAModelState)
    result_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    answer_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    messages: list[dict[str, Any]] = field(default_factory=list)
    solve_attempts: int = 0


class FEAAnalystLoop:
    """Agentic loop for FEA analysis.

    Yields events that the streaming endpoint forwards to the frontend:
      - ("fea_thinking", {"content": str})
      - ("fea_tool_call", {"tool": str, "args": dict})
      - ("fea_tool_result", {"tool": str, "result": Any, "status": str, "summary": str})
      - ("plan", {"steps": [...]})
      - ("plan_update", {"step_id": str, "status": str})
      - ("fea_command", {"commands": [...]})
      - ("fea_solve_request", {"session_id": str, "load_case_id": str})
      - ("fea_user_query", {"session_id": str, "question": str, "options": [...], "context": str})
      - ("fea_view_command", {"action": str, ...})
      - ("fea_complete", {"summary": str, "assumptions": [...]})
    """

    def __init__(
        self,
        *,
        llm: LLMProvider,
        settings: Settings,
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.session = FEASession(session_id=uuid4().hex[:12])

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def provide_results(self, results: dict) -> None:
        """Called by the /api/fea/results endpoint to feed solver results back."""
        # Don't mark solved on errors — let the LLM diagnose and retry
        if not (isinstance(results, dict) and results.get("error")):
            self.session.model_state.results = results
            self.session.model_state.solved = True
        self.session.result_queue.put_nowait(results)

    def provide_answer(self, answer: str) -> None:
        """Called by the /api/fea/answer endpoint to feed user answers back."""
        self.session.answer_queue.put_nowait(answer)

    async def run_stream(
        self,
        query: str,
        *,
        history: list | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Run the FEA analyst agentic loop.

        This is an async generator that yields events. When a solve is needed,
        it yields a fea_solve_request and then awaits results from the queue.
        """
        # Build initial messages
        messages: list[dict] = [
            {"role": "system", "content": FEA_ANALYST_SYSTEM},
        ]
        restored_session_memory = self._extract_latest_fea_session_memory(history)

        if restored_session_memory:
            self._restore_from_session_memory(restored_session_memory)
            restored_note = self._build_restored_session_context(restored_session_memory)
            if restored_note:
                messages.append({"role": "system", "content": restored_note})
            restored_fea_session = restored_session_memory.get("fea_session", {})
            if isinstance(restored_fea_session, dict):
                model_snapshot = restored_fea_session.get("model_snapshot")
                if isinstance(model_snapshot, dict):
                    yield ("fea_state_restored", {
                        "model_snapshot": model_snapshot,
                        "results_snapshot": restored_fea_session.get("results_snapshot"),
                        "model_summary": restored_fea_session.get("model_summary"),
                    })
        elif history:
            # Add minimal visible history only when we do not have structured FEA memory.
            for h in history[-6:]:  # Last 6 messages for context
                if hasattr(h, "role"):
                    messages.append({"role": h.role, "content": h.content})
                elif isinstance(h, dict):
                    messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})

        messages.append({"role": "user", "content": query})
        self.session.messages = messages

        yield ("fea_thinking", {"content": "Analyzing structural problem..."})

        # ── Loop detection state ──────────────────────────────────
        tool_history: list[str] = []      # Flat list of all tool names called
        results_retrieved = False          # Set after fea_get_results is called
        force_summary_attempts = 0         # How many times we've told LLM to write summary
        plan_steps: list[dict[str, Any]] = deepcopy(self.session.model_state.plan)
        plan_emitted = bool(plan_steps)
        total_tool_calls = 0
        result_queries_requested: set[str] = set()
        malformed_tool_call_retries = 0
        restored_from_memory = restored_session_memory is not None
        solved_this_turn = False
        semantic_model_active = isinstance(self.session.model_state.semantic_model, dict)

        # ── Agentic tool-calling loop ─────────────────────────────
        for iteration in range(MAX_TOOL_ITERATIONS):
            # Prune context to fit within token budget
            messages = self._prune_messages(messages)

            # Call LLM with tool definitions
            try:
                response = self._call_llm(
                    messages,
                    semantic_model_active=semantic_model_active,
                )
            except Exception as exc:
                logger.exception("fea_analyst_llm_failed")
                yield ("fea_thinking", {"content": f"LLM call failed: {exc}"})
                break

            finish_reason = self._extract_finish_reason(response)
            if "malformed_function_call" in finish_reason:
                malformed_tool_call_retries += 1
                if malformed_tool_call_retries >= 3:
                    yield ("fea_thinking", {
                        "content": (
                            "The model provider repeatedly rejected malformed tool calls. "
                            "Stopping here so the current model state is preserved for inspection."
                        ),
                    })
                    break
                yield ("fea_thinking", {"content": "Retrying after a malformed tool call..."})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response produced a malformed function/tool call rejected by the model provider. "
                        "Continue from the current FEA model state and retry with 1-2 valid tool calls only. "
                        "Use the exact tool schema, include every required field, and do not output prose."
                    ),
                })
                continue

            # Parse response for tool calls
            tool_calls = self._extract_tool_calls(response)
            if tool_calls:
                malformed_tool_call_retries = 0

            has_ask_user = any(tc["name"] in {"ask_user", "fea_ask_user"} for tc in tool_calls)
            if has_ask_user:
                tool_calls = [
                    tc for tc in tool_calls
                    if tc["name"] in {"ask_user", "fea_ask_user"}
                ][:1]

            if (
                restored_from_memory
                and semantic_model_active
                and tool_calls
                and any(tc["name"] in _LOW_LEVEL_BUILD_TOOLS for tc in tool_calls)
                and not any(tc["name"] in _SEMANTIC_PATCH_TOOLS or tc["name"] == "fea_clear" for tc in tool_calls)
            ):
                yield ("fea_thinking", {"content": "Editing the restored frame semantically before touching FE entities..."})
                messages.append({
                    "role": "user",
                    "content": (
                        "A rectilinear semantic frame model is already restored. "
                        "Do not rebuild it with raw fea_add_nodes/fea_add_elements style calls. "
                        "First use fea_query_model if needed, then use the semantic tools: "
                        "fea_patch_frame_geometry, fea_patch_supports, fea_patch_members, or fea_patch_loads. "
                        "Only fall back to raw FE tools if the requested geometry is genuinely irregular."
                    ),
                })
                continue

            # ── Post-results guard: stop tool calls after results retrieved ──
            if results_retrieved and tool_calls and any(tc["name"] != "fea_get_results" for tc in tool_calls):
                needs_full_result_bundle = solved_this_turn or not restored_from_memory
                missing_queries = sorted(_REQUIRED_RESULT_QUERIES - result_queries_requested)
                if needs_full_result_bundle and missing_queries and any(tc["name"] != "fea_get_results" for tc in tool_calls):
                    yield ("fea_thinking", {"content": "Collecting the remaining result sets..."})
                    messages.append({
                        "role": "user",
                        "content": self._build_missing_results_prompt(missing_queries),
                    })
                    continue
                force_summary_attempts += 1
                if force_summary_attempts >= 2:
                    # LLM is ignoring guidance — force completion
                    text = self._extract_text(response)
                    yield ("fea_complete", {
                        "summary": text or "Analysis complete. Review the model and results in the viewer.",
                        "assumptions": list(self.session.model_state.assumptions),
                        "session_memory": self._build_session_memory(text),
                    })
                    return
                yield ("fea_thinking", {"content": "Preparing engineering summary..."})
                messages.append({"role": "assistant", "content": "I have all the results. Let me write the summary."})
                messages.append({
                    "role": "user",
                    "content": (
                        "STOP calling tools. The solver has run and results were retrieved. "
                        "Write your final engineering summary as plain text. Include:\n"
                        "- Maximum deflection and its location\n"
                        "- Maximum bending moment and its location\n"
                        "- Support reactions\n"
                        "- Engineering interpretation\n"
                        "Respond with PLAIN TEXT ONLY — no JSON, no tool calls."
                    ),
                })
                continue

            if not tool_calls:
                # No tool calls — this is a text response (final answer or thinking)
                text = self._extract_text(response)
                if text:
                    # Check if this is a final answer (after results are available)
                    if self.session.model_state.solved or results_retrieved:
                        needs_full_result_bundle = solved_this_turn or not restored_from_memory
                        missing_queries = sorted(_REQUIRED_RESULT_QUERIES - result_queries_requested)
                        if needs_full_result_bundle and missing_queries:
                            yield ("fea_thinking", {"content": "Gathering full FEA results before reporting..."})
                            messages.append({
                                "role": "user",
                                "content": self._build_missing_results_prompt(missing_queries),
                            })
                            continue
                        yield ("fea_thinking", {"content": text})
                        messages.append({"role": "assistant", "content": text})
                        yield ("fea_complete", {
                            "summary": text,
                            "assumptions": list(self.session.model_state.assumptions),
                            "session_memory": self._build_session_memory(text),
                        })
                        return
                    yield ("fea_thinking", {"content": text})
                    # Add to message history
                    messages.append({"role": "assistant", "content": text})
                    # If not solved yet, the LLM might be thinking out loud
                    # Continue the loop with a forceful nudge
                    messages.append({
                        "role": "user",
                        "content": (
                            "Now call the tools to build the model. Respond ONLY with a JSON object "
                            "in this exact format, nothing else:\n"
                            '```json\n{"tool_calls": [{"name": "fea_add_nodes", "args": {...}}]}\n```'
                        ),
                    })
                    continue
                break

            real_calls = [tc for tc in tool_calls if tc["name"] not in _META_TOOLS]
            if real_calls and not plan_emitted and all(tc["name"] != "todo_write" for tc in tool_calls):
                yield ("fea_thinking", {"content": "Planning the analysis workflow..."})
                messages.append({
                    "role": "user",
                    "content": (
                        "Before executing real FEA model-building tools, call todo_write with a short ordered plan "
                        "(3-6 steps) for this analysis. Then continue from that plan."
                    ),
                })
                continue
            total_tool_calls += len(real_calls)
            if total_tool_calls > MAX_REAL_TOOL_CALLS and not results_retrieved:
                messages.append({
                    "role": "user",
                    "content": (
                        "Tool budget is nearly exhausted. Do not replan or rebuild the same model. "
                        "Either diagnose the current model with fea_check_model, ask the user with "
                        "ask_user if a structural decision is genuinely ambiguous, or solve and report."
                    ),
                })

            if tool_calls:
                tool_entries: list[dict[str, Any]] = []
                for tc_idx, tc in enumerate(tool_calls):
                    tool_entry: dict[str, Any] = {
                        "id": str(tc.get("id") or f"call_{iteration}_{tc_idx}"),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    if "extra_content" in tc:
                        tool_entry["extra_content"] = tc["extra_content"]
                    tool_entries.append(tool_entry)
                messages.append({
                    "role": "assistant",
                    "tool_calls": tool_entries,
                })

            # Process each tool call
            for tc_idx, tc in enumerate(tool_calls):
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_call_id = str(tc.get("id") or f"call_{iteration}_{tc_idx}")

                yield ("fea_tool_call", {"tool": tool_name, "args": tool_args})
                t0 = time.time()

                # Execute tool
                try:
                    commands, result_text = execute_fea_tool(
                        tool_name, tool_args, self.session.model_state, self.settings.project_root,
                    )
                    status = "ok"
                except Exception as exc:
                    logger.exception("fea_tool_execution_failed", extra={"tool": tool_name})
                    commands, result_text = [], f"TOOL ERROR: {exc}"
                    status = "error"
                if result_text.startswith("TOOL ERROR:"):
                    status = "error"

                # Check for solve request
                if result_text.startswith("__SOLVE_REQUEST__"):
                    lc_id = result_text.split("|")[1] if "|" in result_text else "LC1"
                    self.session.solve_attempts += 1

                    # Emit commands to build model
                    if commands:
                        yield ("fea_command", {"commands": commands})

                    # Emit solve request
                    yield ("fea_solve_request", {
                        "session_id": self.session.session_id,
                        "load_case_id": lc_id,
                    })

                    # Wait for results from frontend
                    yield ("fea_thinking", {"content": "Waiting for solver results..."})
                    try:
                        results = await asyncio.wait_for(
                            self.session.result_queue.get(),
                            timeout=120.0,
                        )
                        # Check if solver returned an error
                        if isinstance(results, dict) and results.get("error"):
                            error_msg = results.get("errorMessage", "Unknown solver error")
                            self.session.model_state.solved = False
                            yield ("fea_thinking", {"content": f"Solver error: {error_msg}. Diagnosing..."})
                            status = "error"
                            result_text = (
                                f"SOLVER ERROR: {error_msg}\n"
                                "Diagnose the issue. Common causes: insufficient restraints, "
                                "disconnected elements, zero-length elements, or missing sections/materials.\n"
                                "You can modify the model and re-solve. Use fea_check_model to diagnose, "
                                "or fea_clear to rebuild from scratch."
                            )
                            if self.session.solve_attempts >= 3:
                                result_text += (
                                    "\nMultiple solve failures detected. Consider using ask_user "
                                    "to get clarification from the user about the structural system."
                                )
                        else:
                            result_text = "Solver completed. Results are available. Use fea_get_results to query them."
                            solved_this_turn = True
                    except asyncio.TimeoutError:
                        status = "error"
                        result_text = (
                            "SOLVER ERROR: Solver timed out after 120 seconds. "
                            "Diagnose the model, reduce unnecessary retries, and only continue once "
                            "the current model state is understood."
                        )
                        yield ("fea_thinking", {"content": result_text})

                # Check for ask_user request
                elif result_text.startswith("__ASK_USER__"):
                    query_payload = json.loads(result_text.split("|", 1)[1])
                    yield ("fea_user_query", {
                        "session_id": self.session.session_id,
                        **query_payload,
                    })
                    yield ("fea_thinking", {"content": "Waiting for your input..."})
                    answer = await self.session.answer_queue.get()
                    result_text = json.dumps({
                        "status": "answered",
                        "answer": answer,
                    })

                else:
                    # Emit model-building commands
                    if commands:
                        yield ("fea_command", {"commands": commands})

                    # Emit view commands separately
                    for cmd in commands:
                        if cmd.get("action") in ("show_deformed", "show_moment_diagram", "show_shear_diagram",
                                                   "show_axial_diagram", "fit_view", "set_view", "hide_results"):
                            yield ("fea_view_command", cmd)

                elapsed_ms = int((time.time() - t0) * 1000)
                summary = self._summarize_tool_result(result_text, tool_name, elapsed_ms)
                yield ("fea_tool_result", {
                    "tool": tool_name,
                    "result": result_text,
                    "status": status,
                    "summary": summary,
                })
                if tool_name == "fea_get_results" and status == "ok":
                    query = get_result_query_coverage_key(tool_args.get("query"))
                    if query:
                        result_queries_requested.add(query)
                if tool_name in _SEMANTIC_PATCH_TOOLS and status == "ok":
                    semantic_model_active = isinstance(self.session.model_state.semantic_model, dict)

                if tool_name == "todo_write" and status == "ok":
                    new_steps = tool_args.get("todos", [])
                    if not plan_emitted:
                        yield ("plan", {"steps": new_steps})
                        plan_steps = new_steps
                        plan_emitted = True
                    else:
                        old_map = {s.get("id", ""): s.get("status", "pending") for s in plan_steps}
                        for step in new_steps:
                            step_id = step.get("id", "")
                            new_status = step.get("status", "pending")
                            if step_id and old_map.get(step_id) != new_status:
                                yield ("plan_update", {"step_id": step_id, "status": new_status})
                        plan_steps = new_steps

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": self._format_tool_result_for_context(
                        result_text,
                        tool_name,
                        total_tool_calls=total_tool_calls,
                    ),
                })
                if tool_name in {"ask_user", "fea_ask_user"} and status == "ok":
                    try:
                        answer_data = json.loads(result_text)
                    except (TypeError, json.JSONDecodeError):
                        answer_data = {}
                    answer = str(answer_data.get("answer", "") or "").strip()
                    if answer:
                        messages.append({
                            "role": "user",
                            "content": (
                                "[User's answer to your ask_user question]\n"
                                f"{answer}\n\n"
                                "Continue the same FEA analysis from the current model and plan. "
                                "Do not restart the model and do not invent defaults that the user did not provide."
                            ),
                        })
                elif status == "error":
                    messages.append({
                        "role": "user",
                        "content": self._build_tool_error_repair_prompt(tool_name, result_text),
                    })

            # ── Update loop detection state ─────────────────────────
            iteration_tools = [tc["name"] for tc in tool_calls]
            tool_history.extend(iteration_tools)
            if "fea_get_results" in iteration_tools:
                results_retrieved = True

            # Check for tool-calling loops (only before results are retrieved)
            if not results_retrieved:
                loop_msg = self._check_for_loop(tool_history)
                if loop_msg:
                    yield ("fea_thinking", {"content": "Adjusting approach..."})
                    messages.append({"role": "user", "content": loop_msg})

        # If we exhausted iterations
        yield ("fea_complete", {
            "summary": "Analysis session ended.",
            "assumptions": list(self.session.model_state.assumptions),
            "session_memory": self._build_session_memory("Analysis session ended."),
        })

    # ── LLM interaction ───────────────────────────────────────────

    def _call_llm(
        self,
        messages: list[dict],
        *,
        semantic_model_active: bool = False,
    ) -> str:
        """Call the LLM with messages and tool definitions.

        Uses generate_messages with the tool schemas. The LLM should return
        either a text response or tool calls in its response.
        """
        # Build the request with tools
        # Since we're using the OpenAI-compatible API, include tools in the request
        reasoning = self.settings.fea_analyst_reasoning_effort or None
        max_tok = self.settings.fea_analyst_max_tokens
        temp = self.settings.fea_analyst_temperature

        tools = self._tools_for_current_state(semantic_model_active=semantic_model_active)

        # Try function-calling format first
        try:
            raw = self.llm.generate_messages(
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                reasoning_effort=reasoning,
                tools=tools,
            )
            return raw
        except TypeError:
            # Fallback: LLM provider doesn't support tools parameter
            # Inject tool descriptions into system prompt and parse response manually
            tool_desc = self._tools_as_text(tools)
            augmented = self._sanitize_messages_for_text_mode(messages)
            if augmented and augmented[0]["role"] == "system":
                augmented[0] = {
                    "role": "system",
                    "content": augmented[0]["content"] + "\n\n" + tool_desc,
                }

            return self.llm.generate_messages(
                messages=augmented,
                temperature=temp,
                max_tokens=max_tok,
                reasoning_effort=reasoning,
            )

    @staticmethod
    def _sanitize_messages_for_text_mode(messages: list[dict]) -> list[dict]:
        """Convert tool_calls/tool messages to plain assistant/user messages.

        Some APIs (e.g. Gemini) don't accept OpenAI-style tool messages.
        Convert them to regular text so the LLM sees the context.
        Uses a contextual format that the LLM won't mimic as an action format.
        """
        sanitized = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "tool":
                # Convert tool result to user message
                tool_id = msg.get("tool_call_id", "")
                content = msg.get("content", "")
                sanitized.append({
                    "role": "user",
                    "content": f"[TOOL RESPONSE for {tool_id}]\n{content}",
                })
            elif msg.get("tool_calls"):
                # Convert assistant tool_calls to contextual description
                # Use a format the LLM is unlikely to mimic as a response
                calls = msg["tool_calls"]
                parts = []
                for c in calls:
                    fn = c.get("function", {})
                    parts.append(f"  - {fn.get('name', '?')}: {fn.get('arguments', '{}')}")
                sanitized.append({
                    "role": "assistant",
                    "content": "```json\n{\"tool_calls\": ["
                    + ", ".join(
                        '{{"name": "{}", "args": {}}}'.format(
                            c.get("function", {}).get("name", "?"),
                            c.get("function", {}).get("arguments", "{}"),
                        )
                        for c in calls
                    )
                    + "]}\n```",
                })
            else:
                sanitized.append(msg)
        return sanitized

    def _tools_as_text(self, tool_defs: list[dict[str, Any]] | None = None) -> str:
        """Format tool definitions as text for LLMs that don't support function calling."""
        defs = tool_defs or FEA_TOOLS
        lines = [
            "## Available Tools — CRITICAL INSTRUCTIONS",
            "You MUST call tools by responding with ONLY a JSON code block in this exact format:",
            '```json',
            '{"tool_calls": [{"name": "tool_name", "args": {...}}]}',
            '```',
            "Do NOT explain, do NOT plan. Just output the JSON tool call. "
            "You can call multiple tools in one response.",
            "",
        ]
        for tool in defs:
            fn = tool["function"]
            params = fn.get("parameters", {}).get("properties", {})
            param_desc = ", ".join(f"{k}: {v.get('description', '')}" for k, v in params.items())
            lines.append(f"- **{fn['name']}**({param_desc}): {fn['description']}")
        return "\n".join(lines)

    @staticmethod
    def _tools_for_current_state(*, semantic_model_active: bool) -> list[dict[str, Any]]:
        if not semantic_model_active:
            return FEA_TOOLS
        return [
            tool for tool in FEA_TOOLS
            if tool.get("function", {}).get("name") in _SEMANTIC_SESSION_ALLOWED_TOOLS
        ]

    @staticmethod
    def _summarize_tool_result(result_text: str, tool_name: str, elapsed_ms: int = 0) -> str:
        """Build a compact summary for the frontend tool card."""
        if result_text.startswith("TOOL ERROR:"):
            return result_text[:140]
        if result_text.startswith("SOLVER ERROR:"):
            return result_text[:140]
        try:
            data = json.loads(result_text)
        except (TypeError, json.JSONDecodeError):
            return f"OK ({elapsed_ms}ms)"
        if isinstance(data, dict):
            if "error" in data:
                return f"Error: {str(data['error'])[:100]}"
            if tool_name == "todo_write" and isinstance(data.get("plan"), list):
                return f"Plan updated ({len(data['plan'])} steps)"
            if tool_name == "fea_record_assumptions" and isinstance(data.get("assumptions"), list):
                return f"Recorded {len(data['assumptions'])} assumption(s)"
            if data.get("status") == "answered":
                return "User answered"
        return f"OK ({elapsed_ms}ms)"

    @staticmethod
    def _format_tool_result_for_context(
        result_text: str,
        tool_name: str,
        *,
        total_tool_calls: int,
    ) -> str:
        """Append a short behavioral reminder after each tool result."""
        reminder = (
            "\n[System reminder] Continue from the current model state. "
            "Use the exact tool schema. If a structural decision is genuinely ambiguous, "
            "use ask_user rather than inventing a hidden default."
        )
        remaining = max(0, MAX_REAL_TOOL_CALLS - total_tool_calls)
        if tool_name == "fea_get_results":
            reminder += "\n[System reminder] The model is solved. Summarize the results instead of calling more build tools."
        elif tool_name == "fea_check_model":
            reminder += "\n[System reminder] Fix any reported instability, connectivity, or assignment issues before solving."
        if remaining <= 3:
            reminder += f"\n[Budget: {remaining} real tool calls remaining.]"
        return result_text[:30_000] + reminder

    @staticmethod
    def _prune_messages(messages: list[dict], max_chars: int = 24000) -> list[dict]:
        """Prune message history to stay within a rough token budget.

        Keeps: system prompt (always), user's original query, and the most recent
        messages. Removes older tool call/result pairs from the middle.
        """
        total = sum(len(str(m.get("content", "") or "")) for m in messages)
        if total <= max_chars:
            return messages

        # Always keep first 2 messages (system + user query) and last 8
        if len(messages) <= 10:
            return messages

        keep_head = 2
        keep_tail = 8
        pruned = messages[:keep_head] + messages[-keep_tail:]

        # Ensure no orphaned tool messages at the start of the tail
        while pruned and pruned[keep_head].get("role") == "tool":
            keep_tail -= 1
            if keep_tail < 4:
                break
            pruned = messages[:keep_head] + messages[-keep_tail:]

        # Add a context summary of what was pruned
        n_pruned = len(messages) - len(pruned)
        if n_pruned > 0:
            pruned.insert(keep_head, {
                "role": "user",
                "content": f"[{n_pruned} earlier tool call/result messages pruned for context length. "
                "The model state reflects all previous commands. Continue from where you left off.]",
            })

        return pruned

    @staticmethod
    def _check_for_loop(tool_history: list[str]) -> str | None:
        """Detect repetitive tool-calling patterns and return a corrective message.

        Analyses the recent tool call history for signs the LLM is stuck in a loop
        (e.g., calling fea_add_loads over and over). Returns a corrective prompt
        to inject into the conversation, or None if no loop detected.
        """
        if len(tool_history) < 4:
            return None

        # Tools that are legitimately called multiple times
        _LOOP_EXEMPT = {"fea_check_model", "fea_get_results", "fea_set_view"}

        # ── Single-tool repetition: 3+ of the same in last 4 calls ──
        recent = tool_history[-4:]
        counts: dict[str, int] = {}
        for t in recent:
            counts[t] = counts.get(t, 0) + 1

        for tool, count in counts.items():
            if count >= 3 and tool not in _LOOP_EXEMPT:
                if tool == "fea_add_loads":
                    return (
                        "You have already defined loads — do NOT call fea_add_loads again. "
                        "The next steps are: fea_check_model → fea_solve → fea_get_results. "
                        "Call fea_check_model NOW."
                    )
                if tool == "fea_clear":
                    return (
                        "You have cleared the model multiple times. Something is fundamentally "
                        "wrong with your approach. Use ask_user to ask the user for "
                        "clarification about the structural system."
                    )
                if tool in ("fea_add_nodes", "fea_add_elements"):
                    return (
                        f"You are repeating {tool}. The geometry is already defined. "
                        "Proceed to: assign sections → assign material → set restraints → "
                        "add loads → check → solve."
                    )
                return (
                    f"You are calling {tool} repeatedly ({count} times in last 4 calls). "
                    "Move to the next step: nodes → elements → sections → materials → "
                    "restraints → loads → check → solve → results → summary."
                )

        # ── Repeating sequence: same 5-tool pattern twice ──
        if len(tool_history) >= 10:
            last_5 = tool_history[-5:]
            prev_5 = tool_history[-10:-5]
            if last_5 == prev_5:
                return (
                    "You are stuck in a repeating cycle of the same tool sequence. "
                    "STOP repeating and either: (1) call fea_check_model to diagnose "
                    "what's wrong, or (2) call ask_user to ask the user for help."
                )

        return None

    @staticmethod
    def _build_missing_results_prompt(missing_queries: list[str]) -> str:
        joined = ", ".join(missing_queries)
        return (
            "The model is solved, but your result set is incomplete. "
            f"Before you write the engineering summary, call fea_get_results for these missing queries: {joined}. "
            "Do not rebuild the model and do not output prose yet."
        )

    @staticmethod
    def _build_tool_error_repair_prompt(tool_name: str, result_text: str) -> str:
        return (
            f"The previous call to {tool_name} failed.\n"
            f"Error: {result_text}\n\n"
            "Correct the arguments using the exact tool schema and continue from the current model state. "
            "Do not restart the analysis, do not repeat already successful tools, and do not output prose."
        )

    @staticmethod
    def _parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                data = json.loads(raw_args)
            except json.JSONDecodeError:
                data = parse_json_loose(raw_args)
            if isinstance(data, dict):
                return data
        return {}

    def _extract_tool_calls(self, response: str | dict) -> list[dict]:
        """Extract tool calls from LLM response."""
        # If response is a dict (from function-calling API)
        if isinstance(response, dict):
            calls = response.get("tool_calls", [])
            parsed_calls: list[dict[str, Any]] = []
            for call in calls:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                name = function.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                parsed_call: dict[str, Any] = {
                    "name": name,
                    "args": self._parse_tool_arguments(function.get("arguments", {})),
                }
                call_id = call.get("id") if isinstance(call, dict) else None
                if isinstance(call_id, str) and call_id.strip():
                    parsed_call["id"] = call_id
                extra_content = call.get("extra_content") if isinstance(call, dict) else None
                if extra_content is not None:
                    parsed_call["extra_content"] = extra_content
                parsed_calls.append(parsed_call)
            return parsed_calls

        # If response is a string, try to parse tool calls from JSON
        if isinstance(response, str):
            # Look for JSON blocks with tool_calls
            text = response.strip()

            # Try parsing as JSON directly
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "tool_calls" in data:
                    return [{"name": c["name"], "args": c.get("args", c.get("arguments", {}))} for c in data["tool_calls"]]
            except (json.JSONDecodeError, KeyError):
                pass

            # Try extracting JSON from markdown code blocks
            import re
            json_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
            for block in json_blocks:
                try:
                    data = json.loads(block.strip())
                    if isinstance(data, dict) and "tool_calls" in data:
                        return [{"name": c["name"], "args": c.get("args", c.get("arguments", {}))} for c in data["tool_calls"]]
                except (json.JSONDecodeError, KeyError):
                    continue

            # Try parse_json_loose as last resort
            try:
                data = parse_json_loose(text)
                if isinstance(data, dict) and "tool_calls" in data:
                    return [{"name": c["name"], "args": c.get("args", c.get("arguments", {}))} for c in data["tool_calls"]]
            except Exception:
                pass

        return []

    @staticmethod
    def _message_role(message: Any) -> str:
        if hasattr(message, "role"):
            return str(getattr(message, "role", "") or "")
        if isinstance(message, dict):
            return str(message.get("role", "") or "")
        return ""

    @staticmethod
    def _message_content(message: Any) -> str:
        if hasattr(message, "content"):
            return str(getattr(message, "content", "") or "")
        if isinstance(message, dict):
            return str(message.get("content", "") or "")
        return ""

    @staticmethod
    def _message_response_payload(message: Any) -> dict[str, Any]:
        if hasattr(message, "response_payload"):
            payload = getattr(message, "response_payload")
        elif isinstance(message, dict):
            payload = message.get("response_payload")
            if payload is None:
                payload = message.get("responsePayload")
        else:
            payload = getattr(message, "responsePayload", None)
        return payload if isinstance(payload, dict) else {}

    def _extract_latest_fea_session_memory(self, history: list | None) -> dict[str, Any] | None:
        for message in reversed(history or []):
            if self._message_role(message) != "assistant":
                continue
            payload = self._message_response_payload(message)
            session_memory = payload.get("session_memory")
            if isinstance(session_memory, dict):
                fea_session = session_memory.get("fea_session")
                if isinstance(fea_session, dict):
                    extracted = deepcopy(session_memory)
                    answer_summary = str(extracted.get("answer_summary", "") or "").strip()
                    if not answer_summary:
                        answer_summary = str(payload.get("answer", "") or self._message_content(message)).strip()
                        if answer_summary:
                            extracted["answer_summary"] = answer_summary
                    return extracted
            fea_session = payload.get("fea_session")
            if isinstance(fea_session, dict):
                return {
                    "state": "final",
                    "answer_summary": str(payload.get("answer", "") or self._message_content(message)).strip(),
                    "fea_session": deepcopy(fea_session),
                }
        return None

    def _restore_from_session_memory(self, session_memory: dict[str, Any]) -> None:
        fea_session = session_memory.get("fea_session")
        if not isinstance(fea_session, dict):
            return

        authoring_state = fea_session.get("authoring_state")
        if isinstance(authoring_state, dict):
            self.session.model_state = FEAModelState.from_authoring_snapshot(authoring_state)

        results_snapshot = fea_session.get("results_snapshot")
        if isinstance(results_snapshot, dict) and not isinstance(self.session.model_state.results, dict):
            self.session.model_state.results = deepcopy(results_snapshot)
            self.session.model_state.solved = True

        semantic_model = fea_session.get("semantic_model")
        if isinstance(semantic_model, dict):
            self.session.model_state.semantic_model = deepcopy(semantic_model)

        if not self.session.model_state.assumptions and isinstance(session_memory.get("assumptions"), list):
            self.session.model_state.assumptions = [
                str(item).strip()
                for item in session_memory["assumptions"]
                if str(item).strip()
            ]
        if not self.session.model_state.plan and isinstance(session_memory.get("plan"), list):
            self.session.model_state.plan = deepcopy(session_memory["plan"])

    def _build_restored_session_context(self, session_memory: dict[str, Any]) -> str:
        fea_session = session_memory.get("fea_session")
        if not isinstance(fea_session, dict):
            return ""

        model_summary = fea_session.get("model_summary")
        if not isinstance(model_summary, dict):
            model_summary = self._build_model_summary()

        lines = [
            "<restored-fea-session>",
            "You are continuing an existing FEA authoring model from persisted session memory.",
            (
                "Restored model summary: "
                f"analysis_type={model_summary.get('analysis_type', self.session.model_state.analysis_type)}, "
                f"nodes={model_summary.get('node_count', len(self.session.model_state.nodes))}, "
                f"elements={model_summary.get('element_count', len(self.session.model_state.elements))}, "
                f"load_cases={', '.join(model_summary.get('load_case_ids', [])) or 'none'}, "
                f"solved={'yes' if model_summary.get('solved', self.session.model_state.solved) else 'no'}."
            ),
        ]

        semantic_model = self.session.model_state.semantic_model
        if isinstance(semantic_model, dict):
            geometry = semantic_model.get("geometry", {})
            member_families = semantic_model.get("member_families", {})
            lines.append(
                "Semantic frame model restored: "
                f"kind={semantic_model.get('kind', '?')}, dimension={semantic_model.get('dimension', '?')}, "
                f"spans_x={geometry.get('spans_x', [])}, spans_z={geometry.get('spans_z', [])}, "
                f"storey_heights={geometry.get('storey_heights', [])}, "
                f"columns={member_families.get('columns', {}).get('profile_name', '?')}, "
                f"beams_x={member_families.get('beams_x', {}).get('profile_name', '?')}, "
                f"beams_z={member_families.get('beams_z', {}).get('profile_name', '?')}."
            )

        assumptions = self.session.model_state.assumptions
        if assumptions:
            lines.append("Recorded assumptions:")
            for item in assumptions[:8]:
                lines.append(f"- {item}")

        answer_summary = str(session_memory.get("answer_summary", "") or "").strip()
        if answer_summary:
            lines.append(f"Latest engineering summary: {answer_summary[:1200]}")

        if self.session.model_state.plan:
            lines.append("Latest plan state:")
            for step in self.session.model_state.plan[:8]:
                if not isinstance(step, dict):
                    continue
                step_id = str(step.get("id", "") or "").strip()
                status = str(step.get("status", "pending") or "pending").strip()
                text = str(step.get("text", "") or "").strip()
                if step_id or text:
                    label = step_id or "step"
                    suffix = f" — {text}" if text else ""
                    lines.append(f"- {label}: {status}{suffix}")

        lines.extend([
            "Follow-up rules:",
            "- If the user asks about the current model, existing results, or diagrams, reuse the restored model and results. Do not rebuild.",
            "- If a semantic frame model is restored, prefer fea_query_model plus the semantic patch tools for geometry, supports, members, and loads.",
            "- If the user asks to modify geometry, sections, supports, materials, or loads, edit the restored model incrementally, then run fea_check_model and re-solve if the model changed.",
            "- Only call fea_clear if the user explicitly wants a new model or a full rebuild is clearly required.",
            "</restored-fea-session>",
        ])
        return "\n".join(lines)

    def _build_model_summary(self) -> dict[str, Any]:
        summary = {
            "analysis_type": self.session.model_state.analysis_type,
            "node_count": len(self.session.model_state.nodes),
            "element_count": len(self.session.model_state.elements),
            "load_case_ids": [str(item) for item in self.session.model_state.load_cases.keys()],
            "solved": bool(self.session.model_state.solved and isinstance(self.session.model_state.results, dict)),
        }
        semantic_model = self.session.model_state.semantic_model
        if isinstance(semantic_model, dict):
            geometry = semantic_model.get("geometry", {})
            summary.update({
                "semantic_kind": semantic_model.get("kind"),
                "semantic_dimension": semantic_model.get("dimension"),
                "bay_count_x": len(geometry.get("spans_x", [])) if isinstance(geometry.get("spans_x"), list) else 0,
                "bay_count_z": len(geometry.get("spans_z", [])) if isinstance(geometry.get("spans_z"), list) else 0,
                "storey_count": len(geometry.get("storey_heights", [])) if isinstance(geometry.get("storey_heights"), list) else 0,
            })
        return summary

    def _build_session_memory(self, summary_text: str) -> dict[str, Any]:
        summary = str(summary_text or "").strip()
        fea_session = {
            "version": 2,
            "authoring_state": self.session.model_state.to_authoring_snapshot(),
            "results_snapshot": deepcopy(self.session.model_state.results),
            "model_summary": self._build_model_summary(),
            "semantic_model": deepcopy(self.session.model_state.semantic_model),
        }
        return {
            "state": "final",
            "answer_summary": summary[:2000],
            "plan": deepcopy(self.session.model_state.plan),
            "assumptions": list(self.session.model_state.assumptions),
            "fea_session": fea_session,
        }

    @staticmethod
    def _extract_finish_reason(response: str | dict) -> str:
        if isinstance(response, dict):
            return str(response.get("finish_reason", "") or "").strip().lower()
        return ""

    def _extract_text(self, response: str | dict) -> str:
        """Extract text content from LLM response."""
        if isinstance(response, str):
            # Remove any JSON tool call blocks
            import re
            text = re.sub(r"```(?:json)?\s*\n?\{[^}]*\"tool_calls\"[^}]*\}.*?```", "", response, flags=re.DOTALL)
            return text.strip()
        if isinstance(response, dict):
            return response.get("content", "")
        return str(response)
