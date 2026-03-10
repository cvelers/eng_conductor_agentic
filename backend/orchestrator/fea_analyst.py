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
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from uuid import uuid4

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.orchestrator.fea_prompts import FEA_ANALYST_SYSTEM, FEA_TOOLS
from backend.orchestrator.fea_tools import FEAModelState, execute_fea_tool
from backend.utils.json_utils import parse_json_loose

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 25


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
      - ("fea_command", {"commands": [...]})
      - ("fea_solve_request", {"session_id": str, "load_case_id": str})
      - ("fea_user_query", {"session_id": str, "question": str, "options": [...], "context": str})
      - ("fea_view_command", {"action": str, ...})
      - ("fea_complete", {"summary": str})
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

        # Add history context if available
        if history:
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

        # ── Agentic tool-calling loop ─────────────────────────────
        for iteration in range(MAX_TOOL_ITERATIONS):
            # Prune context to fit within token budget
            messages = self._prune_messages(messages)

            # Call LLM with tool definitions
            try:
                response = self._call_llm(messages)
            except Exception as exc:
                logger.exception("fea_analyst_llm_failed")
                yield ("fea_thinking", {"content": f"LLM call failed: {exc}"})
                break

            # Parse response for tool calls
            tool_calls = self._extract_tool_calls(response)

            # ── Post-results guard: stop tool calls after results retrieved ──
            if results_retrieved and tool_calls:
                force_summary_attempts += 1
                if force_summary_attempts >= 2:
                    # LLM is ignoring guidance — force completion
                    text = self._extract_text(response)
                    yield ("fea_complete", {
                        "summary": text or "Analysis complete. Review the model and results in the viewer.",
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
                    yield ("fea_thinking", {"content": text})
                    # Add to message history
                    messages.append({"role": "assistant", "content": text})

                    # Check if this is a final answer (after results are available)
                    if self.session.model_state.solved or results_retrieved:
                        yield ("fea_complete", {"summary": text})
                        return
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

            # Process each tool call
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                yield ("fea_tool_call", {"tool": tool_name, "args": tool_args})

                # Execute tool
                commands, result_text = execute_fea_tool(
                    tool_name, tool_args, self.session.model_state, self.settings.project_root,
                )

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
                            result_text = (
                                f"SOLVER ERROR: {error_msg}\n"
                                "Diagnose the issue. Common causes: insufficient restraints, "
                                "disconnected elements, zero-length elements, or missing sections/materials.\n"
                                "You can modify the model and re-solve. Use fea_check_model to diagnose, "
                                "or fea_clear to rebuild from scratch."
                            )
                            if self.session.solve_attempts >= 3:
                                result_text += (
                                    "\nMultiple solve failures detected. Consider using fea_ask_user "
                                    "to get clarification from the user about the structural system."
                                )
                        else:
                            result_text = "Solver completed. Results are available. Use fea_get_results to query them."
                    except asyncio.TimeoutError:
                        result_text = "Solver timed out after 120 seconds."
                        yield ("fea_thinking", {"content": result_text})

                # Check for ask_user request
                elif result_text.startswith("__ASK_USER__"):
                    query_payload = json.loads(result_text.split("|", 1)[1])
                    yield ("fea_user_query", {
                        "session_id": self.session.session_id,
                        **query_payload,
                    })
                    yield ("fea_thinking", {"content": "Waiting for your input..."})
                    try:
                        answer = await asyncio.wait_for(
                            self.session.answer_queue.get(),
                            timeout=300.0,
                        )
                        result_text = f"User answered: {answer}"
                    except asyncio.TimeoutError:
                        result_text = "User did not respond within 5 minutes. Proceed with reasonable engineering defaults and explain your assumptions."
                        yield ("fea_thinking", {"content": "No response received, proceeding with defaults..."})

                else:
                    # Emit model-building commands
                    if commands:
                        yield ("fea_command", {"commands": commands})

                    # Emit view commands separately
                    for cmd in commands:
                        if cmd.get("action") in ("show_deformed", "show_moment_diagram", "show_shear_diagram",
                                                   "show_axial_diagram", "fit_view", "set_view", "hide_results"):
                            yield ("fea_view_command", cmd)

                # Add tool call + result to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": f"call_{iteration}_{tc['name']}", "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{iteration}_{tc['name']}",
                    "content": result_text,
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
        yield ("fea_complete", {"summary": "Analysis session ended."})

    # ── LLM interaction ───────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        """Call the LLM with messages and tool definitions.

        Uses generate_messages with the tool schemas. The LLM should return
        either a text response or tool calls in its response.
        """
        # Build the request with tools
        # Since we're using the OpenAI-compatible API, include tools in the request
        reasoning = self.settings.fea_analyst_reasoning_effort or None
        max_tok = self.settings.fea_analyst_max_tokens
        temp = self.settings.fea_analyst_temperature

        # Try function-calling format first
        try:
            raw = self.llm.generate_messages(
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                reasoning_effort=reasoning,
                tools=FEA_TOOLS,
            )
            return raw
        except TypeError:
            # Fallback: LLM provider doesn't support tools parameter
            # Inject tool descriptions into system prompt and parse response manually
            tool_desc = self._tools_as_text()
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

    def _tools_as_text(self) -> str:
        """Format tool definitions as text for LLMs that don't support function calling."""
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
        for tool in FEA_TOOLS:
            fn = tool["function"]
            params = fn.get("parameters", {}).get("properties", {})
            param_desc = ", ".join(f"{k}: {v.get('description', '')}" for k, v in params.items())
            lines.append(f"- **{fn['name']}**({param_desc}): {fn['description']}")
        return "\n".join(lines)

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
                        "wrong with your approach. Use fea_ask_user to ask the user for "
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
                    "what's wrong, or (2) call fea_ask_user to ask the user for help."
                )

        return None

    def _extract_tool_calls(self, response: str | dict) -> list[dict]:
        """Extract tool calls from LLM response."""
        # If response is a dict (from function-calling API)
        if isinstance(response, dict):
            calls = response.get("tool_calls", [])
            return [
                {"name": c["function"]["name"], "args": json.loads(c["function"]["arguments"])}
                for c in calls
            ]

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

            # Fallback: parse "Tool calls: name(args)" format (LLM mimicking sanitized messages)
            tc_match = re.match(r"^Tool calls?:\s*(.+)", text, re.DOTALL)
            if tc_match:
                calls = []
                for m in re.finditer(r"(\w+)\((\{.*?\})\)", tc_match.group(1)):
                    try:
                        args = json.loads(m.group(2))
                        calls.append({"name": m.group(1), "args": args})
                    except json.JSONDecodeError:
                        pass
                if calls:
                    return calls

        return []

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
