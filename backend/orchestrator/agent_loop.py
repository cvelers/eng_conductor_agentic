"""Agentic task-decomposition loop.

Replaces the rigid pipeline in ``engine.py`` with a decompose-then-execute
architecture.  Complex queries are broken into individual tasks, each
executed with its own focused search → calculate → compose cycle.

The key insight: each compose call gets a **clean, isolated context** with
only the relevant clauses and tool outputs — maximising LLM intelligence
per sub-task instead of diluting it across everything at once.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from backend.config import Settings
from backend.orchestrator.core import (
    CentralIntelligenceOrchestrator,
    PlanResult,
    _flatten_tool_outputs,
)
from backend.schemas import (
    Attachment,
    ChatResponse,
    Citation,
    RetrievalTraceStep,
    ToolTraceStep,
)
from backend.utils.citations import build_citation_address
from backend.utils.json_utils import parse_json_loose

logger = logging.getLogger(__name__)


MAX_TOOL_RETRIES = 2


@dataclass
class _TaskSpec:
    """One decomposed sub-task."""

    summary: str
    query: str
    search_query: str
    needs_search: bool = True
    tools: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)


# ── Compose system prompt — same engineering formatting as engine.py ──

_COMPOSE_SYSTEM = (
    "You are a senior structural engineer answering a colleague's question at a desk review.\n\n"
    "FORMATTING — CRITICAL:\n"
    "- All formulas and math expressions MUST be wrapped in dollar signs: $F_{v,Rd}$, $\\frac{a}{b}$, $\\gamma_{M2}$.\n"
    "- Inline math uses single dollars: $F_{v,Rd} = 94.08$ kN.\n"
    "- Never write bare LaTeX commands without dollar-sign delimiters.\n"
    "- Bold **key numerical results**: e.g. **$F_{v,Rd}$ = 94.08 kN**.\n"
    "- Ensure markdown emphasis is balanced (never leave dangling '**').\n\n"
    "CONTENT:\n"
    "1. First sentence MUST directly answer the colleague's question. Start with a capital letter.\n"
    "   - If they asked for a calculation: state what you calculated, for what parameters, and the result.\n"
    "   - If they asked a conceptual question: lead with a direct answer.\n"
    "   - If default values were assumed, name them in the first sentence.\n"
    "2. Explain the method, formula, and key parameters in detail. Show the formula with values substituted in where available.\n"
    "3. Mention governing EC clauses ONLY when explicit clause evidence is provided below.\n"
    "4. Use ONLY the provided evidence. Never invent values.\n"
    "5. If there is no EC clause evidence, do not reference EN 1993 or clause numbers.\n"
    "6. Write as much detail as the evidence supports.\n\n"
    "STRUCTURE:\n"
    "- Use markdown structure: paragraphs, bullet points, and line breaks.\n"
    "- Break different aspects into separate paragraphs.\n"
    "- Display key formulas on their own line using $$...$$ display math.\n\n"
    "CLAUSE REFERENCES:\n"
    "At the very end of your answer, on a new line, output a JSON array of clause IDs\n"
    "you actually used to produce the answer, formatted as:\n"
    "<!--USED_CLAUSES:[\"5.5.2\", \"6.2.3\", \"Table 3.1\"]-->\n"
    "Include ONLY clauses whose content directly informed your answer. If none, output:\n"
    "<!--USED_CLAUSES:[]-->"
)

_DECOMPOSE_SYSTEM = (
    "You decompose user engineering queries into individual, independent sub-tasks.\n"
    "Each sub-task should be self-contained and addressable with a focused search + calculation.\n\n"
    "Return ONLY a JSON array of task objects. Each task:\n"
    "{\n"
    '  "summary": "Short description (shown as plan step)",\n'
    '  "query": "Focused technical query for this task",\n'
    '  "search_query": "Search terms for finding relevant EC3 clauses",\n'
    '  "needs_search": true/false,\n'
    '  "tools": ["tool_name_1", "tool_name_2"],\n'
    '  "inputs": {"param1": value1, ...}\n'
    "}\n\n"
    "Rules:\n"
    "- If the query asks for only ONE thing, return a single-element array.\n"
    "- Only split when there are genuinely separate engineering questions.\n"
    "- Order tasks logically (dependencies first).\n"
    "- Only include tools that are directly relevant from the available tools list.\n"
    "- In 'inputs', only include values the user explicitly states in their query.\n"
    "- NEVER guess or invent values for parameters that an earlier task's tool\n"
    "  will compute (e.g. section_class from section_classification, fy_mpa from\n"
    "  steel_grade_properties).  Omit those parameters entirely — the orchestrator\n"
    "  will resolve them automatically from the session context.\n"
    "- For Eurocode calculations: use 'math_calculator' as the tool. The orchestrator\n"
    "  will extract equations from the retrieved clause text and build the input\n"
    "  automatically. You do NOT need to specify the equations — just select\n"
    "  math_calculator and ensure needs_search is true so relevant clauses are retrieved.\n"
    "- Use 'section_properties' to look up geometric properties of standard sections.\n"
    "- Use 'steel_grade_properties' for material property lookups.\n"
    "- Use 'section_classification_ec3' for cross-section classification.\n"
    "- Use 'unit_converter' when unit conversion is needed.\n"
    "- Return valid JSON only, no markdown fences, no explanation."
)


class AgentLoop:
    """Agentic task-decomposition orchestrator.

    Wraps ``CentralIntelligenceOrchestrator`` via composition to reuse
    intent classification, tool execution, source collection, formatting,
    etc.  Only the top-level ``run_stream`` is replaced.
    """

    def __init__(
        self,
        *,
        orchestrator: CentralIntelligenceOrchestrator,
        settings: Settings,
    ) -> None:
        self.cio = orchestrator
        self.settings = settings
        self._current_tasks: list[_TaskSpec] = []
        self._original_query: str = ""

    # ── Public API ──────────────────────────────────────────────────

    def run(
        self,
        query: str,
        *,
        history: list | None = None,
        thinking_mode: str = "thinking",
        attachments: list[Attachment] | None = None,
        is_edit: bool = False,
    ) -> ChatResponse:
        final: ChatResponse | None = None
        for _etype, payload in self.run_stream(
            query,
            history=history,
            thinking_mode=thinking_mode,
            attachments=attachments,
            is_edit=is_edit,
        ):
            if isinstance(payload, ChatResponse):
                final = payload
        if final is None:
            raise RuntimeError("AgentLoop did not produce a final response.")
        return final

    def run_stream(
        self,
        raw_query: str,
        *,
        history: list | None = None,
        thinking_mode: str = "thinking",
        attachments: list[Attachment] | None = None,
        is_edit: bool = False,
    ) -> Iterator[tuple[str, Any]]:
        attachments = attachments or []
        query = (
            raw_query
            if is_edit
            else self.cio._resolve_followup(raw_query, history or [])
        )
        selected_mode = self.cio._normalize_thinking_mode(thinking_mode)

        # ── Phase 0: Intent classification ──────────────────────────
        classification = self.cio._classify_intent(query, attachments)
        intent = classification["intent"]
        logger.info(
            "agent_intent_classified",
            extra={"intent": intent, "query_preview": query[:80]},
        )

        if intent in ("decline", "greeting", "answer"):
            # Direct-response path — forward all events (machine + response)
            yield from self.cio._handle_direct_response(query, attachments, intent)
            return

        if intent == "fea":
            # FEA analyst sub-agent — delegate entirely
            yield from self._handle_fea(query, attachments, history or [], selected_mode)
            return

        # ── Flow graph: intake ────────────────────────────────────────
        yield ("machine", {"node": "intake", "status": "active", "title": "Query Intake", "detail": "Analyzing your question..."})
        yield ("machine", {"node": "intake", "status": "done", "title": "Query Intake", "detail": "Planning tasks..."})

        # ── Phase 1: Task decomposition ─────────────────────────────
        yield ("thinking", {"content": "Analyzing your question..."})

        tasks = self._decompose(query, selected_mode)
        self._current_tasks = tasks  # Store for plan context in LLM resolution
        self._original_query = query  # Preserve for all downstream resolution
        logger.info(
            "agent_tasks_decomposed",
            extra={"task_count": len(tasks), "summaries": [t.summary for t in tasks]},
        )

        yield (
            "plan",
            {
                "steps": [
                    {"id": i, "text": t.summary, "status": "pending"}
                    for i, t in enumerate(tasks)
                ]
            },
        )

        # ── Phase 2: Execute each task sequentially ─────────────────
        all_sources: list[Citation] = []
        all_tool_trace: list[ToolTraceStep] = []
        all_retrieval_trace: list[RetrievalTraceStep] = []
        all_retrieved: list = []
        all_tool_outputs: dict[str, dict[str, Any]] = {}
        all_llm_clause_ids: set[str] = set()
        answer_parts: list[str] = []
        all_user_inputs: dict[str, Any] = {}
        all_assumed_inputs: dict[str, Any] = {}
        all_assumptions: list[str] = []

        for i, task in enumerate(tasks):
            yield ("plan_update", {"step_id": i, "status": "in_progress"})

            # 2a. SEARCH — focused retrieval for this task only
            task_clauses = []
            if task.needs_search:
                yield ("machine", {"node": "retrieval", "status": "active", "title": "Retrieving", "detail": f"Searching: {task.search_query[:60]}"})
                yield (
                    "tool_start",
                    {"tool": "search", "args": {"query": task.search_query}},
                )
                try:
                    results, trace = self.cio.retriever.retrieve(
                        task.search_query,
                        top_k=self._top_k_for_mode(selected_mode),
                    )
                    task_clauses = results
                    all_retrieved.extend(results)
                    all_retrieval_trace.extend(
                        [
                            RetrievalTraceStep(
                                iteration=s.get("iteration", 0),
                                query=str(s.get("query", task.search_query)),
                                top_clause_ids=s.get("top_clause_ids", []),
                            )
                            for s in trace
                            if isinstance(s, dict)
                        ]
                    )
                    clause_summaries = [
                        {
                            "clause_id": r.clause.clause_id,
                            "title": r.clause.clause_title,
                        }
                        for r in results[:6]
                    ]
                    top_meta = [{"doc_id": r.clause.doc_id, "clause_id": r.clause.clause_id} for r in results[:6]]
                    yield ("machine", {"node": "retrieval", "status": "done", "title": "Retrieved", "detail": f"Found {len(results)} clauses", "meta": {"top": top_meta}})
                    yield (
                        "tool_result",
                        {
                            "tool": "search",
                            "status": "ok",
                            "summary": f"Found {len(results)} relevant clauses",
                            "result": {"clauses": clause_summaries},
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "agent_search_failed",
                        extra={"error": str(exc), "query": task.search_query},
                    )
                    yield ("machine", {"node": "retrieval", "status": "error", "title": "Retrieval", "detail": f"Search failed: {exc}"})
                    yield (
                        "tool_result",
                        {
                            "tool": "search",
                            "status": "error",
                            "summary": f"Search failed: {exc}",
                            "result": {},
                        },
                    )

            # 2b. CALCULATE — tools needed for this task only
            task_tool_outputs: dict[str, dict[str, Any]] = {}
            tool_failures: list[str] = []
            if task.tools:
                ordered = self.cio._normalize_tool_chain(
                    task.tools,
                    already_run=set(all_tool_outputs.keys()),
                )
                yield ("machine", {"node": "tools", "status": "active", "title": "Tools", "detail": "Executing tool chain."})
                for tool_name in ordered:
                    # Equation extraction: for math_calculator, extract equations
                    # from retrieved clause text instead of normal input resolution
                    if tool_name == "math_calculator":
                        inputs = self._extract_equations(
                            task, task_clauses, all_tool_outputs,
                        )
                    else:
                        inputs = self._build_task_tool_inputs(
                            tool_name, task, all_tool_outputs
                        )
                    success = False
                    last_error = ""
                    error_log: list[str] = []

                    for attempt in range(1 + MAX_TOOL_RETRIES):
                        yield (
                            "tool_start",
                            {
                                "tool": tool_name,
                                "args": inputs,
                            },
                        )
                        try:
                            payload = self.cio.tool_runner.run(tool_name, inputs)
                            result = payload.get("result", {})
                            task_tool_outputs[tool_name] = result
                            all_tool_outputs[tool_name] = result

                            step = ToolTraceStep(
                                tool_name=tool_name,
                                status="ok",
                                inputs=inputs,
                                outputs=result.get("outputs", result),
                            )
                            all_tool_trace.append(step)

                            # Compact summary for the UI
                            outputs = result.get("outputs", {})
                            summary_parts = []
                            for k, v in outputs.items():
                                if isinstance(v, (int, float)):
                                    summary_parts.append(
                                        f"{k.replace('_', ' ')} = {v}"
                                    )
                            summary = "; ".join(summary_parts[:3]) or "Complete"

                            yield ("machine", {"node": "tools", "status": "active", "title": "Tools", "detail": f"Tool {tool_name} completed.", "meta": {"tool": tool_name}})
                            yield (
                                "tool_result",
                                {
                                    "tool": tool_name,
                                    "status": "ok",
                                    "summary": summary,
                                    "result": result,
                                },
                            )
                            success = True
                            break
                        except Exception as exc:
                            last_error = str(exc)
                            error_log.append(f"Attempt {attempt + 1}: inputs={json.dumps(inputs, default=str)}, error={last_error}")
                            yield (
                                "tool_result",
                                {
                                    "tool": tool_name,
                                    "status": "error",
                                    "summary": f"Failed (attempt {attempt + 1}): {exc}",
                                    "result": {},
                                },
                            )
                            if attempt == 0:
                                if tool_name == "math_calculator":
                                    # Re-run equation extraction with error context
                                    retried = self._extract_equations(
                                        task, task_clauses, all_tool_outputs,
                                        error_hint=last_error,
                                    )
                                    if retried and retried != inputs:
                                        inputs = retried
                                        yield ("machine", {"node": "tools", "status": "active", "title": "Tools", "detail": "Re-extracting equations..."})
                                    else:
                                        break
                                else:
                                    fixed = self._fix_tool_inputs(
                                        tool_name, inputs, last_error, task
                                    )
                                    if fixed and fixed != inputs:
                                        inputs = fixed
                                        yield ("machine", {"node": "tools", "status": "active", "title": "Tools", "detail": f"Retrying {tool_name} with corrected inputs..."})
                                    else:
                                        break
                            elif attempt == 1:
                                # Attempt 2 retry: UPSTREAM — re-resolve all
                                # params from full context + error log
                                upstream = self._upstream_resolve_inputs(
                                    tool_name, task, all_tool_outputs,
                                    error_log,
                                )
                                if upstream and upstream != inputs:
                                    inputs = upstream
                                    yield ("machine", {"node": "tools", "status": "active", "title": "Tools", "detail": f"Upstream re-resolve for {tool_name}..."})
                                else:
                                    break

                    if not success:
                        step = ToolTraceStep(
                            tool_name=tool_name,
                            status="error",
                            inputs=inputs,
                            error=last_error,
                        )
                        all_tool_trace.append(step)
                        tool_failures.append(f"{tool_name}: {last_error}")
                        yield ("machine", {"node": "tools", "status": "error", "title": "Tools", "detail": f"Tool {tool_name} failed after retries.", "meta": {"tool": tool_name}})

                yield ("machine", {"node": "tools", "status": "done", "title": "Tools", "detail": "Tool execution finished."})

            # Track inputs/assumptions from task
            if task.inputs:
                all_user_inputs.update(task.inputs)

            # 2c. COMPOSE — fresh LLM call with ONLY this task's context
            yield ("machine", {"node": "compose", "status": "active", "title": "Composing Answer", "detail": f"Composing: {task.summary[:60]}"})
            yield (
                "thinking",
                {"content": f"Composing answer for: {task.summary}"},
            )

            raw_partial = self._compose_task_answer(
                task=task,
                original_query=query,
                retrieved=task_clauses,
                tool_outputs=task_tool_outputs,
                tool_failures=tool_failures,
                thinking_mode=selected_mode,
            )

            # Extract LLM-declared clause references from the narrative
            partial, llm_clause_ids = _extract_used_clauses(raw_partial)

            # Stream the partial answer in natural chunks
            for chunk in _chunk_naturally(partial):
                yield ("delta", {"content": chunk})

            answer_parts.append(partial)
            all_llm_clause_ids.update(llm_clause_ids)

            # Collect sources for this task
            task_sources = self.cio._collect_sources(
                task_clauses, task_tool_outputs
            )
            all_sources.extend(task_sources)

            yield ("machine", {"node": "compose", "status": "done", "title": "Composing Answer", "detail": "Composition complete."})
            yield ("plan_update", {"step_id": i, "status": "done"})

        # ── Phase 3: Finalize ───────────────────────────────────────
        yield ("machine", {"node": "output", "status": "active", "title": "Streaming", "detail": "Building final response..."})

        full_answer = "\n\n---\n\n".join(answer_parts) if len(answer_parts) > 1 else (answer_parts[0] if answer_parts else "")

        # Format answer through the response formatter
        full_answer = self.cio.response_formatter.format_markdown(full_answer)

        # Append tool detail tables, assumptions, and references
        appendix = self._build_answer_appendix(
            tool_outputs=all_tool_outputs,
            tool_trace=all_tool_trace,
            assumptions=all_assumptions,
            sources=all_sources,
            retrieved=all_retrieved,
            narrative=full_answer,
            llm_clause_ids=all_llm_clause_ids,
        )
        if appendix:
            full_answer = full_answer + "\n\n" + appendix

        # Deduplicate sources
        seen_addresses: set[str] = set()
        unique_sources: list[Citation] = []
        for s in all_sources:
            if s.citation_address not in seen_addresses:
                seen_addresses.add(s.citation_address)
                unique_sources.append(s)

        what_i_used = self._build_what_i_used(tasks, all_retrieval_trace, all_tool_trace)

        response = ChatResponse(
            answer=full_answer,
            supported=bool(unique_sources or all_tool_trace),
            user_inputs=all_user_inputs,
            assumed_inputs=all_assumed_inputs,
            assumptions=all_assumptions,
            sources=unique_sources,
            tool_trace=all_tool_trace,
            retrieval_trace=all_retrieval_trace,
            what_i_used=what_i_used,
        )
        yield ("machine", {"node": "output", "status": "done", "title": "Streaming", "detail": "Complete."})
        yield ("response", response)

    # ── Answer appendix ─────────────────────────────────────────────

    def _build_answer_appendix(
        self,
        *,
        tool_outputs: dict[str, dict[str, Any]],
        tool_trace: list[ToolTraceStep],
        assumptions: list[str],
        sources: list[Citation],
        retrieved: list,
        narrative: str,
        llm_clause_ids: set[str] | None = None,
    ) -> str:
        """Build the structured appendix (tool tables, assumptions, references).

        When *llm_clause_ids* is provided (from the compose LLM), those are
        used to filter references instead of programmatic regex matching.
        """
        lines: list[str] = []

        # ── Tool detail tables ──
        if tool_outputs:
            for step in tool_trace:
                if step.status != "ok":
                    continue
                payload = tool_outputs.get(step.tool_name)
                if not payload:
                    continue
                outputs = payload.get("outputs", {})
                inputs_used = payload.get("inputs_used", {})
                if not outputs:
                    continue

                pretty = self.cio._pretty_tool_name(step.tool_name)
                lines.append(f'<details class="tool-result">')
                lines.append(f"<summary><strong>{pretty}</strong> detailed results</summary>")
                lines.append("")

                if inputs_used:
                    lines.append('<table class="tool-io-table">')
                    lines.append('<thead><tr><th scope="col">Parameter</th><th scope="col">Value</th></tr></thead>')
                    lines.append("<tbody>")
                    for k, v in inputs_used.items():
                        key_cell = self.cio.response_formatter.pretty_key(k)
                        val_cell = self.cio.response_formatter.format_value(k, v)
                        lines.append(f"<tr><td>{key_cell}</td><td>{val_cell}</td></tr>")
                    lines.append("</tbody></table>")
                    lines.append("")

                lines.append('<table class="tool-io-table tool-output">')
                lines.append('<thead><tr><th scope="col">Output</th><th scope="col">Value</th></tr></thead>')
                lines.append("<tbody>")
                for k, v in outputs.items():
                    if isinstance(v, (int, float, str, bool)):
                        key_cell = self.cio.response_formatter.pretty_key(k)
                        val_cell = self.cio.response_formatter.format_value(k, v)
                        lines.append(f"<tr><td>{key_cell}</td><td><strong>{val_cell}</strong></td></tr>")
                lines.append("</tbody></table>")
                lines.append("")

                notes = payload.get("notes", [])
                if notes:
                    lines.append("**Notes**")
                    lines.append("")
                    for n in notes:
                        lines.append(f"- {n}")
                    lines.append("")

                lines.append("</details>")
                lines.append("")

        # ── Assumptions ──
        if assumptions:
            lines.append('<details class="assumptions">')
            lines.append("<summary>Assumptions used</summary>")
            lines.append("")
            for a in assumptions:
                lines.append(f"- {a}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # ── Tool errors ──
        if any(s.status == "error" for s in tool_trace):
            lines.append("\n**Tool Errors:**\n")
            for s in tool_trace:
                if s.status == "error":
                    lines.append(f"- {s.tool_name}: {s.error}")

        # ── References ──
        if sources:
            # Primary: use LLM-declared clause IDs from the compose step
            if llm_clause_ids:
                norm_llm = {self.cio._normalize_clause_id(cid) for cid in llm_clause_ids}
                filtered = [
                    s for s in sources
                    if self.cio._normalize_clause_id(s.clause_id) in norm_llm
                ]
            else:
                # Fallback: programmatic filter on narrative + tool refs
                try:
                    filtered = self.cio._select_relevant_sources(
                        narrative=narrative, sources=sources,
                        retrieved=retrieved, tool_outputs=tool_outputs,
                    )
                except Exception:
                    filtered = sources
            if not filtered:
                filtered = sources
            relevant = [
                s for s in filtered
                if s.clause_id and s.clause_id != "0"
                and s.clause_title and s.clause_title != "text"
            ]
            if relevant:
                lines.append("\n---\n")
                lines.append("**References:**")
                lines.append("")
                seen_refs: set[str] = set()
                ref_idx = 0
                for s in relevant:
                    ref_key = self.cio._normalize_clause_id(s.clause_id)
                    if ref_key in seen_refs:
                        continue
                    seen_refs.add(ref_key)

                    clause_record = self.cio._lookup_clause(s.doc_id, s.clause_id)
                    ref_idx += 1
                    standard = clause_record.standard if clause_record else self.cio._source_standard_label(s.doc_id)
                    locator = self.cio._format_reference_locator(s.doc_id, s.clause_id)
                    label = f"{standard}, {locator} — {s.clause_title}" if locator else f"{standard} — {s.clause_title}"

                    if clause_record and clause_record.text.strip():
                        text_preview = self.cio._format_clause_text_for_display(clause_record.text)
                        lines.append(f'<details class="ref-clause">')
                        lines.append(f"<summary><strong>{ref_idx}.</strong> {label}</summary>")
                        lines.append("")
                        lines.append(f'<div class="clause-text">')
                        lines.append(text_preview)
                        lines.append("</div>")
                        lines.append("</details>")
                        lines.append("")
                    else:
                        lines.append(f"{ref_idx}. {label}")
                        lines.append("")

        result = "\n".join(lines).strip()
        return self.cio.response_formatter.format_markdown(result) if result else ""

    # ── FEA delegation ─────────────────────────────────────────────

    def _handle_fea(
        self,
        query: str,
        attachments: list[Attachment],
        history: list,
        thinking_mode: str,
    ) -> Iterator[tuple[str, Any]]:
        """Delegate to the FEA analyst sub-agent.

        Yields all FEA events which the streaming endpoint forwards to the
        frontend.  The fea_analyst module handles the agentic tool loop and
        the solve/resume cycle is managed by the async streaming layer in
        app.py.
        """
        from backend.llm.factory import get_fea_analyst_provider
        from backend.orchestrator.fea_analyst import FEAAnalystLoop

        # Flow graph: show FEA Analyst node activating
        yield ("machine", {
            "node": "intake", "status": "active",
            "title": "Query Intake", "detail": "Routing to FEA Analyst...",
        })
        yield ("machine", {
            "node": "intake", "status": "done",
            "title": "Query Intake", "detail": "FEA analysis requested.",
        })
        yield ("machine", {
            "node": "fea_analyst", "status": "active",
            "title": "FEA Analyst", "detail": "Initializing FEA analyst...",
        })

        fea_llm = get_fea_analyst_provider(self.settings)
        analyst = FEAAnalystLoop(llm=fea_llm, settings=self.settings)

        # Store session for result callback
        yield ("fea_session_created", {"session_id": analyst.session_id})

        # The FEA analyst is an async generator. We yield a special marker
        # so the streaming layer in app.py can switch to async iteration.
        yield ("fea_delegate", {
            "analyst": analyst,
            "query": query,
            "history": history,
        })

    # ── Task decomposition ──────────────────────────────────────────

    def _decompose(self, query: str, thinking_mode: str) -> list[_TaskSpec]:
        """Break query into focused sub-tasks.

        All thinking modes use LLM-based decomposition for tool selection.
        Standard mode instructs the LLM to return a single task.
        """
        if self.cio.orchestrator_llm.available:
            try:
                return self._llm_decompose(query, thinking_mode)
            except Exception as exc:
                logger.warning(
                    "agent_decompose_failed",
                    extra={"error": str(exc)},
                )

        # Fallback when LLM is unavailable — no tool selection
        return [self._single_task_fallback(query)]

    def _llm_decompose(self, query: str, thinking_mode: str) -> list[_TaskSpec]:
        valid_tools = list(self.cio.tool_registry.keys())
        # Build rich tool capability descriptions from the registry
        tool_sections: list[str] = []
        for name in valid_tools:
            entry = self.cio.tool_registry[name]
            props = entry.input_schema.get("properties", {})
            out_props = (
                entry.output_schema
                .get("properties", {})
                .get("outputs", {})
                .get("properties", {})
            )
            input_desc = ", ".join(
                f"{k} ({v.get('type', '?')})"
                for k, v in props.items()
            )
            output_desc = ", ".join(
                f"{k} ({v.get('type', '?')})"
                for k, v in out_props.items()
            )
            constraints = entry.constraints
            constraint_str = f"  Constraints: {'; '.join(constraints)}" if constraints else ""
            tool_sections.append(
                f"- {name}: {entry.description}\n"
                f"  Inputs: {input_desc or '(none)'}\n"
                f"  Outputs: {output_desc or '(see tool result)'}"
                + (f"\n{constraint_str}" if constraint_str else "")
            )
        tool_list = "\n".join(tool_sections)

        doc_list = "\n".join(
            f"- {e.standard} ({e.year_version}): {e.title}"
            for e in self.cio.document_registry
        )
        if not doc_list:
            doc_list = "(no documents loaded)"

        mode_hint = ""
        if thinking_mode == "standard":
            mode_hint = "\nStandard mode: return exactly ONE task. Do not decompose into multiple sub-tasks."
        elif thinking_mode == "extended":
            mode_hint = "\nExtended mode: be thorough. Include separate search and calculation tasks where appropriate."

        raw = self.cio.orchestrator_llm.generate(
            system_prompt=_DECOMPOSE_SYSTEM,
            user_prompt=(
                f"User query: {query}\n\n"
                f"Available calculator tools:\n{tool_list}\n\n"
                f"Available Eurocode documents:\n{doc_list}\n"
                f"{mode_hint}\n\n"
                "Return JSON array of tasks."
            ),
            temperature=self.settings.decompose_temperature,
            max_tokens=self.settings.decompose_max_tokens,
            reasoning_effort=(self.settings.decompose_reasoning_effort or None) if thinking_mode != "extended" else None,
        )

        parsed = parse_json_loose(raw)
        if not isinstance(parsed, list):
            # Maybe it's a single object
            if isinstance(parsed, dict):
                parsed = [parsed]
            else:
                raise ValueError(f"Expected JSON array, got: {type(parsed)}")

        tasks: list[_TaskSpec] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            tools = [t for t in item.get("tools", []) if t in self.cio.tool_registry]
            tasks.append(
                _TaskSpec(
                    summary=str(item.get("summary", "Task")),
                    query=str(item.get("query", query)),
                    search_query=str(item.get("search_query", item.get("query", query))),
                    needs_search=bool(item.get("needs_search", True)),
                    tools=tools,
                    inputs=item.get("inputs", {}),
                )
            )

        if not tasks:
            return [self._single_task_fallback(query)]

        return tasks

    def _single_task_fallback(self, query: str) -> _TaskSpec:
        """Build a single task without tool selection (LLM unavailable fallback)."""
        intent = self.cio._query_intent(query)

        needs_search = not intent.get("pure_calculation", False)

        # Extract concrete values from query for tool inputs
        inputs = self._extract_inputs_from_query(query)

        return _TaskSpec(
            summary=_summarize_query(query),
            query=query,
            search_query=query,
            needs_search=needs_search,
            tools=[],
            inputs=inputs,
        )

    def _extract_inputs_from_query(self, query: str) -> dict[str, Any]:
        """Pull concrete engineering values from the query text."""
        inputs: dict[str, Any] = {}
        lowered = query.lower()

        # Section name
        m = re.search(r"\b((?:ipe|hea|heb|hem)\s*\d+)\b", lowered, re.IGNORECASE)
        if m:
            inputs["section_name"] = m.group(1).upper().replace(" ", "")

        # Steel grade
        m = re.search(r"\b(s(?:235|275|355|420|460))\b", lowered, re.IGNORECASE)
        if m:
            inputs["steel_grade"] = m.group(1).upper()

        # Bolt diameter
        m = re.search(r"\bm(\d+)\b", lowered)
        if m:
            inputs["bolt_diameter_mm"] = int(m.group(1))

        # Bolt grade
        m = re.search(r"\b(4\.6|4\.8|5\.6|5\.8|6\.8|8\.8|10\.9|12\.9)\b", lowered)
        if m:
            inputs["bolt_grade"] = m.group(1)

        # Span
        m = re.search(r"(\d+(?:\.\d+)?)\s*m(?:\s+span|\b)", lowered)
        if m:
            inputs["span_m"] = float(m.group(1))

        # UDL
        m = re.search(r"(\d+(?:\.\d+)?)\s*kn/m\b", lowered)
        if m:
            inputs["udl_kn_m"] = float(m.group(1))

        # Point load
        m = re.search(r"(\d+(?:\.\d+)?)\s*kn\b(?!\s*/)", lowered)
        if m and "point" in lowered:
            inputs["point_load_kn"] = float(m.group(1))

        # Elastic critical moment for LTB
        m = re.search(r"m_?cr\s*=\s*(\d+(?:\.\d+)?)\s*kn", lowered, re.IGNORECASE)
        if m:
            inputs["M_cr_kNm"] = float(m.group(1))

        # LTB method per §6.3.2.3 (rolled/welded)
        if "6.3.2.3" in query or ("rolled" in lowered and ("ltb" in lowered or "lateral" in lowered)):
            inputs["method"] = "rolled_welded"

        return inputs

    # ── Task-level tool input building ──────────────────────────────

    def _build_task_tool_inputs(
        self,
        tool_name: str,
        task: _TaskSpec,
        all_tool_outputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build inputs for a tool using the full session context.

        Resolution:
        1. Values extracted from the user query (task.inputs) + settings defaults
        2. LLM resolves ALL expected params when prior tool outputs exist
        """
        # Start with task-level extracted inputs
        base = dict(task.inputs)

        entry = self.cio.tool_registry.get(tool_name)
        if entry:
            props = entry.input_schema.get("properties", {})
            expected = set(props.keys())

            # LLM resolves ALL expected params when prior outputs exist
            flat = _flatten_tool_outputs(all_tool_outputs)
            if flat and self.cio.orchestrator_llm.available:
                # Build plan context for LLM
                plan_context = None
                if self._current_tasks:
                    plan_context = [
                        {
                            "step": idx + 1,
                            "summary": t.summary,
                            "tools": t.tools,
                        }
                        for idx, t in enumerate(self._current_tasks)
                    ]
                resolved = self.cio._llm_resolve_inputs(
                    tool_name, props, list(expected), base,
                    all_tool_outputs, plan_context=plan_context,
                    original_query=self._original_query or None,
                )
                base.update(resolved)

            # Filter to expected params
            if expected:
                base = {k: v for k, v in base.items() if k in expected and v is not None}
            else:
                base = {k: v for k, v in base.items() if v is not None}

            # Normalize enum values (e.g. "UDL" → "udl")
            for k, v in list(base.items()):
                if isinstance(v, str) and k in props:
                    allowed = props[k].get("enum")
                    if allowed and v not in allowed:
                        lower = v.lower()
                        if lower in allowed:
                            base[k] = lower
        else:
            base = {k: v for k, v in base.items() if v is not None}

        return base

    def _fix_tool_inputs(
        self,
        tool_name: str,
        failed_inputs: dict[str, Any],
        error_msg: str,
        task: _TaskSpec,
    ) -> dict[str, Any] | None:
        """Ask the LLM to correct tool inputs after a failure."""
        if not self.cio.orchestrator_llm.available:
            return None

        entry = self.cio.tool_registry.get(tool_name)
        schema_desc = ""
        if entry:
            props = entry.input_schema.get("properties", {})
            required = entry.input_schema.get("required", [])
            schema_desc = json.dumps(
                {"properties": props, "required": required}, indent=2
            )

        try:
            raw = self.cio.orchestrator_llm.generate(
                system_prompt=(
                    "You are a tool-input repair agent. A calculator tool failed due "
                    "to invalid inputs. Return ONLY a JSON object containing the "
                    "parameters that need to CHANGE to fix the error. Do NOT include "
                    "parameters that are already correct — only the ones that caused "
                    "the failure or need updating. No explanation."
                ),
                user_prompt=(
                    f"Tool: {tool_name}\n"
                    f"Original user query: {self._original_query or task.query}\n"
                    f"Sub-task: {task.query}\n"
                    f"Input schema:\n{schema_desc}\n\n"
                    f"Failed inputs: {json.dumps(failed_inputs, default=str)}\n"
                    f"Error: {error_msg}\n\n"
                    "Return ONLY the parameters that need to change as JSON."
                ),
                temperature=self.settings.fix_inputs_temperature,
                max_tokens=self.settings.fix_inputs_max_tokens,
                reasoning_effort=self.settings.fix_inputs_reasoning_effort or None,
            )
            fixed = parse_json_loose(raw)
            if isinstance(fixed, dict):
                # Merge corrections into original inputs so that
                # values resolved by the LLM (e.g. section_class)
                # are preserved when the fix only changes a subset
                # of parameters (e.g. section_name).
                merged = {**failed_inputs, **fixed}
                # Re-apply enum normalization
                if entry:
                    props = entry.input_schema.get("properties", {})
                    for k, v in list(merged.items()):
                        if isinstance(v, str) and k in props:
                            allowed = props[k].get("enum")
                            if allowed and v not in allowed:
                                lower = v.lower()
                                if lower in allowed:
                                    merged[k] = lower
                return merged
        except Exception as exc:
            logger.warning("fix_tool_inputs_failed", extra={"error": str(exc)})

        return None

    def _upstream_resolve_inputs(
        self,
        tool_name: str,
        task: _TaskSpec,
        all_tool_outputs: dict[str, dict[str, Any]],
        error_log: list[str],
    ) -> dict[str, Any] | None:
        """Re-resolve ALL tool inputs from complete context including error history.

        Called after the local _fix_tool_inputs approach has already failed once.
        Goes upstream: gives the LLM the full picture (tool schema, all prior
        outputs, the original query, AND the accumulated error log) so it can
        rethink the inputs from scratch rather than just patching the last attempt.
        """
        if not self.cio.orchestrator_llm.available:
            return None

        entry = self.cio.tool_registry.get(tool_name)
        if not entry:
            return None

        props = entry.input_schema.get("properties", {})
        required = entry.input_schema.get("required", [])

        # Build context of all prior tool outputs
        context_lines: list[str] = []
        for src_tool, result in all_tool_outputs.items():
            outputs = result.get("outputs", {})
            if outputs:
                context_lines.append(f"  {src_tool}: {json.dumps(outputs)}")

        error_history = "\n".join(error_log)

        try:
            raw = self.cio.orchestrator_llm.generate(
                system_prompt=(
                    "You are a tool-input resolution agent. A calculator tool has failed "
                    "multiple times with different inputs. You must resolve the correct "
                    "inputs from scratch using the full context: the user's query, all "
                    "prior tool outputs, and the error history showing what was tried and "
                    "why it failed. Return ONLY a JSON object with ALL required parameters."
                ),
                user_prompt=(
                    f"Tool: {tool_name}\n"
                    f"Description: {entry.description}\n"
                    f"Original user query: {self._original_query or task.query}\n"
                    f"Sub-task: {task.query}\n\n"
                    f"Input schema:\n{json.dumps({'properties': props, 'required': required}, indent=2)}\n\n"
                    f"Prior tool outputs:\n" + ("\n".join(context_lines) or "(none)") + "\n\n"
                    f"Error history (all failed attempts):\n{error_history}\n\n"
                    "Resolve ALL inputs from scratch. Return ONLY a JSON object."
                ),
                temperature=self.settings.upstream_resolve_temperature,
                max_tokens=self.settings.upstream_resolve_max_tokens,
                **({"reasoning_effort": self.settings.upstream_resolve_reasoning_effort} if self.settings.upstream_resolve_reasoning_effort else {}),
            )
            result = parse_json_loose(raw)
            if isinstance(result, dict):
                # Normalize enum values
                for k, v in list(result.items()):
                    if isinstance(v, str) and k in props:
                        allowed = props[k].get("enum")
                        if allowed and v not in allowed:
                            lower = v.lower()
                            if lower in allowed:
                                result[k] = lower
                return result
        except Exception as exc:
            logger.warning("upstream_resolve_failed", extra={"error": str(exc)})

        return None

    # ── Equation extraction for math_calculator ─────────────────────

    _EQUATION_EXTRACT_SYSTEM = (
        "You are a Eurocode equation extraction engine.\n"
        "Given retrieved clause text and a user's engineering query, extract the\n"
        "relevant equation(s) needed to answer the query.\n\n"
        "Return ONLY a JSON object with two keys:\n"
        '  "equations": [{...}, ...],\n'
        '  "variables": {...}\n\n'
        "Each equation object:\n"
        '  "name": result variable name (e.g. "N_pl_Rd", "A_net")\n'
        '  "expression": math expression using Python syntax (e.g. "A * f_y / gamma_M0")\n'
        '  "unit": unit string (e.g. "N", "mm2", "kN")\n'
        '  "description": what this computes (e.g. "Plastic resistance of gross section")\n\n'
        "The variables dict maps variable names to numeric values.\n\n"
        "EXPRESSION SYNTAX — CRITICAL:\n"
        "- Use Python math syntax: sqrt(), **, /, *, +, -, min(), max()\n"
        "- Conditionals MUST use Python ternary syntax: value_if_true if condition else value_if_false\n"
        "  Example: 0.7 if n_bolts >= 3 else (0.6 if n_bolts == 2 else 0.45)\n"
        "  NEVER use Excel-style if(): if(cond, a, b) is INVALID.\n"
        "- Nested conditionals: wrap inner ternaries in parentheses.\n\n"
        "TABLE LOOKUPS — use piecewise conditionals:\n"
        "When a clause references a table with discrete values (e.g. Table 3.8 for beta),\n"
        "encode it as a chained ternary expression.\n"
        "Example — Table 3.8 beta_3 with linear interpolation between p1=55mm and p1=250mm:\n"
        '  "expression": "0.7 + (p_1 - 55) * (0.9 - 0.7) / (250 - 55) if 55 < p_1 < 250 else (0.7 if p_1 <= 55 else 0.9)"\n\n'
        "Rules:\n"
        "- Extract equations directly from the clause text. Follow the Eurocode formulas exactly.\n"
        "- Equations are evaluated sequentially — later equations can reference earlier results by name.\n"
        "- Include ALL intermediate steps (don't skip steps).\n"
        "- For variables: use values from the user query and prior tool outputs.\n"
        "  If a value is available from prior tool outputs, use the EXACT numeric value.\n"
        "  If a standard value is needed (e.g. gamma_M0=1.0, gamma_M1=1.0, gamma_M2=1.25), include it.\n"
        "- Convert units where needed (e.g. cm2 to mm2: multiply by 100).\n"
        "- Add a final equation converting the result to practical units (e.g. N to kN).\n"
        "- Keep equations concise. Avoid overly long expressions.\n"
        "- Return valid JSON only, no markdown fences, no explanation."
    )

    def _extract_equations(
        self,
        task: _TaskSpec,
        task_clauses: list,
        all_tool_outputs: dict[str, dict[str, Any]],
        error_hint: str | None = None,
    ) -> dict[str, Any]:
        """Extract equations from clause text for math_calculator.

        Uses LLM to analyze retrieved clause text against the user's query
        and produce structured {equations, variables} input for math_calculator.

        When *error_hint* is provided (retry after a failed attempt), it is
        appended to the prompt so the LLM can avoid the same mistake.
        """
        if not self.cio.orchestrator_llm.available:
            return task.inputs

        # Build clause text
        clause_evidence = []
        for r in task_clauses:
            text = r.clause.text.strip()
            clause_evidence.append(
                f"[{r.clause.clause_id} — {r.clause.clause_title}]: {text}"
            )
        clause_text = "\n\n".join(clause_evidence) if clause_evidence else "(no clauses retrieved)"

        # Build prior tool outputs context
        context_lines: list[str] = []
        for src_tool, result in all_tool_outputs.items():
            outputs = result.get("outputs", {})
            if outputs:
                context_lines.append(f"  {src_tool}: {json.dumps(outputs)}")
        prior_outputs = "\n".join(context_lines) if context_lines else "(none)"

        # User-provided values
        user_vals = json.dumps(task.inputs, default=str) if task.inputs else "(none)"

        # Use original query so LLM sees all user-stated values (S355, d0=22, etc.)
        original_query = self._original_query or task.query

        error_section = ""
        if error_hint:
            error_section = (
                f"\n\nPREVIOUS ATTEMPT FAILED with error:\n{error_hint}\n"
                "Fix the expression syntax. Remember: use Python ternary "
                "(value if cond else other), NEVER Excel-style if().\n"
            )

        try:
            raw = self.cio.orchestrator_llm.generate(
                system_prompt=self._EQUATION_EXTRACT_SYSTEM,
                user_prompt=(
                    f"Original user query: {original_query}\n\n"
                    f"Current sub-task: {task.query}\n\n"
                    f"User-provided values: {user_vals}\n\n"
                    f"Prior tool outputs (available as known values):\n{prior_outputs}\n\n"
                    f"Retrieved Eurocode clause text:\n{clause_text}\n\n"
                    f"{error_section}"
                    "Extract the equations and variables needed to answer the query. "
                    "Return JSON with 'equations' and 'variables' keys."
                ),
                temperature=self.settings.equation_extract_temperature,
                max_tokens=self.settings.equation_extract_max_tokens,
                **({"reasoning_effort": self.settings.equation_extract_reasoning_effort} if self.settings.equation_extract_reasoning_effort else {}),
            )
            parsed = parse_json_loose(raw)
            if isinstance(parsed, dict) and "equations" in parsed and "variables" in parsed:
                return parsed
            logger.warning(
                "equation_extract_unexpected_format",
                extra={"raw_preview": raw[:200]},
            )
        except Exception as exc:
            logger.warning(
                "equation_extract_failed",
                extra={"error": str(exc), "task": task.summary},
            )

        # Fallback: return task inputs as-is (math_calculator will likely fail,
        # but the retry mechanism can attempt to fix it)
        return task.inputs

    # ── Task-level composition ──────────────────────────────────────

    def _compose_task_answer(
        self,
        *,
        task: _TaskSpec,
        original_query: str,
        retrieved: list,
        tool_outputs: dict[str, dict[str, Any]],
        tool_failures: list[str] | None = None,
        thinking_mode: str,
    ) -> str:
        """Compose a focused answer for one task with isolated context."""

        if not self.cio.orchestrator_llm.available:
            return self._fallback_compose(task, retrieved, tool_outputs)

        # Build clause evidence — only this task's clauses
        clause_evidence = []
        for r in retrieved:
            text = r.clause.text.strip()
            clause_evidence.append(
                f"[{r.clause.clause_id} — {r.clause.clause_title}]: {text}"
            )

        # Build tool evidence — only this task's outputs
        tool_evidence: dict[str, Any] = {}
        for tname, tout in tool_outputs.items():
            tool_evidence[tname] = {
                "inputs_used": tout.get("inputs_used", {}),
                "outputs": tout.get("outputs", {}),
                "notes": tout.get("notes", []),
            }

        clause_text = "\n".join(clause_evidence) if clause_evidence else "(no clauses retrieved)"
        tool_text = json.dumps(tool_evidence, default=str) if tool_evidence else "(no tool results)"

        is_multi_task = task.query != original_query
        context_note = (
            f"This is one part of a multi-part question. The full question was: {original_query}\n"
            f"Focus your answer on this specific sub-task: {task.query}\n"
            if is_multi_task
            else ""
        )

        # If tools failed and we have no tool outputs, tell the LLM honestly
        failure_note = ""
        if tool_failures and not tool_outputs:
            failure_note = (
                "\n\nIMPORTANT: The required calculator tools failed after multiple "
                "retries. You MUST NOT answer the calculation from general knowledge. "
                "Instead, explain that the calculation could not be completed right now "
                "due to a tool error, apologise, and suggest the user try again or "
                "rephrase their query. Be specific about what went wrong.\n"
                f"Tool errors: {'; '.join(tool_failures)}\n"
            )

        reasoning_effort = None
        if thinking_mode == "standard":
            reasoning_effort = self.settings.compose_reasoning_effort or None
        elif thinking_mode == "extended":
            reasoning_effort = "high"

        try:
            raw = self.cio.orchestrator_llm.generate(
                system_prompt=_COMPOSE_SYSTEM,
                user_prompt=(
                    f"{context_note}"
                    f"Colleague's question: {task.query}\n\n"
                    f"Retrieved EC3 clauses:\n{clause_text}\n\n"
                    f"Tool results:\n{tool_text}\n\n"
                    f"{failure_note}"
                    "Give a detailed engineering answer. Wrap ALL math in $...$ delimiters."
                ),
                temperature=self.settings.compose_temperature,
                max_tokens=self.settings.compose_max_tokens,
                reasoning_effort=reasoning_effort,
            )
            return raw.strip()
        except Exception as exc:
            logger.warning(
                "agent_compose_failed",
                extra={"error": str(exc), "task": task.summary},
            )
            return self._fallback_compose(task, retrieved, tool_outputs)

    def _fallback_compose(
        self,
        task: _TaskSpec,
        retrieved: list,
        tool_outputs: dict[str, dict[str, Any]],
    ) -> str:
        """Build a basic answer without LLM."""
        parts: list[str] = []

        if tool_outputs:
            for tname, payload in tool_outputs.items():
                outputs = payload.get("outputs", {})
                if outputs:
                    pretty = tname.replace("_ec3", "").replace("_", " ").title()
                    parts.append(f"**{pretty}** results:")
                    for k, v in outputs.items():
                        if isinstance(v, (int, float, str, bool)):
                            parts.append(f"- {k}: **{v}**")

        if retrieved:
            parts.append("\n**Relevant clauses:**")
            for r in retrieved[:4]:
                parts.append(
                    f"- Cl. {r.clause.clause_id}: {r.clause.clause_title}"
                )

        if not parts:
            parts.append(
                "I processed your request but could not generate a detailed answer. "
                "Please check the tool results above for the raw calculation output."
            )

        return "\n".join(parts)

    # ── Utilities ───────────────────────────────────────────────────

    def _top_k_for_mode(self, mode: str) -> int:
        if mode == "standard":
            return 4
        if mode == "extended":
            return 10
        return self.settings.top_k_clauses

    def _build_what_i_used(
        self,
        tasks: list[_TaskSpec],
        retrieval_trace: list[RetrievalTraceStep],
        tool_trace: list[ToolTraceStep],
    ) -> list[str]:
        summaries = [f"Agent mode: {len(tasks)} task(s)"]
        for i, t in enumerate(tasks):
            summaries.append(f"Task {i + 1}: {t.summary}")
        if retrieval_trace:
            summaries.append(f"Retrieval: {len(retrieval_trace)} search pass(es)")
        if tool_trace:
            chain = " → ".join(s.tool_name for s in tool_trace)
            summaries.append(f"Tool chain: {chain}")
        return summaries


# ── Module-level helpers ────────────────────────────────────────────

_USED_CLAUSES_RE = re.compile(
    r"<!--\s*USED_CLAUSES\s*:\s*(\[.*?\])\s*-->",
    re.DOTALL,
)


def _extract_used_clauses(raw: str) -> tuple[str, set[str]]:
    """Strip the <!--USED_CLAUSES:[...]-->  tag from the LLM output.

    Returns (cleaned_narrative, set_of_clause_ids).
    """
    match = _USED_CLAUSES_RE.search(raw)
    if not match:
        return raw, set()
    cleaned = raw[: match.start()].rstrip() + raw[match.end() :]
    try:
        ids = json.loads(match.group(1))
        if isinstance(ids, list):
            return cleaned, {str(c) for c in ids if c}
    except (json.JSONDecodeError, ValueError):
        pass
    return cleaned, set()


def _chunk_naturally(text: str, target: int = 80) -> list[str]:
    """Split text at paragraph/sentence boundaries for natural streaming."""
    if not text:
        return []

    chunks: list[str] = []

    # First split by paragraphs
    paragraphs = re.split(r"\n\n+", text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= target * 2:
            # Small paragraph — yield as-is (with trailing newlines)
            chunks.append(para + "\n\n")
        else:
            # Large paragraph — split by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sent in sentences:
                if current and len(current) + len(sent) > target:
                    chunks.append(current)
                    current = sent + " "
                else:
                    current += sent + " "
            if current.strip():
                chunks.append(current.strip() + "\n\n")

    return chunks


def _summarize_query(query: str) -> str:
    """Create a short summary of a query for display as a plan step."""
    cleaned = query.strip()
    if len(cleaned) <= 60:
        return cleaned
    # Truncate at a word boundary
    truncated = cleaned[:57]
    last_space = truncated.rfind(" ")
    if last_space > 30:
        truncated = truncated[:last_space]
    return truncated + "..."
