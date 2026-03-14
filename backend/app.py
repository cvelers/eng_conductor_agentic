from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from openai import OpenAI

from backend.auth import create_auth_router, require_auth
from backend.attachments import AttachmentBundle, process_attachments, saved_attachments_payload
from backend.config import Settings
from backend.threads import create_threads_router
from backend.llm.factory import get_search_provider, get_orchestrator_provider
from backend.logging_config import configure_logging
from backend.registries.document_registry import load_all_clauses, load_document_registry
from backend.retrieval.agentic_search import AgenticRetriever

from backend.schemas import ChatRequest, FEAAnswerRequest, FEAResultsRequest

# New agent imports
from backend.agent.loop import run_agent_loop
from backend.agent.tools import TOOLS, build_tool_dispatcher
from backend.agent.prompt import SYSTEM_PROMPT
from backend.agent.stream_adapter import adapt_event
from backend.agent.context import (
    compact_if_needed,
    context_usage_snapshot,
    convert_frontend_history,
    estimate_messages_tokens,
    should_continue_from_ask_user,
)

# FEA (kept as separate mode)
from backend.orchestrator.fea_analyst import FEAAnalystLoop
from backend.orchestrator.fea_routing import should_route_to_fea

logger = logging.getLogger(__name__)


def _append_thread_log(settings: Settings, payload: dict) -> None:
    log_path = settings.resolved_orchestrator_thread_log_path
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entries: list[dict] = []

        if log_path.exists():
            raw = log_path.read_text(encoding="utf-8").strip()
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        entries = [item for item in parsed if isinstance(item, dict)]
                    elif isinstance(parsed, dict):
                        entries = [parsed]
                except Exception:
                    migrated: list[dict] = []
                    for line in raw.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(item, dict):
                            migrated.append(item)
                    entries = migrated

        entries.append(payload)
        log_path.write_text(json.dumps(entries, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("thread_log_write_failed", extra={"path": str(log_path)})


def _history_payload(history: list) -> list[dict]:
    rows: list[dict] = []
    for item in history or []:
        if hasattr(item, "model_dump"):
            rows.append(item.model_dump())
        elif isinstance(item, dict):
            rows.append(dict(item))
        else:
            rows.append({"role": str(getattr(item, "role", "")), "content": str(getattr(item, "content", ""))})
    return rows


def _prepare_attachment_bundle(
    settings: Settings,
    *,
    message_text: str,
    attachments: list,
    request_id: str,
) -> AttachmentBundle:
    if not attachments:
        return AttachmentBundle(user_content=message_text, routing_text=message_text)
    try:
        return process_attachments(
            settings,
            attachments,
            message_text=message_text,
            request_id=request_id,
        )
    except Exception:
        logger.exception(
            "attachment_processing_failed",
            extra={"request_id": request_id, "attachments_count": len(attachments)},
        )
        fallback_note = (
            "Technical attachments were supplied, but the attachment-processing step failed. "
            "Use the text prompt only and state that attachment parsing was unavailable if needed."
        )
        composed = f"{message_text}\n\n{fallback_note}".strip() if message_text else fallback_note
        return AttachmentBundle(user_content=composed, routing_text=composed)


def _format_ask_user_continuation(message_text: str) -> str:
    return (
        "[User's answer to your ask_user question]\n"
        f"{message_text}\n\n"
        "CRITICAL: You are CONTINUING a previous calculation. "
        "All section properties, material data, clauses, and calculations "
        "from earlier tool calls are still in the conversation history — "
        "reuse them. Do NOT call todo_write with a new plan. Do NOT "
        "re-fetch section properties, material data, or clauses you "
        "already have. Simply use the user's answer together with your "
        "existing data to complete the remaining calculation steps."
    )


def _prepare_request_bundle_for_execution(
    settings: Settings,
    *,
    fea_provider,
    request: ChatRequest,
    request_id: str,
) -> tuple[bool, bool, AttachmentBundle]:
    """Prepare the request bundle and decide whether this turn should enter FEA.

    FEA routing is evaluated against the raw user turn first, before an ask_user
    continuation wrapper can lock the turn into the normal chat path.
    """
    is_ask_reply = should_continue_from_ask_user(
        request.history,
        request.is_ask_user_reply,
    )
    raw_bundle = _prepare_attachment_bundle(
        settings,
        message_text=request.message,
        attachments=request.attachments,
        request_id=request_id,
    )
    route_to_fea = should_route_to_fea(
        fea_provider,
        raw_bundle.routing_text,
        request.history,
    )
    if route_to_fea or not is_ask_reply:
        return route_to_fea, is_ask_reply, raw_bundle

    wrapped_bundle = _prepare_attachment_bundle(
        settings,
        message_text=_format_ask_user_continuation(request.message),
        attachments=request.attachments,
        request_id=request_id,
    )
    return False, True, wrapped_bundle

def create_app(settings: Settings | None = None) -> FastAPI:
    load_dotenv()
    active_settings = settings or Settings.load()
    configure_logging(active_settings.log_level)

    # ── Data loading ─────────────────────────────────────────────────
    doc_registry = load_document_registry(active_settings.resolved_document_registry_path)
    clauses = load_all_clauses(active_settings.project_root, doc_registry)

    # ── Search provider (for retriever's internal LLM reranking) ─────
    search_provider = get_search_provider(active_settings)
    retriever = AgenticRetriever(
        settings=active_settings,
        search_provider=search_provider,
        clauses=clauses,
    )

    # ── OpenAI client for agent loop ─────────────────────────────────
    client = OpenAI(
        api_key=active_settings.orchestrator_api_key,
        base_url=active_settings.orchestrator_base_url,
    )

    # ── Tool dispatcher ──────────────────────────────────────────────
    tool_dispatcher = build_tool_dispatcher(retriever, clauses, search_provider)

    # ── Validator client (independent LLM for grounding checks) ──────
    validator_client: OpenAI | None = None
    if active_settings.validator_enabled:
        validator_api_key = active_settings.validator_api_key or active_settings.orchestrator_api_key
        validator_client = OpenAI(
            api_key=validator_api_key,
            base_url=active_settings.validator_base_url,
        )

    # ── FEA (separate mode, unchanged) ───────────────────────────────
    fea_sessions: dict[str, FEAAnalystLoop] = {}

    # FEA needs its own orchestrator provider for the analyst LLM
    fea_provider = get_orchestrator_provider(active_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):  # noqa: ARG001
        yield

    app = FastAPI(title="Eurocodes Chatbot", version="0.4.0", lifespan=lifespan)
    app.state.settings = active_settings
    app.state.fea_sessions = fea_sessions

    app.include_router(create_auth_router(active_settings))
    app.include_router(create_threads_router(active_settings))

    frontend_dir = active_settings.project_root / "frontend"
    data_dir = active_settings.project_root / "data"
    app.mount("/data", StaticFiles(directory=data_dir), name="data")
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> dict:
        """Non-streaming endpoint. Returns the same response the user sees."""
        request_id = uuid4().hex
        route_to_fea, _is_ask_reply, attachment_bundle = _prepare_request_bundle_for_execution(
            active_settings,
            fea_provider=fea_provider,
            request=request,
            request_id=request_id,
        )
        if route_to_fea:
            raise HTTPException(status_code=400, detail="FEA requests not supported on /api/chat")

        messages = convert_frontend_history(request.history)
        messages.append({"role": "user", "content": attachment_bundle.user_content})

        messages = compact_if_needed(
            messages, SYSTEM_PROMPT,
            context_window=active_settings.agent_context_window,
        )

        _WEB_TOOLS = {"web_search", "fetch_url"}
        active_tools = TOOLS if request.web_search else [
            t for t in TOOLS if t["function"]["name"] not in _WEB_TOOLS
        ]

        full_response = ""
        assumptions: list[str] = []
        session_memory: dict[str, object] = {}
        async for event in run_agent_loop(
            client=client,
            model=active_settings.orchestrator_model,
            system_prompt=SYSTEM_PROMPT,
            messages=messages,
            tools=active_tools,
            tool_dispatcher=tool_dispatcher,
            max_rounds=active_settings.agent_max_rounds,
            temperature=active_settings.agent_temperature,
            max_tokens=active_settings.agent_max_tokens,
            reasoning_effort=active_settings.orchestrator_reasoning_effort or None,
            grounding_validation=active_settings.validator_enabled,
            validator_client=validator_client,
            validator_model=active_settings.validator_model,
            validator_temperature=active_settings.validator_temperature,
            validator_max_tokens=active_settings.validator_max_tokens,
            validator_reasoning_effort=active_settings.validator_reasoning_effort or None,
        ):
            if event.get("type") == "fea_handoff":
                raise HTTPException(status_code=400, detail="FEA requests not supported on /api/chat")
            if event.get("type") == "delta":
                full_response += event.get("content", "")
            elif event.get("type") == "_session_memory":
                memory = event.get("memory", {})
                session_memory = memory if isinstance(memory, dict) else {}
            elif event.get("type") == "done":
                full_response = event.get("content", full_response)
                assumptions = event.get("assumptions", []) if isinstance(event.get("assumptions", []), list) else []
            elif event.get("type") == "error":
                raise HTTPException(status_code=500, detail=event.get("message", "Agent error"))

        return {
            "answer": full_response,
            "assumptions": assumptions,
            "session_memory": session_memory,
            "attachments": saved_attachments_payload(attachment_bundle.saved_attachments),
        }

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:

        async def _stream_generator():
            request_id = uuid4().hex
            all_events: list[dict] = []
            error_detail: str | None = None
            final_answer: str = ""
            try:
                route_to_fea, _is_ask_reply, attachment_bundle = _prepare_request_bundle_for_execution(
                    active_settings,
                    fea_provider=fea_provider,
                    request=request,
                    request_id=request_id,
                )
                if attachment_bundle.saved_attachments:
                    yield json.dumps(
                        {
                            "type": "user_attachments",
                            "attachments": saved_attachments_payload(attachment_bundle.saved_attachments),
                        },
                        default=str,
                    ) + "\n"
                # ── FEA detection ────────────────────────────────────
                if route_to_fea:
                    analyst = FEAAnalystLoop(
                        llm=fea_provider,
                        settings=active_settings,
                    )
                    fea_sessions[analyst.session_id] = analyst
                    yield json.dumps({"type": "fea_session_created", "session_id": analyst.session_id}) + "\n"
                    try:
                        async for fea_event_type, fea_payload in analyst.run_stream(
                            attachment_bundle.routing_text, history=request.history,
                        ):
                            yield json.dumps({"type": fea_event_type, **fea_payload}) + "\n"
                            all_events.append({"event_type": fea_event_type, **fea_payload})
                            await asyncio.sleep(0.005)
                    finally:
                        fea_sessions.pop(analyst.session_id, None)
                    yield json.dumps({"type": "machine", "node": "fea_analyst", "status": "done", "title": "FEA Analyst", "detail": "Analysis complete."}) + "\n"
                    return

                # ── Agent loop (main path) ───────────────────────────
                messages = convert_frontend_history(request.history)

                # Detect ask_user continuation: if the last assistant
                # message's tool_context ends with an ask_user call,
                # the user's message is a reply — tell the agent to
                # continue from where it left off, not replan.
                messages.append({"role": "user", "content": attachment_bundle.user_content})

                # Auto-compact if conversation is long
                messages = compact_if_needed(
                    messages, SYSTEM_PROMPT,
                    context_window=active_settings.agent_context_window,
                )

                # Filter web tools if web search is disabled
                _WEB_TOOLS = {"web_search", "fetch_url"}
                active_tools = TOOLS if request.web_search else [
                    t for t in TOOLS if t["function"]["name"] not in _WEB_TOOLS
                ]

                session_tokens = 0
                tool_context = ""
                session_memory: dict[str, object] = {}
                async for event in run_agent_loop(
                    client=client,
                    model=active_settings.orchestrator_model,
                    system_prompt=SYSTEM_PROMPT,
                    messages=messages,
                    tools=active_tools,
                    tool_dispatcher=tool_dispatcher,
                    max_rounds=active_settings.agent_max_rounds,
                    temperature=active_settings.agent_temperature,
                    max_tokens=active_settings.agent_max_tokens,
                    reasoning_effort=active_settings.orchestrator_reasoning_effort or None,
                    grounding_validation=active_settings.validator_enabled,
                    validator_client=validator_client,
                    validator_model=active_settings.validator_model,
                    validator_temperature=active_settings.validator_temperature,
                    validator_max_tokens=active_settings.validator_max_tokens,
                    validator_reasoning_effort=active_settings.validator_reasoning_effort or None,
                ):
                    if event.get("type") == "_session_tokens":
                        session_tokens = event.get("tokens", 0)
                        continue
                    if event.get("type") == "_tool_context":
                        tool_context = event.get("summary", "")
                        continue
                    if event.get("type") == "_session_memory":
                        memory = event.get("memory", {})
                        session_memory = memory if isinstance(memory, dict) else {}
                        continue
                    if event.get("type") == "fea_handoff":
                        handoff_task = str(event.get("task", "") or "").strip() or attachment_bundle.routing_text
                        analyst = FEAAnalystLoop(
                            llm=fea_provider,
                            settings=active_settings,
                        )
                        fea_sessions[analyst.session_id] = analyst
                        yield json.dumps({"type": "fea_session_created", "session_id": analyst.session_id}) + "\n"
                        try:
                            async for fea_event_type, fea_payload in analyst.run_stream(
                                handoff_task, history=request.history,
                            ):
                                yield json.dumps({"type": fea_event_type, **fea_payload}) + "\n"
                                all_events.append({"event_type": fea_event_type, **fea_payload})
                                await asyncio.sleep(0.005)
                        finally:
                            fea_sessions.pop(analyst.session_id, None)
                        yield json.dumps(
                            {
                                "type": "machine",
                                "node": "fea_analyst",
                                "status": "done",
                                "title": "FEA Analyst",
                                "detail": "Analysis complete.",
                            }
                        ) + "\n"
                        return
                    adapted = adapt_event(event)
                    all_events.append(adapted)
                    if event.get("type") == "done":
                        final_answer = event.get("content", "")
                        # Inject tool context into the final event so the
                        # frontend can store it with the assistant message
                        if tool_context:
                            adapted["tool_context"] = tool_context
                        if session_memory:
                            adapted["session_memory"] = session_memory
                    yield json.dumps(adapted, default=str) + "\n"
                    await asyncio.sleep(0.005)

                # Context circle shows the session history size (what the
                # next request will actually receive), not the transient
                # in-loop total with all tool calls.
                if not session_tokens:
                    session_tokens = estimate_messages_tokens(messages, SYSTEM_PROMPT)
                cw = active_settings.agent_context_window
                tokens_left = max(0, cw - session_tokens)
                used_pct = round(session_tokens / cw * 100, 1) if cw else 0
                level = "low" if used_pct < 50 else "medium" if used_pct < 75 else "high" if used_pct < 90 else "critical"
                usage = {
                    "estimated_tokens": session_tokens,
                    "context_window": cw,
                    "tokens_left": tokens_left,
                    "used_percent": used_pct,
                    "level": level,
                    "needs_compaction": used_pct >= 85,
                }
                yield json.dumps({"type": "context_usage", **usage}) + "\n"

            except Exception as exc:
                error_detail = str(exc)
                logger.exception("chat_stream_failed")
                yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
            finally:
                _append_thread_log(
                    active_settings,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "request_id": request_id,
                        "endpoint": "/api/chat/stream",
                        "request": {
                            "message": request.message,
                            "history_len": len(request.history),
                            "attachments_count": len(request.attachments),
                        },
                        "events_count": len(all_events),
                        "answer_len": len(final_answer),
                        "error": error_detail,
                    },
                )

        return StreamingResponse(_stream_generator(), media_type="application/x-ndjson")

    @app.post("/api/fea/results")
    async def fea_results(request: FEAResultsRequest):
        analyst = fea_sessions.get(request.session_id)
        if analyst is None:
            raise HTTPException(status_code=404, detail=f"FEA session '{request.session_id}' not found.")
        analyst.provide_results(request.results)
        return {"status": "ok"}

    @app.post("/api/fea/answer")
    async def fea_answer(request: FEAAnswerRequest):
        analyst = fea_sessions.get(request.session_id)
        if analyst is None:
            raise HTTPException(status_code=404, detail=f"FEA session '{request.session_id}' not found.")
        analyst.provide_answer(request.answer)
        return {"status": "ok"}

    @app.get("/api/tools")
    async def list_tools():
        return [
            {
                "tool_name": t["function"]["name"],
                "description": t["function"]["description"],
            }
            for t in TOOLS
        ]

    @app.post("/api/context-usage")
    async def get_context_usage(request: ChatRequest):
        """Return context usage snapshot for the frontend circle indicator.

        Called after each message exchange to update the UI.
        """
        messages = convert_frontend_history(request.history)
        if request.message:
            messages.append({"role": "user", "content": request.message})
        return context_usage_snapshot(
            messages, SYSTEM_PROMPT,
            context_window=active_settings.agent_context_window,
        )

    return app


app = create_app()
