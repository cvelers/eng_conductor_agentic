from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.auth import create_auth_router, require_auth
from backend.config import Settings
from backend.llm.factory import get_orchestrator_provider, get_search_provider, get_tool_writer_provider
from backend.logging_config import configure_logging
from backend.orchestrator.engine import CentralIntelligenceOrchestrator
from backend.registries.document_registry import load_all_clauses, load_document_registry
from backend.registries.tool_registry import load_tool_registry
from backend.retrieval.agentic_search import AgenticRetriever
from backend.schemas import ChatRequest, ChatResponse
from backend.tools.runner import MCPToolRunner
from backend.tools.writer import ToolWriter

logger = logging.getLogger(__name__)


def _chunk_text(value: str, size: int = 36) -> list[str]:
    if not value:
        return []
    return [value[idx : idx + size] for idx in range(0, len(value), size)]


class ToolGenerateRequest(BaseModel):
    description: str


def create_app(settings: Settings | None = None) -> FastAPI:
    load_dotenv()
    active_settings = settings or Settings.load()

    configure_logging(active_settings.log_level)

    doc_registry = load_document_registry(active_settings.resolved_document_registry_path)
    clauses = load_all_clauses(active_settings.project_root, doc_registry)
    tool_registry = load_tool_registry(active_settings.resolved_tool_registry_path)

    search_provider = get_search_provider(active_settings)
    orchestrator_provider = get_orchestrator_provider(active_settings)

    retriever = AgenticRetriever(
        settings=active_settings,
        search_provider=search_provider,
        clauses=clauses,
    )

    tool_runner = MCPToolRunner(project_root=active_settings.project_root, registry=tool_registry)

    orchestrator = CentralIntelligenceOrchestrator(
        settings=active_settings,
        orchestrator_llm=orchestrator_provider,
        retriever=retriever,
        tool_runner=tool_runner,
        tool_registry=tool_registry,
    )

    tool_writer_provider = get_tool_writer_provider(active_settings)

    tool_writer = ToolWriter(
        llm=tool_writer_provider,
        retriever=retriever,
        tool_registry=orchestrator.tool_registry,
        project_root=active_settings.project_root,
    )

    app = FastAPI(title="Eurocodes Chatbot", version="0.3.0")
    app.state.orchestrator = orchestrator
    app.state.tool_writer = tool_writer
    app.state.settings = active_settings

    app.include_router(create_auth_router(active_settings))
    auth_dep = require_auth(active_settings)

    frontend_dir = active_settings.project_root / "frontend"
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            response = orchestrator.run(request.message, history=request.history)
            return response
        except Exception as exc:
            logger.exception("chat_failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:
        async def event_generator():
            final_response: ChatResponse | None = None
            try:
                for event_type, payload in orchestrator.run_stream(request.message, history=request.history):
                    if event_type == "machine":
                        yield json.dumps({"type": "machine", **payload}) + "\n"
                        await asyncio.sleep(0.005)
                    elif event_type == "response":
                        final_response = payload
            except Exception as exc:
                logger.exception("chat_stream_failed")
                yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
                return

            if final_response is None:
                yield json.dumps({"type": "error", "detail": "No response generated."}) + "\n"
                return

            for piece in _chunk_text(final_response.answer, size=32):
                yield json.dumps({"type": "delta", "delta": piece}) + "\n"
                await asyncio.sleep(0.006)
            yield json.dumps({"type": "final", "response": final_response.model_dump()}) + "\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    @app.post("/api/tools/generate")
    async def generate_tool(request: ToolGenerateRequest):
        try:
            result = tool_writer.generate(request.description)
            return result
        except Exception as exc:
            logger.exception("tool_generation_failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/tools")
    async def list_tools():
        return [
            {
                "tool_name": entry.tool_name,
                "description": entry.description,
                "tags": entry.tags,
            }
            for entry in tool_registry
        ]

    return app


app = create_app()
