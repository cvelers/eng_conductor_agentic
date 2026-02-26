from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


class Citation(BaseModel):
    doc_id: str
    clause_id: str
    clause_title: str
    pointer: str
    citation_address: str


class ToolTraceStep(BaseModel):
    tool_name: str
    status: str
    inputs: dict[str, Any]
    outputs: dict[str, Any] | None = None
    error: str | None = None


class RetrievalTraceStep(BaseModel):
    iteration: int
    query: str
    top_clause_ids: list[str]


class ChatResponse(BaseModel):
    answer: str
    supported: bool = True
    user_inputs: dict[str, Any]
    assumed_inputs: dict[str, Any]
    assumptions: list[str]
    sources: list[Citation]
    tool_trace: list[ToolTraceStep]
    retrieval_trace: list[RetrievalTraceStep]
    what_i_used: list[str]
