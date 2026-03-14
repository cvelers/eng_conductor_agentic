from __future__ import annotations

import importlib
import sys
from typing import Any

class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.last_json: dict[str, Any] | None = None

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, headers: dict[str, Any], json: dict[str, Any]) -> _FakeResponse:
        self.last_json = json
        return _FakeResponse(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "todo_write",
                                        "arguments": '{"todos":[{"id":"model","text":"Build model","status":"in_progress"}]}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )


def test_openai_compat_provider_returns_tool_calls(monkeypatch) -> None:
    fake_client = _FakeClient()
    fake_httpx = type("FakeHttpx", (), {"Client": lambda *args, **kwargs: fake_client})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    openai_compat = importlib.import_module("backend.llm.openai_compat")
    OpenAICompatProvider = openai_compat.OpenAICompatProvider

    provider = OpenAICompatProvider(
        provider_name="test",
        base_url="https://example.test",
        api_key="key",
        model="model",
    )

    result = provider.generate_messages(
        messages=[{"role": "system", "content": "You are a tool caller."}],
        tools=[{"type": "function", "function": {"name": "todo_write", "parameters": {"type": "object"}}}],
    )

    assert isinstance(result, dict)
    assert result["tool_calls"][0]["function"]["name"] == "todo_write"
    assert fake_client.last_json is not None
    assert fake_client.last_json["tools"][0]["function"]["name"] == "todo_write"
    assert fake_client.last_json["tool_choice"] == "auto"


def test_openai_compat_provider_normalizes_gemini_tool_history(monkeypatch) -> None:
    fake_client = _FakeClient()
    fake_httpx = type("FakeHttpx", (), {"Client": lambda *args, **kwargs: fake_client})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    openai_compat = importlib.import_module("backend.llm.openai_compat")
    OpenAICompatProvider = openai_compat.OpenAICompatProvider

    provider = OpenAICompatProvider(
        provider_name="gemini",
        base_url="https://example.test",
        api_key="key",
        model="model",
    )

    provider.generate_messages(
        messages=[
            {"role": "system", "content": "You are a tool caller."},
            {"role": "user", "content": "Build the model."},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "fea_add_nodes", "arguments": {"nodes": [{"id": "1"}]}},
                    },
                    {
                        "id": "call_explicit",
                        "type": "function",
                        "function": {"name": "todo_write", "arguments": '{"todos":[]}'},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2_0",
                "content": "Added nodes",
                "validator_content": "verbose result that should not be forwarded",
            },
            {
                "role": "tool",
                "tool_call_id": "call_explicit",
                "content": "",
            },
        ],
        tools=[{"type": "function", "function": {"name": "todo_write", "parameters": {"type": "object"}}}],
    )

    assert fake_client.last_json is not None
    outbound_messages = fake_client.last_json["messages"]

    assistant_message = outbound_messages[2]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"].startswith("[Assistant requested tool calls:")
    assert assistant_message["tool_calls"][0]["id"] == "call_2_0"
    assert assistant_message["tool_calls"][0]["function"]["arguments"] == '{"nodes": [{"id": "1"}]}'
    assert assistant_message["tool_calls"][1]["id"] == "call_explicit"

    first_tool_message = outbound_messages[3]
    assert first_tool_message == {
        "role": "tool",
        "tool_call_id": "call_2_0",
        "content": "Added nodes",
        "name": "fea_add_nodes",
    }

    second_tool_message = outbound_messages[4]
    assert second_tool_message == {
        "role": "tool",
        "tool_call_id": "call_explicit",
        "content": "[Tool result omitted.]",
        "name": "todo_write",
    }


def test_openai_compat_provider_preserves_malformed_function_finish_reason(monkeypatch) -> None:
    class _MalformedClient(_FakeClient):
        def post(self, url: str, headers: dict[str, Any], json: dict[str, Any]) -> _FakeResponse:
            self.last_json = json
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "function_call_filter: malformed_function_call",
                            "message": {"content": None},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
                }
            )

    fake_client = _MalformedClient()
    fake_httpx = type("FakeHttpx", (), {"Client": lambda *args, **kwargs: fake_client})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    openai_compat = importlib.import_module("backend.llm.openai_compat")
    OpenAICompatProvider = openai_compat.OpenAICompatProvider

    provider = OpenAICompatProvider(
        provider_name="gemini",
        base_url="https://example.test",
        api_key="key",
        model="model",
    )

    result = provider.generate_messages(
        messages=[{"role": "system", "content": "You are a tool caller."}],
        tools=[{"type": "function", "function": {"name": "todo_write", "parameters": {"type": "object"}}}],
    )

    assert isinstance(result, dict)
    assert result["tool_calls"] == []
    assert result["finish_reason"] == "function_call_filter: malformed_function_call"
