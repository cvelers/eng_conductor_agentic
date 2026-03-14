from __future__ import annotations

import logging
import json
from typing import Any

from backend.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
        default_reasoning_effort: str | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.default_reasoning_effort = default_reasoning_effort

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _coerce_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part)
        if content is None:
            return ""
        return str(content)

    def _is_gemini_compat(self) -> bool:
        return self.provider_name.strip().lower() == "gemini"

    @staticmethod
    def _coerce_tool_arguments(arguments: Any) -> str:
        if isinstance(arguments, str):
            return arguments
        try:
            return json.dumps(arguments if arguments is not None else {})
        except TypeError:
            return json.dumps({})

    def _normalize_messages_for_request(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert internal chat history into provider-safe OpenAI messages.

        The app stores extra bookkeeping fields on messages that are useful for
        validation and UI state but are not part of the OpenAI chat schema.
        Gemini's OpenAI-compatible endpoint is also stricter than OpenAI about
        tool-call history: assistant tool-call messages need non-empty content,
        and tool-result messages should include the called tool name.
        """
        normalized: list[dict[str, Any]] = []
        tool_name_by_id: dict[str, str] = {}

        for msg_index, message in enumerate(messages):
            role = str(message.get("role", "user") or "user")

            if role == "assistant" and message.get("tool_calls"):
                raw_tool_calls = message.get("tool_calls") or []
                tool_calls: list[dict[str, Any]] = []
                tool_names: list[str] = []
                for tc_index, raw_call in enumerate(raw_tool_calls):
                    if not isinstance(raw_call, dict):
                        continue
                    function = raw_call.get("function", {})
                    if not isinstance(function, dict):
                        function = {}
                    name = str(function.get("name", "") or "").strip()
                    if not name:
                        continue
                    call_id = str(raw_call.get("id") or f"call_{msg_index}_{tc_index}")
                    tool_name_by_id[call_id] = name
                    tool_names.append(name)
                    call_entry: dict[str, Any] = {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": self._coerce_tool_arguments(function.get("arguments", {})),
                        },
                    }
                    if "extra_content" in raw_call:
                        call_entry["extra_content"] = raw_call["extra_content"]
                    tool_calls.append(call_entry)

                if not tool_calls:
                    continue

                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                content = self._coerce_content(message.get("content"))
                if self._is_gemini_compat() and not content.strip():
                    names = ", ".join(tool_names[:4])
                    suffix = "..." if len(tool_names) > 4 else ""
                    content = f"[Assistant requested tool calls: {names}{suffix}]"
                elif content or "content" in message:
                    content = content
                if content:
                    assistant_message["content"] = content
                normalized.append(assistant_message)
                continue

            if role == "tool":
                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": str(message.get("tool_call_id", "") or ""),
                    "content": self._coerce_content(message.get("content")),
                }
                if self._is_gemini_compat():
                    if not tool_message["content"].strip():
                        tool_message["content"] = "[Tool result omitted.]"
                    tool_name = tool_name_by_id.get(tool_message["tool_call_id"])
                    if tool_name:
                        tool_message["name"] = tool_name
                normalized.append(tool_message)
                continue

            normalized.append({
                "role": role,
                "content": self._coerce_content(message.get("content")),
            })

        return normalized

    def _call_chat_completions(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4000,
        reasoning_effort: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | dict[str, Any]:
        if not self.available:
            raise RuntimeError(f"{self.provider_name} API key is not configured.")

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": self._normalize_messages_for_request(messages),
        }
        # For thinking models (Gemini 3.x, OpenAI o-series) this controls
        # how many tokens the model spends on internal reasoning.  "low"
        # keeps the thinking budget small so more of max_tokens is available
        # for the actual output — critical for lightweight calls like
        # classification and greetings.
        effective = reasoning_effort if reasoning_effort is not None else self.default_reasoning_effort
        if effective:
            payload["reasoning_effort"] = effective
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        import httpx

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
            try:
                resp.raise_for_status()
            except Exception as exc:
                detail = getattr(resp, "text", "") or getattr(resp, "content", "")
                detail_text = str(detail).strip()
                if detail_text:
                    raise RuntimeError(
                        f"{self.provider_name} chat/completions request failed: {detail_text[:600]}"
                    ) from exc
                raise
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"{self.provider_name} returned no choices.")

        choice = choices[0]
        finish_reason = str(choice.get("finish_reason", "")).strip().lower()
        message = choice.get("message", {}) or {}
        content = self._coerce_content(message.get("content", ""))
        tool_calls = message.get("tool_calls") or []

        # Log token usage so we can see the thinking-vs-output split.
        usage = data.get("usage", {})
        logger.info(
            "llm_call_complete",
            extra={
                "provider": self.provider_name,
                "model": self.model,
                "finish_reason": finish_reason,
                "max_tokens": max_tokens,
                "reasoning_effort": effective,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "content_len": len(content),
                "content_preview": content[:100],
                "tool_calls": len(tool_calls),
            },
        )

        if finish_reason == "length":
            logger.warning(
                "llm_output_truncated",
                extra={
                    "provider": self.provider_name,
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "content_preview": content[:120],
                },
            )

        if tool_calls or "malformed_function_call" in finish_reason:
            return {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
            }

        return content

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        reasoning_effort: str | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._call_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def generate_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 8000,
        reasoning_effort: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | dict[str, Any]:
        """Generate from a full messages list (multi-turn)."""
        return self._call_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )

    def generate_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 8000,
    ):
        """Stream tokens as they arrive. Yields text strings."""
        import json as _json

        if not self.available:
            raise RuntimeError(f"{self.provider_name} API key is not configured.")

        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": self._normalize_messages_for_request(messages),
            "stream": True,
        }
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        import httpx

        with httpx.Client(timeout=self.timeout_s) as client:
            with client.stream("POST", url, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = _json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue

    def generate_multimodal(
        self,
        *,
        system_prompt: str,
        content_parts: list[dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 4000,
        reasoning_effort: str | None = None,
    ) -> str:
        """Generate with multimodal content (text + images via OpenAI vision API)."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]
        return self._call_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
