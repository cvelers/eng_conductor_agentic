# Model Registry (Verified on February 28, 2026)

This file maps requested labels to exact model identifiers and thinking-mode settings for the benchmark run.

## Final mapping used in ECB-2026-v1

| Requested label | Exact model identifier used | Thinking mode to use | Notes |
|---|---|---|---|
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | Maximum / high-effort adaptive thinking | Anthropic page states Sonnet 4.6 is available as `claude-sonnet-4-6` and reports benchmarks with adaptive thinking and max reasoning effort. |
| ChatGPT 5.3 | `gpt-5.2-chat-latest` (Chat model) | `reasoning_effort = xhigh` | As of Feb 28, 2026, OpenAI publicly lists `GPT-5.3-Codex` release, while ChatGPT release notes explicitly mention GPT-5.2 rollout (Feb 16, 2026). No public ChatGPT-specific GPT-5.3 chat identifier was found in official OpenAI release notes. |
| Gemini 3.1 | `gemini-3.1-pro` (or `gemini-3.1-pro-preview` where required by endpoint) | `thinkingLevel = high` | Google Gemini docs list Gemini 3.1 Pro model code and thinking-level controls where default is high; use highest available thinking level. |
| Orchestrator | `eng_conductor_orchestrator` | `thinking_mode = thinking` | Local app schema and orchestrator engine support `standard`, `thinking`, `extended`; benchmark requires `thinking`. |

## Sources

- Anthropic Sonnet page: [anthropic.com/claude/sonnet](https://www.anthropic.com/claude/sonnet)
- OpenAI model release notes: [openai.com/index/model-release-notes](https://openai.com/index/model-release-notes/)
- OpenAI ChatGPT release notes: [help.openai.com ChatGPT release notes](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)
- Gemini models docs: [ai.google.dev Gemini models](https://ai.google.dev/gemini-api/docs/models)
- Gemini 3.1 Pro model card: [ai.google.dev/models/gemini-3.1-pro-preview](https://ai.google.dev/gemini-api/docs/models#gemini-3_1-pro-preview)
- Gemini thinking docs: [ai.google.dev thinking](https://ai.google.dev/gemini-api/docs/thinking)
- DeepMind Gemini Pro model page (thinking mode references): [deepmind.google/models/gemini/pro](https://deepmind.google/models/gemini/pro/)

## Local code references for orchestrator mode

- `/Users/ivancvetkovic/eng_conductor/backend/schemas.py`
- `/Users/ivancvetkovic/eng_conductor/backend/orchestrator/engine.py`
