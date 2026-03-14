# Eurocodes Chatbot — EC3 Grounded Intelligence

A single-page civil and structural engineering assistant with a FastAPI backend, a vanilla JS chat UI, Eurocode clause retrieval, direct engineering-calculation tools, and a separate browser-backed FEA analysis mode. The assistant stays scoped to civil/structural engineering questions and direct follow-ups to prior in-scope answers.

## Architecture

```
┌────────────┐    ┌──────────────────────────────────────────┐
│  Chat UI   │───▶│  FastAPI App                             │
│  (vanilla  │    │  /api/chat + /api/chat/stream            │
│   JS)      │◀───│                                          │
└────────────┘    └──────────────────────────────────────────┘
                            │
                            ▼
                ┌─────────────────────────────┐
                │  Agent Loop                 │
                │  Scope gate → Plan → Tools │
                │  → Grounding → Answer      │
                └─────────────────────────────┘
                   │                    │
                   ▼                    ▼
        ┌──────────────────────┐   ┌─────────────────────────┐
        │ Eurocode Retrieval   │   │ Engineering Tools       │
        │ BM25F + optional LLM │   │ eurocodepy + local math │
        │ sufficiency passes   │   │ + utility tool handlers │
        └──────────────────────┘   └─────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ EC3 Clause Database  │
        └──────────────────────┘

Separate path for FEA requests:
Chat UI ↔ FastAPI ↔ FEA Analyst ↔ Browser solver/viewer
```

**Runtime flow:** request intake → civil-engineering scope check → retrieval/tool calls → grounding validation → final answer

## Features

- **Civil-engineering scope gate** — the assistant refuses out-of-scope requests and only handles civil/structural engineering questions or direct in-scope follow-ups
- **Agent loop orchestration** — tool-calling chat loop with planning, continuation memory, grounding checks, and streamed UI events
- **Eurocode retrieval** — lexical search over local EC3 data with optional LLM-guided sufficiency/refinement
- **Direct engineering calculations** — eurocodepy-backed EC3 checks plus local calculators and math utilities
- **Separate FEA mode** — a dedicated analyst builds structural models and drives a browser-side solver/viewer
- **Multimodal-ready technical inputs** — photos, screenshots, and attached technical files are part of the chat input surface for multimodal model workflows; the same civil-engineering scope rules still apply
- **Grounded responses** — answers are assembled from retrieved clauses and tool outputs
- **Streaming UI** — flow graph and answer tokens stream incrementally to the frontend
- **Markdown + math rendering** — formatted answers with KaTeX-rendered equations

## Tool Surface

The chat agent currently exposes several tool categories:

- **Eurocode retrieval tools**: `eurocode_search`, `read_clause`
- **Engineering calculation tools**: `search_engineering_tools`, `engineering_calculator`, `math_calculator`
- **Conversation control tools**: `todo_write`, `ask_user`
- **General utility tools**: web/file/system helpers used by the current agent runtime

Engineering calculations are primarily backed by the registry in `backend/eurocodepy/registry.py`, which currently includes EC3 section checks, LTB, flexural buckling, Euler critical force, profile lookups, steel-grade lookups, and bolt-property lookups.

## Quick Start

```bash
# 1. Create venv and install deps
make venv

# 2. Configure LLM keys (copy and edit)
cp .env.example .env
# Edit .env with your API keys

# 3. Run
make run
# Open http://localhost:8000

# 4. Test
make test
```

## Configuration (.env)

```ini
# Orchestrator / main chat model (default: Gemini-compatible)
ORCHESTRATOR_PROVIDER=gemini
ORCHESTRATOR_MODEL=gemini-3.1-flash-lite-preview
ORCHESTRATOR_API_KEY=your-key-here

# Retrieval / search model
SEARCH_PROVIDER=gemini
SEARCH_MODEL=gemini-3.1-flash-lite-preview
SEARCH_API_KEY=your-key-here

# Optional validator model
VALIDATOR_API_KEY=your-key-here

# Feature flags
AGENTIC_SEARCH_ENABLED=true
RECURSIVE_RETRIEVAL_ENABLED=false
EMBEDDINGS_ENABLED=false
```

## Example Prompts

**Conceptual:**
- "Explain EC3 section classification rules for I-sections"
- "What are the buckling curves and when is each used?"

**Computational:**
- "Given IPE300, S355, what is the bending resistance?"
- "Simply supported beam, 6m span, 15 kN/m UDL — max moment and deflection?"
- "Calculate bolt shear resistance for 4× M20 grade 8.8 bolts"
- "Column buckling check: IPE300, S355, 5m length, pinned-pinned"
- "Fillet weld resistance: 5mm throat, 200mm length, S355"

**Attachments / multimodal:**
- Upload a photo, screenshot, or technical document together with the prompt
- Intended use includes engineering drawings, calculation screenshots, detail photos, and similar technical inputs
- The same scope gate still applies: non-civil-engineering content should be refused rather than answered

## Project Structure

```
eng_conductor/
├── backend/
│   ├── app.py                      # FastAPI application
│   ├── config.py                   # Runtime settings + cognitive config
│   ├── agent/                      # Agent loop, prompt, context, tools
│   ├── retrieval/agentic_search.py # Clause retrieval engine
│   ├── eurocodepy/                 # Engineering-tool registry/search/dispatch
│   ├── orchestrator/               # FEA analyst + FEA routing/tools
│   ├── llm/                        # LLM provider abstraction
│   ├── registries/                 # Document registry loading
│   └── utils/                      # Parsing, citations, JSON helpers
├── frontend/
│   ├── index.html                  # Single-page chat UI
│   ├── app.js                      # Chat state, streaming, attachments
│   ├── fea/                        # Solver, elements, worker
│   ├── viewer/                     # 3D viewer / result visualization
│   └── styles.css                  # Frontend styling
├── tools/
│   └── mcp/                        # Local calculation modules
├── data/
│   ├── document_registry.json
│   └── ec3/                        # EC3 clause data (JSON)
├── cognitive_config.json           # Model/runtime tuning
└── tests/
```
