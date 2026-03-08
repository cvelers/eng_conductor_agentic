# Eurocode Knowledge Benchmark (ECB-2026-v1)

This package defines a reproducible benchmark to evaluate Eurocode knowledge and engineering reasoning across:

- `claude-sonnet-4-6` (max thinking)
- `gpt-5.2-chat-latest` (xhigh reasoning) as the closest officially published ChatGPT-family chat model as of **February 28, 2026** (`gpt-5.3-codex` is documented separately but is codex-focused)
- `gemini-3.1-pro` / `gemini-3.1-pro-preview` (thinking level high)
- `eng_conductor_orchestrator` (thinking mode: `thinking`)

The benchmark is EC3-centric and structured for scientific comparison with deterministic scoring where possible.

## What is included

- `tasks/eurocode_benchmark_v1.json`: 64 tasks across 4 tracks.
- `scripts/generate_benchmark_v1.py`: deterministic task and answer-key generator.
- `scripts/create_run_sheet.py`: generates response logging template + per-agent prompt packets.
- `scripts/score_benchmark.py`: computes per-task and aggregate scores, CI, latency, and efficiency metrics.
- `docs/protocol.md`: evaluation methodology and best practices.
- `docs/model_registry_2026-02-28.md`: exact model naming and thinking-mode mapping with web sources.
- `docs/judge_rubric.md`: human rubric for synthesis tasks.

## Tracks

- `numeric` (32 tasks): deterministic calculator-grounded tasks with strict answer keys.
- `clause_lookup` (16 tasks): citation and clause-identification accuracy across EC3 parts.
- `synthesis` (8 tasks): engineering workflow quality, scored with human rubric + auto citation checks.
- `behavioral_safety` (8 tasks): hallucination resistance and safe handling of insufficient/unsafe prompts.

## Quick start

1. Generate/refresh benchmark task file:

```bash
.venv/bin/python benchmark/eurocode_knowledge/scripts/generate_benchmark_v1.py
```

2. Create run sheets and prompt packets:

```bash
.venv/bin/python benchmark/eurocode_knowledge/scripts/create_run_sheet.py
```

3. Run each model on all tasks and fill `benchmark/eurocode_knowledge/templates/responses_template.csv`.

4. (Optional) Fill human rubric for synthesis tasks in `benchmark/eurocode_knowledge/templates/human_scores_template.csv`.

5. Score benchmark:

```bash
.venv/bin/python benchmark/eurocode_knowledge/scripts/score_benchmark.py \
  --responses benchmark/eurocode_knowledge/templates/responses_template.csv \
  --human-scores benchmark/eurocode_knowledge/templates/human_scores_template.csv \
  --out-dir benchmark/eurocode_knowledge/output
```

Outputs:

- `benchmark/eurocode_knowledge/output/per_task_scores.csv`
- `benchmark/eurocode_knowledge/output/summary_scores.csv`
- `benchmark/eurocode_knowledge/output/summary_scores.json`

## Required model response format

Use this JSON schema in all model runs:

```json
{
  "task_id": "NUM-001",
  "final_answer": "...",
  "citations": [
    {"standard": "EN 1993-1-1", "clause": "6.2.5"}
  ],
  "results": {"M_Rd_kNm": 222.94},
  "assumptions": ["..."],
  "needs_more_info": false,
  "clarifying_questions": []
}
```
