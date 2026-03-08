# AutoBench Program — eng_conductor Self-Improvement

## Goal

Maximize `accuracy_pct` on the Eurocode knowledge benchmark (64 tasks across
4 tracks: numeric, clause_lookup, synthesis, behavioral_safety).

Current baseline (pre-fix): 39.3% (59/150 points on 15-question preliminary).
Target: 70%+ on the full 64-task benchmark.

## Scalar Metric

`accuracy_pct` — the percentage of total available score achieved across all
benchmark tasks. This is the ONLY metric that matters for keep/discard
decisions.

## Modifiable Files

You may modify files in these directories:

- `tools/mcp/*.py` — calculation tool implementations
- `backend/orchestrator/agent_loop.py` — task decomposition and composition
- `backend/orchestrator/core.py` — orchestrator logic, tool input resolution
- `backend/orchestrator/tool_validator.py` — output validation ranges
- `backend/orchestrator/sanity_checker.py` — cross-tool consistency checks
- `tools/tool_registry.json` — tool metadata and schemas

## Read-Only Files (NEVER modify)

- `benchmark/autobench/run_benchmark.py` — the fixed eval harness
- `benchmark/eurocode_knowledge/` — benchmark tasks and scoring scripts
- `backend/llm/` — LLM provider integrations
- `backend/config.py` — settings

## Strategy Priorities

1. **Fix remaining numerical tool errors** — each wrong tool output directly
   costs benchmark points. Check each tool against EC3 reference values.

2. **Improve tool chain routing** — ensure the decomposer selects the right
   tools and passes the right parameters for each benchmark question.

3. **Improve input resolution** — when the agent omits parameters (e.g.,
   `stress_type`, `buckling_axis`), ensure the orchestrator infers them
   correctly from the query context.

4. **Reduce false tool failures** — ensure error handling and retry logic
   catches recoverable errors (e.g., missing optional params).

5. **Improve clause retrieval** — better search queries yield better clause
   evidence, improving citation scores.

## Constraints

- Changes must not break existing passing tests
- Changes must maintain the tool JSON I/O contract (Pydantic models)
- Each iteration should make ONE focused change
- Commit message must describe what changed and why
- If accuracy drops, the commit is discarded (`git reset --hard HEAD~1`)

## One-Change-Per-Iteration Rule

Each autobench iteration should:
1. Identify ONE specific failure from the most recent benchmark run
2. Trace the root cause (wrong tool output, wrong tool selection, missing param)
3. Make the minimal code change to fix it
4. Test the specific task locally if possible
5. Run the full benchmark
6. Keep or discard based on accuracy_pct

## Logs

Results are appended to `benchmark/autobench/results.tsv`:
```
commit_hash	accuracy_pct	avg_latency_s	description	timestamp
```
