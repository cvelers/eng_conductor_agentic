#!/usr/bin/env python3
"""AutoBench — Fixed eval harness for eng_conductor.

Runs the eng_conductor agent loop against benchmark tasks and scores results.
THIS FILE IS THE FIXED EVALUATOR — never modify it in the autobench loop.

Usage:
    python -m benchmark.autobench.run_benchmark
    python -m benchmark.autobench.run_benchmark --tasks NUM-001 NUM-002
    python -m benchmark.autobench.run_benchmark --track numeric
    python -m benchmark.autobench.run_benchmark --output results/run_001.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from backend.config import Settings
from backend.llm.factory import get_orchestrator_provider, get_search_provider
from backend.logging_config import configure_logging
from backend.orchestrator.agent_loop import AgentLoop
from backend.orchestrator.core import CentralIntelligenceOrchestrator
from backend.registries.document_registry import load_all_clauses, load_document_registry
from backend.registries.tool_registry import load_tool_registry
from backend.retrieval.agentic_search import AgenticRetriever
from backend.schemas import ChatResponse
from backend.tools.runner import MCPToolRunner

logger = logging.getLogger(__name__)

BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "eurocode_knowledge" / "tasks" / "eurocode_benchmark_v1.json"
PRELIMINARY_PATH = PROJECT_ROOT / "benchmark" / "eurocode_knowledge" / "tasks" / "eurocode_preliminary_v1.json"
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "autobench" / "results"


def _build_agent_loop(settings: Settings) -> AgentLoop:
    """Construct an AgentLoop exactly as app.py does."""
    doc_registry = load_document_registry(settings.resolved_document_registry_path)
    clauses = load_all_clauses(settings.project_root, doc_registry)
    tool_registry = load_tool_registry(settings.resolved_tool_registry_path)

    search_provider = get_search_provider(settings)
    orchestrator_provider = get_orchestrator_provider(settings)

    retriever = AgenticRetriever(
        settings=settings,
        search_provider=search_provider,
        clauses=clauses,
    )

    tool_runner = MCPToolRunner(project_root=settings.project_root, registry=tool_registry)

    orchestrator = CentralIntelligenceOrchestrator(
        settings=settings,
        orchestrator_llm=orchestrator_provider,
        retriever=retriever,
        tool_runner=tool_runner,
        tool_registry=tool_registry,
        document_registry=doc_registry,
        clauses=clauses,
    )

    return AgentLoop(orchestrator=orchestrator, settings=settings)


def _extract_response_json(response: ChatResponse) -> dict[str, Any]:
    """Build the JSON payload in the format score_benchmark.py expects."""
    # Collect results from tool trace outputs
    results: dict[str, Any] = {}
    for step in response.tool_trace:
        if step.status == "ok" and step.outputs:
            results.update(step.outputs)

    # Collect citations
    citations: list[dict[str, str]] = []
    for source in response.sources:
        citations.append({
            "standard": source.doc_id,
            "clause": source.clause_id,
        })

    return {
        "final_answer": response.answer,
        "results": results,
        "citations": citations,
        "assumptions": response.assumptions,
        "needs_more_info": False,
        "clarifying_questions": [],
    }


def run_benchmark(
    agent_loop: AgentLoop,
    tasks: list[dict[str, Any]],
    agent_id: str = "eng_conductor",
) -> list[dict[str, Any]]:
    """Run all tasks through the agent and return response records."""
    records: list[dict[str, Any]] = []

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        prompt = task["prompt"]
        print(f"[{i+1}/{len(tasks)}] Running {task_id}...", end=" ", flush=True)

        t0 = time.monotonic()
        try:
            response = agent_loop.run(prompt, thinking_mode="thinking")
            latency_s = time.monotonic() - t0

            response_json = _extract_response_json(response)
            response_text = response.answer

            print(f"OK ({latency_s:.1f}s)")
        except Exception as exc:
            latency_s = time.monotonic() - t0
            response_json = {}
            response_text = f"ERROR: {exc}"
            print(f"FAIL ({latency_s:.1f}s): {exc}")

        records.append({
            "agent_id": agent_id,
            "model_name": "eng_conductor",
            "task_id": task_id,
            "response_text": response_text,
            "response_json": response_json,
            "latency_s": round(latency_s, 2),
        })

    return records


def score_records(
    tasks: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score records using the benchmark scoring logic inline.

    We import and reuse the scoring functions from score_benchmark.py
    to ensure consistency.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "benchmark" / "eurocode_knowledge" / "scripts"))
    from score_benchmark import (
        ResponseRecord,
        _score_numeric,
        _score_clause_lookup,
        _score_behavioral,
        _score_synthesis,
    )

    task_lookup = {t["task_id"]: t for t in tasks}
    scores: list[dict[str, Any]] = []
    total_score = 0.0

    for rec in records:
        task = task_lookup.get(rec["task_id"])
        if not task:
            continue

        rr = ResponseRecord(
            agent_id=rec["agent_id"],
            model_name=rec["model_name"],
            task_id=rec["task_id"],
            response_text=rec["response_text"],
            response_json=rec.get("response_json"),
            latency_s=rec.get("latency_s"),
            cost_usd=None,
        )

        task_type = task.get("task_type", task.get("track", ""))
        if task_type == "numeric":
            result = _score_numeric(task, rr)
        elif task_type == "clause_lookup":
            result = _score_clause_lookup(task, rr)
        elif task_type == "behavioral_safety":
            result = _score_behavioral(task, rr)
        elif task_type == "synthesis":
            result = _score_synthesis(task, rr, [])
        else:
            result = {"task_score": 0.0}

        task_score = result.get("task_score", 0.0)
        total_score += task_score

        scores.append({
            "task_id": rec["task_id"],
            "track": task.get("track", ""),
            "task_score": round(task_score, 4),
            "latency_s": rec.get("latency_s"),
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in result.items() if k != "task_score"},
        })

    n = len(scores) or 1
    accuracy_pct = round(100.0 * total_score / n, 2)
    avg_latency = round(
        sum(s.get("latency_s", 0) or 0 for s in scores) / n, 1
    )

    return {
        "accuracy_pct": accuracy_pct,
        "total_score": round(total_score, 4),
        "n_tasks": len(scores),
        "avg_latency_s": avg_latency,
        "per_task": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoBench eval harness")
    parser.add_argument("--tasks", nargs="*", help="Specific task IDs to run")
    parser.add_argument("--track", help="Run only tasks from this track")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--agent-id", default="eng_conductor")
    parser.add_argument(
        "--benchmark-file",
        help="Path to benchmark JSON file (default: eurocode_benchmark_v1.json)",
    )
    parser.add_argument(
        "--preliminary", action="store_true", default=True,
        help="Use the preliminary benchmark (default). Pass --no-preliminary for the full 64-task one.",
    )
    parser.add_argument(
        "--no-preliminary", action="store_false", dest="preliminary",
        help="Use the full 64-task benchmark instead of the preliminary one.",
    )
    args = parser.parse_args()

    settings = Settings.load()
    configure_logging(settings.log_level)

    # Load benchmark tasks
    if args.benchmark_file:
        bench_path = Path(args.benchmark_file)
    elif args.preliminary:
        bench_path = PRELIMINARY_PATH
    else:
        bench_path = BENCHMARK_PATH
    benchmark = json.loads(bench_path.read_text(encoding="utf-8"))
    all_tasks = benchmark["tasks"]

    # Filter tasks
    if args.tasks:
        all_tasks = [t for t in all_tasks if t["task_id"] in args.tasks]
    elif args.track:
        all_tasks = [t for t in all_tasks if t.get("track") == args.track]

    if not all_tasks:
        print("No tasks matched filters.")
        sys.exit(1)

    print(f"AutoBench: {len(all_tasks)} tasks to run")
    print("=" * 60)

    # Build agent
    agent_loop = _build_agent_loop(settings)

    # Run
    records = run_benchmark(agent_loop, all_tasks, agent_id=args.agent_id)

    # Score
    scoring = score_records(all_tasks, records)

    # Save
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"run_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": timestamp,
        "agent_id": args.agent_id,
        "accuracy_pct": scoring["accuracy_pct"],
        "avg_latency_s": scoring["avg_latency_s"],
        "n_tasks": scoring["n_tasks"],
        "total_score": scoring["total_score"],
        "records": records,
        "scoring": scoring,
    }
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Print summary
    print("=" * 60)
    print(f"accuracy_pct={scoring['accuracy_pct']}  "
          f"avg_latency={scoring['avg_latency_s']}s  "
          f"tasks={scoring['n_tasks']}")
    print(f"Report saved: {output_path}")

    # Track-level breakdown
    track_scores: dict[str, list[float]] = {}
    for s in scoring["per_task"]:
        track = s.get("track", "unknown")
        track_scores.setdefault(track, []).append(s["task_score"])
    for track, vals in sorted(track_scores.items()):
        avg = sum(vals) / len(vals) * 100
        print(f"  {track}: {avg:.1f}% ({len(vals)} tasks)")


if __name__ == "__main__":
    main()
