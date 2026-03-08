#!/usr/bin/env python3
"""AutoBench self-improvement loop.

This script is meant to be run by Claude Code as a scheduled task.
It follows the autoresearch pattern:

1. Read program.md for strategy
2. Read recent results.tsv for history
3. Read last benchmark report to identify failures
4. Propose ONE focused code change
5. Git commit
6. Run benchmark: python -m benchmark.autobench.run_benchmark
7. Parse accuracy_pct
8. If improved → keep commit
9. If not → git reset --hard HEAD~1
10. Log to results.tsv
11. Repeat

IMPORTANT: This script is a TEMPLATE for the Claude Code scheduled task.
The actual loop is run by Claude Code (Opus 4.6) as the proposer — this
script provides the structure and the keep/discard logic.

To run manually:
    python -m benchmark.autobench.autobench_loop
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "autobench" / "results"
RESULTS_TSV = PROJECT_ROOT / "benchmark" / "autobench" / "results.tsv"


def get_current_commit() -> str:
    """Get current git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_last_accuracy() -> float | None:
    """Read the last accuracy_pct from results.tsv."""
    if not RESULTS_TSV.exists():
        return None
    lines = RESULTS_TSV.read_text().strip().splitlines()
    if len(lines) < 2:  # header + at least one row
        return None
    last = lines[-1].split("\t")
    try:
        return float(last[1])
    except (IndexError, ValueError):
        return None


def run_benchmark(task_filter: str | None = None) -> dict:
    """Run the benchmark and return the report."""
    cmd = [sys.executable, "-m", "benchmark.autobench.run_benchmark"]
    if task_filter:
        cmd.extend(["--track", task_filter])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"run_{timestamp}.json"
    cmd.extend(["--output", str(output_path)])

    print(f"Running benchmark: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour timeout
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if output_path.exists():
        return json.loads(output_path.read_text())
    return {}


def log_result(
    commit_hash: str,
    accuracy_pct: float,
    avg_latency_s: float,
    description: str,
) -> None:
    """Append a row to results.tsv."""
    timestamp = datetime.now(timezone.utc).isoformat()

    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit_hash\taccuracy_pct\tavg_latency_s\tdescription\ttimestamp\n")

    with RESULTS_TSV.open("a") as f:
        f.write(f"{commit_hash}\t{accuracy_pct}\t{avg_latency_s}\t{description}\t{timestamp}\n")


def discard_last_commit() -> None:
    """Revert the last commit."""
    subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )


def main() -> None:
    """Run one iteration of the autobench loop.

    This is the basic version — when run as a Claude Code scheduled task,
    the Claude agent handles steps 1-5 (reading program.md, analyzing
    failures, proposing changes, committing). This script handles the
    remaining steps (run benchmark, keep/discard, log).
    """
    print("=" * 60)
    print("AutoBench Loop — Running benchmark evaluation")
    print("=" * 60)

    previous_accuracy = get_last_accuracy()
    commit_before = get_current_commit()

    # Run benchmark
    report = run_benchmark()

    if not report:
        print("ERROR: Benchmark produced no results.")
        return

    accuracy_pct = report.get("accuracy_pct", 0.0)
    avg_latency = report.get("avg_latency_s", 0.0)
    commit_after = get_current_commit()

    print(f"\nResult: accuracy_pct={accuracy_pct}%  avg_latency={avg_latency}s")

    if previous_accuracy is not None:
        delta = accuracy_pct - previous_accuracy
        print(f"Previous: {previous_accuracy}%  Delta: {delta:+.2f}%")

        if delta < 0 and commit_after != commit_before:
            print("REGRESSION detected — discarding last commit")
            discard_last_commit()
            log_result(
                commit_after,
                accuracy_pct,
                avg_latency,
                f"DISCARDED (regression: {previous_accuracy}% -> {accuracy_pct}%)",
            )
            return
        elif delta >= 0:
            print("IMPROVEMENT or STABLE — keeping commit")
        else:
            print("No commit to discard")
    else:
        print("First run — establishing baseline")

    log_result(commit_after, accuracy_pct, avg_latency, "autobench run")


if __name__ == "__main__":
    main()
