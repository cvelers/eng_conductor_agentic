#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResponseRecord:
    agent_id: str
    model_name: str
    task_id: str
    response_text: str
    response_json: dict[str, Any] | None
    latency_s: float | None
    cost_usd: float | None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_json_loose(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Try extracting the first JSON object block.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _normalize_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _extract_results(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    results = payload.get("results")
    if isinstance(results, dict):
        return results
    return {}


def _extract_text(payload: dict[str, Any] | None, raw_text: str) -> str:
    if payload and isinstance(payload.get("final_answer"), str):
        return payload["final_answer"]
    return raw_text


def _collect_clause_tokens(payload: dict[str, Any] | None, raw_text: str) -> str:
    parts: list[str] = [raw_text]
    if payload:
        citations = payload.get("citations")
        if isinstance(citations, list):
            for item in citations:
                if isinstance(item, dict):
                    clause = str(item.get("clause", ""))
                    standard = str(item.get("standard", ""))
                    parts.append(f"{standard} {clause}")
                else:
                    parts.append(str(item))
        results = payload.get("results")
        if isinstance(results, dict):
            parts.append(json.dumps(results))
    return "\n".join(parts)


def _score_numeric(task: dict[str, Any], record: ResponseRecord) -> dict[str, float]:
    payload = record.response_json
    response_text = record.response_text
    expected = task["expected"]
    tolerance = expected.get("tolerance", {})
    rel_tol = float(tolerance.get("relative", 0.02))
    abs_tol = float(tolerance.get("absolute", 0.05))

    targets: dict[str, Any] = expected.get("targets", {})
    results = _extract_results(payload)

    hit_count = 0
    for key, target in targets.items():
        pred = results.get(key)

        if isinstance(target, bool):
            if isinstance(pred, bool) and pred == target:
                hit_count += 1
            elif isinstance(pred, str) and pred.strip().lower() in {"true", "false"}:
                pred_bool = pred.strip().lower() == "true"
                if pred_bool == target:
                    hit_count += 1
            continue

        t_num = _normalize_float(target)
        p_num = _normalize_float(pred)

        if t_num is None:
            if str(pred).strip().lower() == str(target).strip().lower():
                hit_count += 1
            continue

        if p_num is None:
            continue

        tol = max(abs_tol, rel_tol * abs(t_num))
        if abs(p_num - t_num) <= tol:
            hit_count += 1

    numeric_acc = hit_count / max(len(targets), 1)

    required_clause_ids: list[str] = expected.get("required_clause_ids", [])
    clause_space = _collect_clause_tokens(payload, response_text).lower()
    clause_hits = 0
    for clause_id in required_clause_ids:
        if str(clause_id).strip().lower() in clause_space:
            clause_hits += 1
    clause_cov = clause_hits / max(len(required_clause_ids), 1)

    scoring = task.get("scoring", {})
    n_w = float(scoring.get("numeric_component", 0.85))
    c_w = float(scoring.get("citation_component", 0.15))
    score = n_w * numeric_acc + c_w * clause_cov

    return {
        "task_score": score,
        "numeric_accuracy": numeric_acc,
        "citation_coverage": clause_cov,
    }


def _score_clause_lookup(task: dict[str, Any], record: ResponseRecord) -> dict[str, float]:
    payload = record.response_json
    response_text = record.response_text
    expected = task["expected"]

    required_clauses: list[str] = expected.get("required_clause_ids", [])
    clause_space = _collect_clause_tokens(payload, response_text).lower()
    clause_hits = sum(1 for c in required_clauses if str(c).strip().lower() in clause_space)
    clause_score = clause_hits / max(len(required_clauses), 1)

    required_keywords: list[str] = expected.get("required_keywords", [])
    text = _extract_text(payload, response_text).lower()
    kw_hits = sum(1 for kw in required_keywords if str(kw).strip().lower() in text)
    kw_score = kw_hits / max(len(required_keywords), 1)

    scoring = task.get("scoring", {})
    c_w = float(scoring.get("clause_component", 0.7))
    p_w = float(scoring.get("paraphrase_component", 0.3))
    score = c_w * clause_score + p_w * kw_score

    return {
        "task_score": score,
        "clause_accuracy": clause_score,
        "keyword_coverage": kw_score,
    }


def _score_behavioral(task: dict[str, Any], record: ResponseRecord) -> dict[str, float]:
    text = record.response_text
    if record.response_json:
        text = f"{text}\n{json.dumps(record.response_json)}"

    expected = task["expected"]
    required_regex: list[str] = expected.get("required_regex", [])
    forbidden_regex: list[str] = expected.get("forbidden_regex", [])

    required_hits = 0
    for pattern in required_regex:
        if re.search(pattern, text, flags=re.IGNORECASE):
            required_hits += 1

    forbidden_hits = 0
    for pattern in forbidden_regex:
        if re.search(pattern, text, flags=re.IGNORECASE):
            forbidden_hits += 1

    req_score = required_hits / max(len(required_regex), 1)
    forbidden_penalty = 1.0 if forbidden_hits > 0 else 0.0
    score = req_score * (1.0 - forbidden_penalty)

    return {
        "task_score": score,
        "required_behavior_score": req_score,
        "forbidden_hit": 1.0 if forbidden_hits > 0 else 0.0,
    }


def _human_score_lookup(human_rows: list[dict[str, Any]], agent_id: str, task_id: str) -> float | None:
    candidates = [
        row
        for row in human_rows
        if str(row.get("agent_id", "")).strip() == agent_id
        and str(row.get("task_id", "")).strip() == task_id
    ]
    if not candidates:
        return None

    row = candidates[0]
    keys = [
        "completeness",
        "actionability",
        "soundness",
        "safety",
        "citation_fidelity",
    ]
    vals: list[float] = []
    for key in keys:
        value = row.get(key)
        num = _normalize_float(value)
        if num is None:
            continue
        # Assume rubric 1..5
        vals.append(max(0.0, min(1.0, (num - 1.0) / 4.0)))

    if not vals:
        return None
    return sum(vals) / len(vals)


def _score_synthesis(
    task: dict[str, Any],
    record: ResponseRecord,
    human_rows: list[dict[str, Any]],
) -> dict[str, float]:
    payload = record.response_json
    response_text = record.response_text
    expected = task["expected"]
    required_clauses: list[str] = expected.get("required_clause_ids", [])

    clause_space = _collect_clause_tokens(payload, response_text).lower()
    clause_hits = sum(1 for c in required_clauses if str(c).strip().lower() in clause_space)
    auto_citation = clause_hits / max(len(required_clauses), 1)

    human = _human_score_lookup(human_rows, record.agent_id, record.task_id)

    scoring = task.get("scoring", {})
    auto_w = float(scoring.get("auto_weight", 0.2))
    human_w = float(scoring.get("human_weight", 0.8))

    if human is None:
        task_score = auto_citation
        human_available = 0.0
    else:
        task_score = auto_w * auto_citation + human_w * human
        human_available = 1.0

    return {
        "task_score": task_score,
        "auto_citation": auto_citation,
        "human_quality": human if human is not None else float("nan"),
        "human_available": human_available,
    }


def _load_responses(path: Path) -> list[ResponseRecord]:
    if path.suffix.lower() == ".jsonl":
        raw_rows = _load_jsonl(path)
    elif path.suffix.lower() == ".json":
        raw = _load_json(path)
        if isinstance(raw, list):
            raw_rows = raw
        else:
            raise ValueError("responses .json must be a list.")
    elif path.suffix.lower() == ".csv":
        raw_rows = _load_csv(path)
    else:
        raise ValueError("Unsupported responses format. Use .jsonl/.json/.csv")

    records: list[ResponseRecord] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue

        agent_id = str(row.get("agent_id", "")).strip()
        model_name = str(row.get("model_name", "")).strip()
        task_id = str(row.get("task_id", "")).strip()
        response_text = str(row.get("response_text", ""))

        response_json_obj: dict[str, Any] | None = None
        if isinstance(row.get("response_json"), dict):
            response_json_obj = row["response_json"]
        elif isinstance(row.get("response_json"), str) and row.get("response_json"):
            response_json_obj = _parse_json_loose(str(row.get("response_json")))

        if response_json_obj is None:
            response_json_obj = _parse_json_loose(response_text)

        latency_s = _normalize_float(row.get("latency_s"))
        cost_usd = _normalize_float(row.get("cost_usd"))

        if not agent_id or not task_id:
            continue

        records.append(
            ResponseRecord(
                agent_id=agent_id,
                model_name=model_name,
                task_id=task_id,
                response_text=response_text,
                response_json=response_json_obj,
                latency_s=latency_s,
                cost_usd=cost_usd,
            )
        )

    return records


def _bootstrap_ci(values: list[float], n_bootstrap: int = 2000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])

    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [random.choice(values) for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo_idx = int(0.025 * len(means))
    hi_idx = int(0.975 * len(means))
    return (means[lo_idx], means[min(hi_idx, len(means) - 1)])


def _safe_p90(values: list[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = int(math.ceil(0.9 * len(ordered))) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default="benchmark/eurocode_knowledge/tasks/eurocode_benchmark_v1.json",
        help="Benchmark task JSON file.",
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="Responses file (.jsonl/.json/.csv).",
    )
    parser.add_argument(
        "--human-scores",
        default="",
        help="Optional CSV with human rubric rows.",
    )
    parser.add_argument(
        "--out-dir",
        default="benchmark/eurocode_knowledge/output",
        help="Output directory for score reports.",
    )
    args = parser.parse_args()

    benchmark = _load_json(Path(args.benchmark))
    tasks = benchmark.get("tasks", [])
    task_by_id = {task["task_id"]: task for task in tasks if isinstance(task, dict)}

    responses = _load_responses(Path(args.responses))

    human_rows: list[dict[str, Any]] = []
    if args.human_scores:
        human_path = Path(args.human_scores)
        if human_path.exists():
            human_rows = _load_csv(human_path)

    per_task_rows: list[dict[str, Any]] = []

    for rec in responses:
        task = task_by_id.get(rec.task_id)
        if task is None:
            continue

        track = task.get("track", "")
        score_parts: dict[str, float]
        if track == "numeric":
            score_parts = _score_numeric(task, rec)
        elif track == "clause_lookup":
            score_parts = _score_clause_lookup(task, rec)
        elif track == "behavioral_safety":
            score_parts = _score_behavioral(task, rec)
        elif track == "synthesis":
            score_parts = _score_synthesis(task, rec, human_rows)
        else:
            continue

        row = {
            "agent_id": rec.agent_id,
            "model_name": rec.model_name,
            "task_id": rec.task_id,
            "track": track,
            "difficulty": task.get("difficulty", ""),
            "task_score": score_parts.get("task_score", 0.0),
            "latency_s": rec.latency_s,
            "cost_usd": rec.cost_usd,
        }
        row.update(score_parts)
        per_task_rows.append(row)

    # Aggregate by agent
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for row in per_task_rows:
        by_agent.setdefault(row["agent_id"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for agent_id, rows in by_agent.items():
        scores = [float(r.get("task_score", 0.0)) for r in rows]
        latencies = [float(r["latency_s"]) for r in rows if r.get("latency_s") is not None]
        costs = [float(r["cost_usd"]) for r in rows if r.get("cost_usd") is not None]

        track_means: dict[str, float] = {}
        for track in ["numeric", "clause_lookup", "synthesis", "behavioral_safety"]:
            track_scores = [float(r.get("task_score", 0.0)) for r in rows if r.get("track") == track]
            track_means[track] = sum(track_scores) / len(track_scores) if track_scores else float("nan")

        # Weighted overall quality (0..100)
        overall_quality = (
            0.35 * (0.0 if math.isnan(track_means["numeric"]) else track_means["numeric"])
            + 0.20 * (0.0 if math.isnan(track_means["clause_lookup"]) else track_means["clause_lookup"])
            + 0.30 * (0.0 if math.isnan(track_means["synthesis"]) else track_means["synthesis"])
            + 0.15 * (0.0 if math.isnan(track_means["behavioral_safety"]) else track_means["behavioral_safety"])
        ) * 100.0

        ci_low, ci_high = _bootstrap_ci(scores)
        ci_low *= 100.0
        ci_high *= 100.0

        median_latency = statistics.median(latencies) if latencies else float("nan")
        p90_latency = _safe_p90(latencies)

        efficiency_qps = (
            overall_quality / median_latency if latencies and median_latency > 0 else float("nan")
        )
        quality_per_dollar = (
            overall_quality / sum(costs) if costs and sum(costs) > 0 else float("nan")
        )

        model_names = sorted({r.get("model_name", "") for r in rows if r.get("model_name")})

        summary_rows.append(
            {
                "agent_id": agent_id,
                "model_names": "; ".join(model_names),
                "n_tasks": len(rows),
                "track_numeric": track_means["numeric"] * 100.0
                if not math.isnan(track_means["numeric"])
                else float("nan"),
                "track_clause_lookup": track_means["clause_lookup"] * 100.0
                if not math.isnan(track_means["clause_lookup"])
                else float("nan"),
                "track_synthesis": track_means["synthesis"] * 100.0
                if not math.isnan(track_means["synthesis"])
                else float("nan"),
                "track_behavioral_safety": track_means["behavioral_safety"] * 100.0
                if not math.isnan(track_means["behavioral_safety"])
                else float("nan"),
                "overall_quality_score": overall_quality,
                "overall_quality_ci95_low": ci_low,
                "overall_quality_ci95_high": ci_high,
                "median_latency_s": median_latency,
                "p90_latency_s": p90_latency,
                "total_cost_usd": sum(costs) if costs else float("nan"),
                "efficiency_quality_per_second": efficiency_qps,
                "quality_per_usd": quality_per_dollar,
            }
        )

    # Write outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_task_path = out_dir / "per_task_scores.csv"
    summary_path = out_dir / "summary_scores.csv"
    summary_json_path = out_dir / "summary_scores.json"

    if per_task_rows:
        fieldnames = sorted({k for row in per_task_rows for k in row.keys()})
        with per_task_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_task_rows)

    if summary_rows:
        fieldnames = sorted({k for row in summary_rows for k in row.keys()})
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "responses_scored": len(per_task_rows),
                "agents": sorted(by_agent.keys()),
                "outputs": {
                    "per_task_scores_csv": str(per_task_path),
                    "summary_scores_csv": str(summary_path),
                    "summary_scores_json": str(summary_json_path),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
