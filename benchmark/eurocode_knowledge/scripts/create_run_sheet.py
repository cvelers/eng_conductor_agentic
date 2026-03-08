#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_AGENTS = [
    {
        "agent_id": "claude_sonnet_4_6",
        "model_name": "claude-sonnet-4-6",
        "thinking_mode": "max",
    },
    {
        "agent_id": "chatgpt_5_3_requested",
        "model_name": "gpt-5.2-chat-latest",
        "thinking_mode": "xhigh",
    },
    {
        "agent_id": "gemini_3_1_pro",
        "model_name": "gemini-3.1-pro",
        "thinking_mode": "high",
    },
    {
        "agent_id": "eng_conductor_orchestrator",
        "model_name": "orchestrator",
        "thinking_mode": "thinking",
    },
]

SYSTEM_PROMPT = (
    "You are being evaluated on Eurocode engineering performance. "
    "Answer strictly in JSON with keys: task_id, final_answer, citations, results, assumptions, "
    "needs_more_info, clarifying_questions. "
    "Do not include markdown fences or extra prose outside JSON. "
    "Use Eurocode clause/table citations where possible."
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_agents(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_AGENTS
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError("Agents config must be a JSON list.")
    agents: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        agent_id = str(item.get("agent_id", "")).strip()
        model_name = str(item.get("model_name", "")).strip()
        thinking_mode = str(item.get("thinking_mode", "")).strip()
        if not agent_id:
            continue
        agents.append(
            {
                "agent_id": agent_id,
                "model_name": model_name,
                "thinking_mode": thinking_mode,
            }
        )
    return agents


def _write_prompt_packets(
    out_dir: Path,
    agents: list[dict[str, str]],
    tasks: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for agent in agents:
        path = out_dir / f"prompt_packet_{agent['agent_id']}.md"
        lines: list[str] = []
        lines.append(f"# Prompt Packet: {agent['agent_id']}")
        lines.append("")
        lines.append(f"Model: `{agent['model_name']}`")
        lines.append(f"Thinking mode: `{agent['thinking_mode']}`")
        lines.append("")
        lines.append("## Global System Prompt")
        lines.append("")
        lines.append("```text")
        lines.append(SYSTEM_PROMPT)
        lines.append("```")
        lines.append("")
        lines.append("## Task Prompts")
        lines.append("")

        for task in tasks:
            lines.append(f"### {task['task_id']} ({task['track']})")
            lines.append("")
            lines.append("```text")
            lines.append(task["prompt"])
            lines.append("```")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default="benchmark/eurocode_knowledge/tasks/eurocode_benchmark_v1.json",
        help="Benchmark tasks JSON.",
    )
    parser.add_argument(
        "--agents",
        default="",
        help="Optional JSON list with agent config.",
    )
    parser.add_argument(
        "--out-csv",
        default="benchmark/eurocode_knowledge/templates/responses_template.csv",
        help="Output CSV path for response logging template.",
    )
    parser.add_argument(
        "--out-prompts-dir",
        default="benchmark/eurocode_knowledge/templates/prompt_packets",
        help="Output directory for per-agent prompt packets.",
    )
    args = parser.parse_args()

    benchmark = _load_json(Path(args.benchmark))
    tasks: list[dict[str, Any]] = benchmark.get("tasks", [])
    agents = _load_agents(Path(args.agents)) if args.agents else DEFAULT_AGENTS

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "agent_id",
        "model_name",
        "thinking_mode",
        "task_id",
        "track",
        "difficulty",
        "system_prompt",
        "prompt",
        "started_at_utc",
        "ended_at_utc",
        "latency_s",
        "cost_usd",
        "input_tokens",
        "output_tokens",
        "response_text",
        "response_json",
    ]

    rows: list[dict[str, str]] = []
    for agent in agents:
        for task in tasks:
            rows.append(
                {
                    "agent_id": agent["agent_id"],
                    "model_name": agent["model_name"],
                    "thinking_mode": agent["thinking_mode"],
                    "task_id": task.get("task_id", ""),
                    "track": task.get("track", ""),
                    "difficulty": task.get("difficulty", ""),
                    "system_prompt": SYSTEM_PROMPT,
                    "prompt": task.get("prompt", ""),
                    "started_at_utc": "",
                    "ended_at_utc": "",
                    "latency_s": "",
                    "cost_usd": "",
                    "input_tokens": "",
                    "output_tokens": "",
                    "response_text": "",
                    "response_json": "",
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_prompt_packets(Path(args.out_prompts_dir), agents, tasks)

    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(rows),
                "agents": [a["agent_id"] for a in agents],
                "tasks": len(tasks),
                "out_csv": str(out_csv),
                "out_prompts_dir": args.out_prompts_dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
