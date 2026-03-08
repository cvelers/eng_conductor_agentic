#!/usr/bin/env python3
"""Plot AutoBench results from results.tsv.

Usage:
    python -m benchmark.autobench.plot_results
    python -m benchmark.autobench.plot_results --output path/to/chart.png
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TSV = PROJECT_ROOT / "benchmark" / "autobench" / "results.tsv"
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark" / "autobench" / "results" / "accuracy_plot.png"


def parse_results(tsv_path: Path) -> dict[str, list]:
    """Parse results.tsv into column lists."""
    data: dict[str, list] = {
        "iteration": [],
        "commit_hash": [],
        "accuracy_pct": [],
        "avg_latency_s": [],
        "description": [],
        "timestamp": [],
    }

    if not tsv_path.exists():
        return data

    lines = tsv_path.read_text().strip().splitlines()
    if len(lines) < 2:
        return data

    for i, line in enumerate(lines[1:], start=1):
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        data["iteration"].append(i)
        data["commit_hash"].append(parts[0])
        try:
            data["accuracy_pct"].append(float(parts[1]))
        except ValueError:
            data["accuracy_pct"].append(0.0)
        try:
            data["avg_latency_s"].append(float(parts[2]))
        except ValueError:
            data["avg_latency_s"].append(0.0)
        data["description"].append(parts[3] if len(parts) > 3 else "")
        data["timestamp"].append(parts[4] if len(parts) > 4 else "")

    return data


def plot(data: dict[str, list], output_path: Path) -> None:
    """Generate accuracy_pct over iterations plot."""
    import matplotlib.pyplot as plt

    if not data["iteration"]:
        print("No data to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    iterations = data["iteration"]
    accuracy = data["accuracy_pct"]
    latency = data["avg_latency_s"]
    commits = data["commit_hash"]
    descriptions = data["description"]

    # Color points: green for kept, red for discarded
    colors = ["#d32f2f" if "DISCARD" in d.upper() else "#2e7d32" for d in descriptions]

    # Accuracy line + scatter
    ax1.plot(iterations, accuracy, color="#1565c0", linewidth=2, zorder=2, label="accuracy_pct")
    ax1.scatter(iterations, accuracy, c=colors, s=60, zorder=3, edgecolors="white", linewidths=0.5)

    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, color="#1565c0")
    ax1.tick_params(axis="y", labelcolor="#1565c0")
    ax1.set_ylim(bottom=max(0, min(accuracy) - 5), top=min(100, max(accuracy) + 5))

    # Latency on secondary y-axis
    ax2 = ax1.twinx()
    ax2.bar(iterations, latency, alpha=0.15, color="#ff8f00", width=0.6, zorder=1, label="avg_latency_s")
    ax2.set_ylabel("Avg Latency (s)", fontsize=12, color="#ff8f00")
    ax2.tick_params(axis="y", labelcolor="#ff8f00")

    # Annotate commits
    for i, (x, y, c) in enumerate(zip(iterations, accuracy, commits)):
        ax1.annotate(
            c[:7],
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=7,
            color="#555",
        )

    # Title and legend
    best = max(accuracy)
    best_iter = iterations[accuracy.index(best)]
    ax1.set_title(
        f"AutoBench: eng_conductor Accuracy Over Iterations\n"
        f"Best: {best:.1f}% (iteration {best_iter})",
        fontsize=14,
        fontweight="bold",
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2e7d32", markersize=8, label="Kept"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d32f2f", markersize=8, label="Discarded"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticks(iterations)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AutoBench results")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path")
    args = parser.parse_args()

    data = parse_results(RESULTS_TSV)
    plot(data, Path(args.output))


if __name__ == "__main__":
    main()
