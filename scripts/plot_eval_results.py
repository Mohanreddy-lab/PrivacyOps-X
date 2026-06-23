from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from shared import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot policy comparison charts from evaluation JSON files."
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/plots/policy_comparison.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    all_task_ids = sorted(
        {task_id for report in reports for task_id in report.get("by_task", {}).keys()}
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.8 / max(1, len(reports))
    positions = list(range(len(all_task_ids)))

    for index, report in enumerate(reports):
        scores = [
            report.get("by_task", {}).get(task_id, {}).get("mean_final_score", 0.0)
            for task_id in all_task_ids
        ]
        xs = [position + index * width for position in positions]
        label = report.get("policy") or Path(report.get("model_path") or f"run-{index}").name
        ax.bar(xs, scores, width=width, label=label)

    ax.set_title("PrivacyOps-X policy comparison")
    ax.set_xlabel("Task")
    ax.set_ylabel("Mean final score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([position + width * (len(reports) - 1) / 2 for position in positions])
    ax.set_xticklabels(all_task_ids, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    ensure_parent(args.output)
    fig.savefig(args.output, dpi=200)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
