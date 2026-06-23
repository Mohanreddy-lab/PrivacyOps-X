from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
EVALS = OUTPUTS / "evals"
PLOTS = OUTPUTS / "plots"

FINALE_TASK_ID = "finale_cross_border_recovery_cascade"
RANDOM_REPORT = EVALS / "random_finale_live.json"
TEACHER_REPORT = EVALS / "teacher_finale_live.json"
SELF_REPORT = EVALS / "self_improvement_cycle_live.json"
COMPARISON_PLOT = PLOTS / "finale_live_random_vs_teacher.png"
SELF_PLOT = PLOTS / "self_improvement_curve_live.png"


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _ensure_demo_artifacts(refresh: bool) -> None:
    python = sys.executable

    if refresh or not RANDOM_REPORT.exists():
        _run(
            [
                python,
                "scripts/evaluate_policies.py",
                "--policy",
                "random",
                "--task-ids",
                FINALE_TASK_ID,
                "--output",
                str(RANDOM_REPORT),
            ]
        )

    if refresh or not TEACHER_REPORT.exists():
        _run(
            [
                python,
                "scripts/evaluate_policies.py",
                "--policy",
                "teacher",
                "--task-ids",
                FINALE_TASK_ID,
                "--output",
                str(TEACHER_REPORT),
            ]
        )

    if refresh or not SELF_REPORT.exists() or not SELF_PLOT.exists():
        _run(
            [
                python,
                "scripts/run_self_improvement_cycle.py",
                "--task-id",
                FINALE_TASK_ID,
                "--output",
                str(SELF_REPORT),
                "--plot-output",
                str(SELF_PLOT),
            ]
        )

    if refresh or not COMPARISON_PLOT.exists():
        _run(
            [
                python,
                "scripts/plot_eval_results.py",
                "--inputs",
                str(RANDOM_REPORT),
                str(TEACHER_REPORT),
                "--output",
                str(COMPARISON_PLOT),
            ]
        )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or summarize the judge-facing finale demo artifacts."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute the finale demo artifacts instead of reusing existing ones.",
    )
    args = parser.parse_args()

    _ensure_demo_artifacts(args.refresh)

    random_report = _load_json(RANDOM_REPORT)
    teacher_report = _load_json(TEACHER_REPORT)
    self_report = _load_json(SELF_REPORT)

    random_score = float(random_report["overall"]["mean_final_score"])
    teacher_score = float(teacher_report["overall"]["mean_final_score"])
    baseline_score = float(self_report["baseline_score"])
    improved_score = float(self_report["improved_score"])

    print("PrivacyOps-X final demo")
    print(f"Random baseline: {random_score:.4f}")
    print(f"Teacher oracle: {teacher_score:.4f}")
    print(f"Self-improvement: {baseline_score:.4f} -> {improved_score:.4f}")

    sft_report = EVALS / "sft_checkpoint.json"
    if sft_report.exists():
        sft_score = float(_load_json(sft_report)["overall"]["mean_final_score"])
        print(f"GPU SFT checkpoint: {sft_score:.4f}")
    else:
        print("GPU SFT checkpoint: pending")

    print("Plots saved to:")
    print(f"- {COMPARISON_PLOT.relative_to(ROOT)}")
    print(f"- {SELF_PLOT.relative_to(ROOT)}")
    print("Reports saved to:")
    print(f"- {RANDOM_REPORT.relative_to(ROOT)}")
    print(f"- {TEACHER_REPORT.relative_to(ROOT)}")
    print(f"- {SELF_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
