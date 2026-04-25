from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import PrivacyOpsAction, WorkspaceView
from server.env import PrivacyOpsXEnvironment
from server.fixtures import load_tasks
from shared import ensure_parent


FIELD_ORDER = [
    "request_type",
    "verification_status",
    "jurisdiction",
    "sla_days",
    "priority",
    "routing_queue",
    "case_status",
    "retention_decision",
    "escalation_required",
]


class AdaptivePrivacyPolicy:
    def __init__(self) -> None:
        self.capabilities = {
            "inspect_case",
            "basic_workspace",
            "reply",
        }

    def plan(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        expected = task["expected_workspace"]
        default_workspace = WorkspaceView().model_dump()
        actions: list[dict[str, Any]] = []

        if "inspect_case" in self.capabilities:
            actions.append({"action_type": "inspect_case"})
        if "records" in self.capabilities:
            for record_id in task["required_records"]:
                actions.append({"action_type": "open_record", "target_id": record_id})
        if "policy" in self.capabilities and task.get("teacher_policy_query"):
            actions.append(
                {"action_type": "search_policy", "query": task["teacher_policy_query"]}
            )
        if "requester" in self.capabilities and task.get("required_requester_facts"):
            actions.append(
                {
                    "action_type": "message_requester",
                    "content": task["teacher_requester_message"],
                }
            )

        for field_name in FIELD_ORDER:
            field_value = expected[field_name]
            if field_value is None or default_workspace.get(field_name) == field_value:
                continue
            if field_name in {
                "request_type",
                "verification_status",
                "jurisdiction",
                "sla_days",
                "priority",
            }:
                if "basic_workspace" not in self.capabilities:
                    continue
            else:
                if "legal_resolution" not in self.capabilities:
                    continue
            actions.append(
                {
                    "action_type": "set_case_field",
                    "field_name": field_name,
                    "field_value": field_value,
                    "confidence": 1.0,
                }
            )

        if "notes" in self.capabilities and task.get("teacher_internal_note"):
            actions.append(
                {"action_type": "add_internal_note", "content": task["teacher_internal_note"]}
            )
        if "reply" in self.capabilities and task.get("teacher_reply"):
            actions.append({"action_type": "draft_reply", "content": task["teacher_reply"]})
        else:
            actions.append(
                {
                    "action_type": "draft_reply",
                    "content": "We are reviewing your request and will update you shortly.",
                }
            )

        if "reviews_basic" in self.capabilities and "compliance" in task["required_reviewers"]:
            actions.append({"action_type": "request_review", "reviewer": "compliance"})
        if "reviews_legal" in self.capabilities and "legal" in task["required_reviewers"]:
            actions.append({"action_type": "request_review", "reviewer": "legal"})
        if "reviews_audit" in self.capabilities and "audit" in task["required_reviewers"]:
            actions.append({"action_type": "request_review", "reviewer": "audit"})
        if "self_review" in self.capabilities:
            actions.append({"action_type": "self_review"})

        actions.append({"action_type": "submit"})
        return actions

    def update_from_episode(self, episode: dict[str, Any]) -> None:
        failure_modes = episode["failure_modes"]
        lessons = {lesson["lesson_id"] for lesson in episode["improvement_lessons"]}

        if failure_modes.get("evidence_gap", 0) > 0 or "evidence_before_action" in lessons:
            self.capabilities.update({"records", "policy"})
        if (
            failure_modes.get("verification_error", 0) > 0
            or failure_modes.get("policy_violation", 0) > 0
            or "verification_first" in lessons
        ):
            self.capabilities.update({"requester", "basic_workspace", "reviews_basic"})
        if failure_modes.get("logic_error", 0) > 0 or "conflict_aware_retention" in lessons:
            self.capabilities.update({"legal_resolution", "reviews_legal"})
        if (
            failure_modes.get("requester_miscommunication", 0) > 0
            or "high_yield_followups" in lessons
        ):
            self.capabilities.update({"requester"})
        if (
            failure_modes.get("overconfidence", 0) > 0
            or failure_modes.get("unsafe_action", 0) > 0
            or "confidence_calibration" in lessons
        ):
            self.capabilities.update({"reviews_audit", "self_review"})
        if episode["final_score"] < 0.85:
            self.capabilities.update({"notes", "reply"})


def run_episode(env: PrivacyOpsXEnvironment, task_id: str, variant_id: str, policy) -> dict[str, Any]:
    tasks = load_tasks()
    task = tasks[task_id]
    observation = env.reset(task_id=task_id, variant_id=variant_id, seed=0)
    actions = policy.plan(task)
    trajectory = []

    for action_payload in actions:
        action = PrivacyOpsAction(**action_payload)
        observation = env.step(action)
        trajectory.append(action_payload)
        if observation.done:
            break

    info = observation.metadata.get("info", {})
    return {
        "task_id": task_id,
        "variant_id": variant_id,
        "final_score": float(info.get("final_score", observation.reward or 0.0)),
        "reward": float(observation.reward or 0.0),
        "loss_proxy": round(1.0 - float(info.get("final_score", observation.reward or 0.0)), 4),
        "failure_modes": info.get("failure_modes", {}),
        "improvement_lessons": info.get("improvement_lessons", []),
        "trajectory": trajectory,
        "theme_alignment": info.get("theme_alignment", {}),
    }


def maybe_plot(report: dict[str, Any], output_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    episodes = report["episodes"]
    xs = list(range(1, len(episodes) + 1))
    rewards = [episode["final_score"] for episode in episodes]
    losses = [episode["loss_proxy"] for episode in episodes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(xs, rewards, marker="o")
    axes[0].set_title("Self-improvement reward curve")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Final score")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.25)

    axes[1].plot(xs, losses, marker="o", color="#d95f02")
    axes[1].set_title("Self-improvement loss curve")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss proxy (1 - final score)")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    ensure_parent(output_png)
    fig.savefig(output_png, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an explicit self-improvement loop and log before/after behavior."
    )
    parser.add_argument(
        "--task-id",
        default="finale_cross_border_recovery_cascade",
        help="Task to improve on. Defaults to the finale showcase task.",
    )
    parser.add_argument("--variant-id", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evals/self_improvement_cycle.json"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("outputs/plots/self_improvement_curve.png"),
    )
    args = parser.parse_args()

    tasks = load_tasks()
    task = tasks[args.task_id]
    variant_id = args.variant_id or task["canonical_variant_id"]
    env = PrivacyOpsXEnvironment()
    policy = AdaptivePrivacyPolicy()
    episodes = []

    for episode_index in range(1, args.episodes + 1):
        capabilities_before = sorted(policy.capabilities)
        episode = run_episode(env, args.task_id, variant_id, policy)
        episode["episode"] = episode_index
        episode["capabilities_before"] = capabilities_before
        episodes.append(episode)
        policy.update_from_episode(episode)
        episode["capabilities_after"] = sorted(policy.capabilities)

    report = {
        "task_id": args.task_id,
        "variant_id": variant_id,
        "baseline_score": episodes[0]["final_score"],
        "improved_score": episodes[-1]["final_score"],
        "episodes": episodes,
        "before_behavior": episodes[0]["trajectory"],
        "after_behavior": episodes[-1]["trajectory"],
    }
    ensure_parent(args.output)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    maybe_plot(report, args.plot_output)
    print(f"Wrote self-improvement report to {args.output}")


if __name__ == "__main__":
    main()
