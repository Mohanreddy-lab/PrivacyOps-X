from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import PrivacyOpsAction
from server.env import PrivacyOpsXEnvironment
from server.fixtures import load_tasks
from training_history import save_training_artifacts


def _observation_to_text(observation) -> str:
    return (
        f"Ticket summary:\n{observation.ticket_summary}\n\n"
        f"Workspace: {observation.workspace.model_dump_json()}\n"
        f"Visible records: {[record.record_id for record in observation.visible_records]}\n"
        f"Visible policies: {[article.article_id for article in observation.visible_policy_articles]}\n"
        f"Latest requester message: {observation.latest_requester_message}\n"
        f"Review findings: {[finding.message for finding in observation.review_findings]}\n"
        f"Steps remaining: {observation.steps_remaining}\n"
        f"Current reward: {observation.reward}\n"
    )


def _coerce_field_value(field_name: str, field_value: str):
    if field_name == "sla_days":
        return int(field_value)
    if field_name == "escalation_required":
        return field_value.lower() in {"true", "1", "yes"}
    return field_value


class PrivacyOpsToolEnv:
    def __init__(self) -> None:
        self.env = PrivacyOpsXEnvironment()
        self.reward = 0.0
        self.final_score = 0.0
        self.last_observation = None

    def reset(self, **kwargs) -> str:
        self.reward = 0.0
        self.final_score = 0.0
        task_id = kwargs.get("task_id")
        variant_id = kwargs.get("variant_id")
        seed = kwargs.get("seed", 0)
        self.last_observation = self.env.reset(task_id=task_id, variant_id=variant_id, seed=seed)
        return _observation_to_text(self.last_observation)

    def _step(self, action: PrivacyOpsAction) -> str:
        self.last_observation = self.env.step(action)
        self.reward = float(self.last_observation.reward or 0.0)
        self.final_score = float(
            self.last_observation.metadata.get("info", {}).get("final_score", self.reward)
        )
        return _observation_to_text(self.last_observation)

    def inspect_case(self) -> str:
        """Open the full inbound case file and reveal the full requester thread."""
        return self._step(PrivacyOpsAction(action_type="inspect_case"))

    def open_record(self, target_id: str) -> str:
        """Open a record by record id.

        Args:
            target_id: The record id to open.
        """
        return self._step(PrivacyOpsAction(action_type="open_record", target_id=target_id))

    def search_policy(self, query: str) -> str:
        """Search policy articles by free-text query.

        Args:
            query: Search query describing the case constraints.
        """
        return self._step(PrivacyOpsAction(action_type="search_policy", query=query))

    def open_policy_article(self, target_id: str) -> str:
        """Open a specific policy article by article id.

        Args:
            target_id: Policy article id.
        """
        return self._step(
            PrivacyOpsAction(action_type="open_policy_article", target_id=target_id)
        )

    def set_case_field(self, field_name: str, field_value: str, confidence: float = 1.0) -> str:
        """Update a structured workspace field.

        Args:
            field_name: One of the workspace field names.
            field_value: Value to set. Use strings for all fields; booleans and integers are parsed automatically.
            confidence: Confidence attached to the update.
        """
        return self._step(
            PrivacyOpsAction(
                action_type="set_case_field",
                field_name=field_name,
                field_value=_coerce_field_value(field_name, field_value),
                confidence=confidence,
            )
        )

    def add_internal_note(self, content: str) -> str:
        """Add an internal analyst note.

        Args:
            content: Internal rationale or evidence summary.
        """
        return self._step(PrivacyOpsAction(action_type="add_internal_note", content=content))

    def draft_reply(self, content: str) -> str:
        """Draft a customer-facing reply.

        Args:
            content: The reply text to draft.
        """
        return self._step(PrivacyOpsAction(action_type="draft_reply", content=content))

    def message_requester(self, content: str) -> str:
        """Send a follow-up message to the requester.

        Args:
            content: Clarification or verification request.
        """
        return self._step(PrivacyOpsAction(action_type="message_requester", content=content))

    def request_review(self, reviewer: str) -> str:
        """Request a deterministic internal review.

        Args:
            reviewer: Reviewer name: compliance, legal, or audit.
        """
        return self._step(PrivacyOpsAction(action_type="request_review", reviewer=reviewer))

    def self_review(self) -> str:
        """Run a self-review to surface unresolved issues before submitting."""
        return self._step(PrivacyOpsAction(action_type="self_review"))

    def submit(self) -> str:
        """Submit the case for final scoring."""
        return self._step(PrivacyOpsAction(action_type="submit"))


def reward_func(environments, **kwargs) -> list[float]:
    return [max(env.reward, env.final_score) for env in environments]


def build_dataset(task_ids: list[str]) -> Dataset:
    tasks = load_tasks()
    rows = []
    for task_id in task_ids:
        task = tasks[task_id]
        for variant in task["variants"]:
            rows.append(
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": (
                                "Resolve the privacy case safely, gather evidence, coordinate "
                                "reviewers, and maximize the final benchmark score."
                            ),
                        }
                    ],
                    "task_id": task_id,
                    "variant_id": variant["variant_id"],
                    "seed": 0,
                }
            )
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an agent directly against PrivacyOps-X with TRL GRPO and OpenEnv tools."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="outputs/checkpoints/privacyops_x_grpo")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--task-ids", nargs="*", default=None)
    args = parser.parse_args()

    task_ids = args.task_ids or list(load_tasks().keys())
    dataset = build_dataset(task_ids)

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=reward_func,
        args=GRPOConfig(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            log_completions=True,
            report_to="none",
            chat_template_kwargs={"enable_thinking": False},
        ),
        environment_factory=PrivacyOpsToolEnv,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    save_training_artifacts(trainer.state.log_history, args.output_dir, "grpo")


if __name__ == "__main__":
    main()
