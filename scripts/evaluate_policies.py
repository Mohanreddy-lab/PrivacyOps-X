from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import PrivacyOpsAction
from server.env import PrivacyOpsXEnvironment
from server.fixtures import load_tasks
from server.teacher import build_teacher_plan

from shared import build_user_prompt, ensure_parent, extract_json


FIELD_CHOICES = {
    "request_type": ["access", "erasure", "access_erasure", "suppression"],
    "verification_status": ["verified", "verification_required", "rejected_identity"],
    "jurisdiction": ["cpra", "gdpr", "coppa", "other"],
    "sla_days": [30, 45],
    "priority": ["low", "medium", "high", "urgent"],
    "routing_queue": [
        "triage",
        "fulfillment",
        "manual_privacy_review",
        "privacy_legal",
        "fraud_privacy_joint",
    ],
    "case_status": [
        "pending_verification",
        "approved",
        "partially_fulfilled",
        "escalated",
        "denied",
        "closed",
    ],
    "retention_decision": [
        "none",
        "retain_billing",
        "retain_legal_hold",
        "partial_delete",
        "suppress_marketing",
    ],
    "escalation_required": [True, False],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate teacher, random, or local-checkpoint policies on PrivacyOps-X."
    )
    parser.add_argument("--policy", choices=["teacher", "random", "model"], default="teacher")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def sample_random_action(
    observation,
    task: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    unseen_records = [
        record_id
        for record_id in task["required_records"]
        if record_id not in {record.record_id for record in observation.visible_records}
    ]
    visible_policy_ids = {article.article_id for article in observation.visible_policy_articles}
    actions: list[dict[str, Any]] = []

    if not unseen_records and observation.steps_remaining <= 2:
        actions.append({"action_type": "submit"})
    if unseen_records:
        actions.append({"action_type": "open_record", "target_id": rng.choice(unseen_records)})
    actions.extend(
        [
            {"action_type": "inspect_case"},
            {
                "action_type": "search_policy",
                "query": rng.choice(
                    [
                        "verify identity delete records",
                        "guardian legal hold billing",
                        "privacy request escalation partial fulfillment",
                    ]
                ),
            },
            {
                "action_type": "message_requester",
                "content": rng.choice(
                    [
                        "Please verify your identity and clarify the records in scope.",
                        "Can you confirm the account emails and any guardian authority?",
                        "We may need to retain some records. Please confirm what you need.",
                    ]
                ),
            },
            {
                "action_type": "request_review",
                "reviewer": rng.choice(["compliance", "legal", "audit"]),
            },
            {
                "action_type": "set_case_field",
                "field_name": rng.choice(list(FIELD_CHOICES.keys())),
                "field_value": rng.choice(FIELD_CHOICES[rng.choice(list(FIELD_CHOICES.keys()))]),
                "confidence": round(rng.uniform(0.4, 1.0), 2),
            },
            {
                "action_type": "add_internal_note",
                "content": rng.choice(
                    [
                        "Need more review.",
                        "Possible deletion request.",
                        "Potential hold or billing issue.",
                    ]
                ),
            },
            {
                "action_type": "draft_reply",
                "content": rng.choice(
                    [
                        "We are reviewing your request.",
                        "Please wait while we process your case.",
                        "We will update you after verification.",
                    ]
                ),
            },
            {"action_type": "submit"},
        ]
    )
    action = rng.choice(actions)
    if action["action_type"] == "set_case_field":
        field_name = action["field_name"]
        action["field_value"] = rng.choice(FIELD_CHOICES[field_name])
    return action


class LocalCheckpointPolicy:
    def __init__(self, model_path: str, max_new_tokens: int) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.max_new_tokens = max_new_tokens

    def next_action(self, task_id: str, observation, history: list[str]) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an agent operating a privacy operations benchmark. "
                    "Return exactly one compact JSON object describing the next action."
                ),
            },
            {"role": "user", "content": build_user_prompt(task_id, observation, history)},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        try:
            return extract_json(text)
        except Exception:
            return {"action_type": "submit"}


def main() -> None:
    args = parse_args()
    tasks = load_tasks()
    task_ids = args.task_ids or list(tasks.keys())
    rng = random.Random(args.seed)

    local_model = None
    if args.policy == "model":
        if not args.model_path:
            raise SystemExit("--model-path is required when --policy=model")
        local_model = LocalCheckpointPolicy(args.model_path, args.max_new_tokens)

    env = PrivacyOpsXEnvironment()
    episodes: list[dict[str, Any]] = []

    for task_id in task_ids:
        task = tasks[task_id]
        teacher_plan = build_teacher_plan(task_id)
        for variant in task["variants"]:
            observation = env.reset(task_id=task_id, variant_id=variant["variant_id"], seed=args.seed)
            history: list[str] = []
            step_limit = int(task["step_limit"]) + 2
            for step_index in range(1, step_limit + 1):
                if observation.done:
                    break
                if args.policy == "teacher":
                    action_payload = (
                        teacher_plan[step_index - 1]
                        if step_index <= len(teacher_plan)
                        else {"action_type": "submit"}
                    )
                elif args.policy == "random":
                    action_payload = sample_random_action(observation, task, rng)
                else:
                    action_payload = local_model.next_action(task_id, observation, history)
                action = PrivacyOpsAction(**action_payload)
                history.append(json.dumps(action_payload, sort_keys=True))
                observation = env.step(action)

            info = observation.metadata.get("info", {})
            episodes.append(
                {
                    "task_id": task_id,
                    "variant_id": variant["variant_id"],
                    "policy": args.policy,
                    "steps": env.state.step_count,
                    "final_score": float(info.get("final_score", observation.reward or 0.0)),
                    "partial_score": float(info.get("partial_score", 0.0)),
                    "failure_modes": info.get("failure_modes", {}),
                    "theme_alignment": info.get("theme_alignment", {}),
                }
            )

    grouped: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        grouped[episode["task_id"]].append(episode["final_score"])

    by_task = {
        task_id: {
            "mean_final_score": round(sum(scores) / len(scores), 4),
            "num_episodes": len(scores),
        }
        for task_id, scores in grouped.items()
    }
    overall_mean = round(
        sum(episode["final_score"] for episode in episodes) / max(1, len(episodes)),
        4,
    )

    payload = {
        "policy": args.policy,
        "model_path": args.model_path,
        "episodes": episodes,
        "by_task": by_task,
        "overall": {
            "mean_final_score": overall_mean,
            "num_episodes": len(episodes),
        },
    }
    ensure_parent(args.output)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report to {args.output}")


if __name__ == "__main__":
    main()
