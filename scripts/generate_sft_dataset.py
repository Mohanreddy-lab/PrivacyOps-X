from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.env import PrivacyOpsXEnvironment
from server.fixtures import load_tasks
from server.teacher import build_teacher_actions, build_teacher_plan

from shared import build_messages, ensure_parent, messages_to_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a conversational SFT dataset from teacher trajectories."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/train/privacyops_x_sft.jsonl"),
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task ids to include. Defaults to every loaded task.",
    )
    args = parser.parse_args()

    tasks = load_tasks()
    selected_task_ids = args.tasks or list(tasks.keys())

    ensure_parent(args.output)
    env = PrivacyOpsXEnvironment()
    rows_written = 0

    with args.output.open("w", encoding="utf-8") as handle:
        for task_id in selected_task_ids:
            task = tasks[task_id]
            teacher_plan = build_teacher_plan(task_id)
            teacher_actions = build_teacher_actions(task_id)
            for variant in task["variants"]:
                result = env.reset(task_id=task_id, variant_id=variant["variant_id"], seed=0)
                observation = result
                history: list[str] = []
                for step_index, action in enumerate(teacher_plan, start=1):
                    messages = build_messages(task_id, observation, history, action)
                    row = {
                        "task_id": task_id,
                        "variant_id": variant["variant_id"],
                        "step": step_index,
                        "messages": messages,
                        "text": messages_to_text(messages),
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows_written += 1
                    history.append(json.dumps(action, sort_keys=True))
                    observation = env.step(teacher_actions[step_index - 1])

    print(f"Wrote {rows_written} examples to {args.output}")


if __name__ == "__main__":
    main()
