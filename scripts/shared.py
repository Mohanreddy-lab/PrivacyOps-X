from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import SYSTEM_PROMPT, extract_json
from models import PrivacyOpsObservation


def build_user_prompt(
    task_id: str,
    observation: PrivacyOpsObservation,
    history: list[str],
) -> str:
    return (
        f"Task: {task_id}\n"
        f"Ticket summary:\n{observation.ticket_summary}\n\n"
        f"Workspace: {observation.workspace.model_dump_json()}\n"
        f"Visible records: {[record.record_id for record in observation.visible_records]}\n"
        f"Visible policies: {[article.article_id for article in observation.visible_policy_articles]}\n"
        f"Requester thread: {[turn.model_dump(mode='json') for turn in observation.requester_thread[-6:]]}\n"
        f"Revealed requester facts: {observation.revealed_requester_facts}\n"
        f"Review findings: {[finding.message for finding in observation.review_findings]}\n"
        f"Stakeholder inbox: {[message.model_dump(mode='json') for message in observation.stakeholder_inbox[-6:]]}\n"
        f"Milestones: {[milestone.model_dump(mode='json') for milestone in observation.milestones]}\n"
        f"Theme alignment: {observation.theme_alignment.model_dump()}\n"
        f"Explanation trace: {observation.explanation_trace}\n"
        f"Draft reply: {observation.draft_reply}\n"
        f"Recent history: {history[-4:]}\n"
        f"Steps remaining: {observation.steps_remaining}\n"
    )


def build_messages(
    task_id: str,
    observation: PrivacyOpsObservation,
    history: list[str],
    action_payload: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task_id, observation, history)},
        {"role": "assistant", "content": json.dumps(action_payload, sort_keys=True)},
    ]


def messages_to_text(messages: list[dict[str, str]]) -> str:
    blocks = []
    for message in messages:
        blocks.append(f"{message['role'].upper()}:\n{message['content']}")
    return "\n\n".join(blocks)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
