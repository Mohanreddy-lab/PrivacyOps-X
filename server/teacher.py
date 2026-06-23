"""Teacher-policy helpers for dataset generation and evaluation."""

from __future__ import annotations

from typing import Any

try:
    from ..models import PrivacyOpsAction, WorkspaceView
except ImportError:
    from models import PrivacyOpsAction, WorkspaceView

from .fixtures import load_tasks


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


def build_teacher_plan(task_id: str) -> list[dict[str, Any]]:
    tasks = load_tasks()
    if task_id not in tasks:
        raise KeyError(f"Unknown task_id: {task_id}")
    task = tasks[task_id]
    expected = task["expected_workspace"]
    default_workspace = WorkspaceView().model_dump()

    actions: list[dict[str, Any]] = [{"action_type": "inspect_case"}]
    for record_id in task["required_records"]:
        actions.append({"action_type": "open_record", "target_id": record_id})

    query = task.get("teacher_policy_query")
    if query:
        actions.append({"action_type": "search_policy", "query": query})

    requester_message = task.get("teacher_requester_message")
    if requester_message and task.get("required_requester_facts"):
        actions.append({"action_type": "message_requester", "content": requester_message})

    for field_name in FIELD_ORDER:
        field_value = expected[field_name]
        if field_value is None:
            continue
        if default_workspace.get(field_name) == field_value:
            continue
        actions.append(
            {
                "action_type": "set_case_field",
                "field_name": field_name,
                "field_value": field_value,
                "confidence": 1.0,
            }
        )

    internal_note = task.get("teacher_internal_note")
    if internal_note:
        actions.append({"action_type": "add_internal_note", "content": internal_note})

    reply = task.get("teacher_reply")
    if reply:
        actions.append({"action_type": "draft_reply", "content": reply})

    for reviewer in task["required_reviewers"]:
        actions.append({"action_type": "request_review", "reviewer": reviewer})

    if task.get("teacher_use_self_review"):
        actions.append({"action_type": "self_review"})

    actions.append({"action_type": "submit"})
    return actions


def build_teacher_actions(task_id: str) -> list[PrivacyOpsAction]:
    return [PrivacyOpsAction(**action) for action in build_teacher_plan(task_id)]
