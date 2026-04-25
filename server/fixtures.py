"""Fixture loading for PrivacyOps-X."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT_DIR / "tasks"


REQUESTER_PLAYBOOKS: dict[str, dict[str, Any]] = {
    "easy_verified_access_with_injection": {
        "required_requester_facts": ["requester_confirms_primary_email"],
        "optimal_steps": 15,
        "step_limit": 15,
        "generic_requester_reply": (
            "I only need the normal copy of my data. Please let me know what you need "
            "from me to proceed safely."
        ),
        "requester_rules": [
            {
                "rule_id": "easy_confirm_primary_email",
                "keywords_all": ["verify", "identity"],
                "keywords_any": ["email", "45 day", "timeline", "copy"],
                "reply": (
                    "Yes, this is the same email on the account. The standard 45 day "
                    "timeline is fine. I just want the normal copy of my data."
                ),
                "fact_ids": [
                    "requester_confirms_primary_email",
                    "requester_accepts_standard_timeline",
                ],
            },
            {
                "rule_id": "easy_confirm_email_only",
                "keywords_all": ["verify", "identity"],
                "reply": "Yes, this is the same email I use for the account.",
                "fact_ids": ["requester_confirms_primary_email"],
            },
        ],
    },
    "medium_unverified_erasure_multi_account": {
        "required_requester_facts": [
            "requester_provided_account_aliases",
            "requester_acknowledges_billing_retention",
        ],
        "optimal_steps": 19,
        "step_limit": 19,
        "generic_requester_reply": (
            "I still want the deletion processed, but tell me exactly how to verify "
            "my identity and what records you cannot remove right away."
        ),
        "requester_rules": [
            {
                "rule_id": "medium_verify_and_retention",
                "keywords_all": ["verify", "identity"],
                "keywords_any": ["billing", "invoice", "retain", "records"],
                "reply": (
                    "The two account emails were elena.stahl@example.eu and "
                    "e.stahl.billing@example.eu. You can keep the invoice record if the "
                    "law requires it, but please delete the rest after verification."
                ),
                "fact_ids": [
                    "requester_provided_account_aliases",
                    "requester_acknowledges_billing_retention",
                ],
            },
            {
                "rule_id": "medium_verify_only",
                "keywords_all": ["verify", "identity"],
                "reply": (
                    "The accounts were elena.stahl@example.eu and "
                    "e.stahl.billing@example.eu. I can share invoice details if needed."
                ),
                "fact_ids": ["requester_provided_account_aliases"],
            },
        ],
    },
    "hard_guardian_minor_legal_hold_fraud": {
        "required_requester_facts": [
            "guardian_docs_offered",
            "guardian_accepts_partial_action",
        ],
        "optimal_steps": 20,
        "step_limit": 20,
        "generic_requester_reply": (
            "I need to know what proof you require from me as guardian and what you "
            "can still do while the investigation is open."
        ),
        "requester_rules": [
            {
                "rule_id": "hard_guardian_docs_and_partial_action",
                "keywords_all": ["guardian", "authority"],
                "keywords_any": [
                    "legal hold",
                    "fraud",
                    "suppress marketing",
                    "cannot delete",
                ],
                "reply": (
                    "I can provide the guardianship paperwork today. I understand the "
                    "legal hold and fraud review prevent full deletion for now, but "
                    "please suppress marketing immediately."
                ),
                "fact_ids": [
                    "guardian_docs_offered",
                    "guardian_accepts_partial_action",
                ],
            },
            {
                "rule_id": "hard_guardian_docs_only",
                "keywords_all": ["guardian", "authority"],
                "reply": "I can send you the guardianship documents right away.",
                "fact_ids": ["guardian_docs_offered"],
            },
        ],
    },
    "finale_cross_border_recovery_cascade": {
        "required_requester_facts": [
            "requester_provided_account_aliases",
            "guardian_docs_offered",
            "requester_acknowledges_billing_retention",
            "guardian_accepts_partial_action",
        ],
        "optimal_steps": 24,
        "step_limit": 28,
        "generic_requester_reply": (
            "Tell me what proof you need, which linked records you can still process, "
            "and what must stay retained while the investigation is open."
        ),
        "requester_rules": [
            {
                "rule_id": "finale_full_alignment",
                "keywords_all": ["guardian", "authority"],
                "keywords_any": [
                    "linked",
                    "billing",
                    "retain",
                    "partial fulfillment",
                    "cannot delete",
                ],
                "reply": (
                    "The linked aliases are maya.ross@example.eu and "
                    "maya.contracting@example.eu, and the child profile is on the same family plan. "
                    "I can provide guardianship documents now. I understand some billing or legally held "
                    "records may need to stay retained, and partial fulfillment is fine while the fraud "
                    "review remains open."
                ),
                "fact_ids": [
                    "requester_provided_account_aliases",
                    "guardian_docs_offered",
                    "requester_acknowledges_billing_retention",
                    "guardian_accepts_partial_action",
                ],
            },
            {
                "rule_id": "finale_guardian_docs_only",
                "keywords_all": ["guardian", "authority"],
                "reply": (
                    "I can send the guardianship documents right away if that is the required first step."
                ),
                "fact_ids": ["guardian_docs_offered"],
            },
            {
                "rule_id": "finale_linked_accounts_only",
                "keywords_any": ["linked", "account aliases", "billing", "retain"],
                "reply": (
                    "The linked aliases are maya.ross@example.eu and "
                    "maya.contracting@example.eu. If some billing records must stay retained, "
                    "please process the rest as soon as you can."
                ),
                "fact_ids": [
                    "requester_provided_account_aliases",
                    "requester_acknowledges_billing_retention",
                ],
            },
        ],
    },
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_policies() -> dict[str, dict[str, Any]]:
    raw = _load_json(TASKS_DIR / "policies.json")
    return {article["article_id"]: article for article in raw["policies"]}


def load_tasks() -> dict[str, dict[str, Any]]:
    task_files = [
        "easy_verified_access.json",
        "medium_unverified_erasure.json",
        "hard_guardian_legal_hold.json",
        "finale_cross_border_cascade.json",
    ]
    tasks: dict[str, dict[str, Any]] = {}
    for filename in task_files:
        task = _load_json(TASKS_DIR / filename)
        playbook = REQUESTER_PLAYBOOKS.get(task["task_id"], {})
        task.setdefault("required_requester_facts", [])
        task.setdefault("generic_requester_reply", "Please tell me the next required step.")
        task.setdefault("requester_rules", [])
        task.update(playbook)
        tasks[task["task_id"]] = task
    return tasks


def task_order() -> list[str]:
    return [
        "easy_verified_access_with_injection",
        "medium_unverified_erasure_multi_account",
        "hard_guardian_minor_legal_hold_fraud",
    ]
