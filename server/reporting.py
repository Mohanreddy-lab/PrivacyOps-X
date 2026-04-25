"""Judge-facing reporting helpers for PrivacyOps-X."""

from __future__ import annotations

from typing import Any

try:
    from ..models import (
        BenchmarkBreakdown,
        ImprovementLesson,
        MilestoneStatus,
        PrivacyOpsState,
        ThemeAlignmentScores,
    )
except ImportError:
    from models import (
        BenchmarkBreakdown,
        ImprovementLesson,
        MilestoneStatus,
        PrivacyOpsState,
        ThemeAlignmentScores,
    )

from .engines import (
    clamp,
    contains_any_keyword,
    fraction_keywords_present,
    reviewers_used,
)


MILESTONE_TITLES = {
    "intake_triage": "Intake and triage",
    "evidence_collection": "Evidence collection",
    "policy_grounding": "Policy grounding",
    "stakeholder_alignment": "Stakeholder alignment",
    "safe_communication": "Safe communication",
    "final_readiness": "Final readiness",
}


def _fraction_matches(actual: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    actual_set = set(actual)
    return sum(1 for item in expected if item in actual_set) / len(expected)


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return clamp(sum(values) / len(values))


def _workspace_accuracy(state: PrivacyOpsState, expected: dict[str, Any]) -> float:
    fields = [
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
    hits = 0
    for field_name in fields:
        if getattr(state.workspace, field_name) == expected[field_name]:
            hits += 1
    return hits / len(fields)


def _milestone_status(
    milestone_id: str,
    progress: float,
    rationale: str,
    state: PrivacyOpsState,
    task: dict[str, Any],
) -> MilestoneStatus:
    step_limit = max(1, int(task.get("step_limit", 1)))
    status = "complete" if progress >= 0.999 else "available"
    if status != "complete" and state.step_count >= int(0.7 * step_limit) and progress < 0.65:
        status = "at_risk"
    return MilestoneStatus(
        milestone_id=milestone_id,
        title=MILESTONE_TITLES[milestone_id],
        status=status,
        progress=round(clamp(progress), 4),
        rationale=rationale,
    )


def build_milestones(state: PrivacyOpsState, task: dict[str, Any]) -> list[MilestoneStatus]:
    expected = task["expected_workspace"]
    visible_policy_ids = list({*state.viewed_policy_ids, *state.visible_policy_ids})
    record_fraction = _fraction_matches(state.viewed_record_ids, task["required_records"])
    requester_fraction = _fraction_matches(
        state.revealed_requester_facts, task.get("required_requester_facts", [])
    )
    policy_fraction = _fraction_matches(
        visible_policy_ids, task["required_policy_articles"]
    )
    reviewers_fraction = _fraction_matches(
        sorted(reviewers_used(state)), sorted(task["required_reviewers"])
    )
    note_fraction = fraction_keywords_present(
        "\n".join(state.note_history), task["required_note_keywords"]
    )
    reply_fraction = fraction_keywords_present(
        state.draft_reply, task["required_reply_keywords"]
    )
    safe_reply = 0.0
    if state.draft_reply:
        safe_reply = 0.0 if contains_any_keyword(state.draft_reply, task["forbidden_reply_keywords"]) else 1.0
    workspace_accuracy = _workspace_accuracy(state, expected)
    audit_or_self_review = float(
        any(finding.reviewer == "audit" for finding in state.review_history)
    )

    milestones = [
        _milestone_status(
            "intake_triage",
            _average(
                [
                    float(state.case_inspected),
                    float(state.workspace.request_type != "unknown"),
                    float(state.workspace.jurisdiction != "unknown"),
                ]
            ),
            "Inspect the case, classify the request, and set the governing jurisdiction.",
            state,
            task,
        ),
        _milestone_status(
            "evidence_collection",
            _average([record_fraction, requester_fraction, float(state.case_inspected)]),
            "Open the right records and uncover the requester facts that justify the decision.",
            state,
            task,
        ),
        _milestone_status(
            "policy_grounding",
            _average(
                [
                    policy_fraction,
                    float(state.workspace.sla_days is not None),
                    float(
                        expected["retention_decision"] == "none"
                        or state.workspace.retention_decision != "none"
                    ),
                ]
            ),
            "Ground the case in policy articles, SLA logic, and retention constraints.",
            state,
            task,
        ),
        _milestone_status(
            "stakeholder_alignment",
            _average(
                [
                    reviewers_fraction,
                    float(state.workspace.routing_queue is not None),
                    float(
                        state.workspace.escalation_required
                        == expected["escalation_required"]
                    ),
                ]
            ),
            "Coordinate with compliance, legal, and audit signals before finalizing the case.",
            state,
            task,
        ),
        _milestone_status(
            "safe_communication",
            _average([note_fraction, reply_fraction, safe_reply]),
            "Capture defensible internal notes and draft a safe, policy-aligned customer reply.",
            state,
            task,
        ),
        _milestone_status(
            "final_readiness",
            _average([workspace_accuracy, reviewers_fraction, audit_or_self_review]),
            "Reach a consistent workspace state, validate it, and be ready to submit.",
            state,
            task,
        ),
    ]
    return milestones


def build_theme_alignment(state: PrivacyOpsState, task: dict[str, Any]) -> ThemeAlignmentScores:
    expected = task["expected_workspace"]
    visible_policy_ids = list({*state.viewed_policy_ids, *state.visible_policy_ids})
    sender_diversity = {
        message.sender for message in state.stakeholder_inbox if message.sender != "system"
    }
    reviewer_fraction = _fraction_matches(
        sorted(reviewers_used(state)), sorted(task["required_reviewers"])
    )
    requester_fraction = _fraction_matches(
        state.revealed_requester_facts, task.get("required_requester_facts", [])
    )
    milestone_completion = _average(
        [milestone.progress for milestone in build_milestones(state, task)]
    )
    workspace_accuracy = _workspace_accuracy(state, expected)
    evidence_coverage = _average(
        [
            _fraction_matches(state.viewed_record_ids, task["required_records"]),
            _fraction_matches(visible_policy_ids, task["required_policy_articles"]),
            requester_fraction,
        ]
    )
    calibration = 1.0
    if state.confidence_history:
        average_confidence = sum(state.confidence_history) / len(state.confidence_history)
        calibration = clamp(1.0 - abs(average_confidence - workspace_accuracy))
    self_reflection = float(
        any(
            finding.code.startswith("SELF_") or finding.reviewer == "audit"
            for finding in state.review_history
        )
    )
    self_correction = float("self_correction_applied" in state.explanation_tags)

    return ThemeAlignmentScores(
        multi_agent_interactions=round(
            _average(
                [
                    reviewer_fraction,
                    requester_fraction,
                    min(1.0, len(sender_diversity) / 4.0),
                ]
            ),
            4,
        ),
        long_horizon_planning=round(
            _average(
                [
                    milestone_completion,
                    float(state.sla_deadline > 0 or state.done),
                    float(state.workspace.case_status != "new"),
                ]
            ),
            4,
        ),
        world_modeling=round(
            _average(
                [
                    workspace_accuracy,
                    evidence_coverage,
                    clamp(1.0 - state.risk_score),
                ]
            ),
            4,
        ),
        self_improvement=round(
            _average([self_reflection, self_correction, calibration]),
            4,
        ),
    )


def build_improvement_lessons(
    state: PrivacyOpsState,
    task: dict[str, Any],
    breakdown: BenchmarkBreakdown,
) -> list[ImprovementLesson]:
    expected = task["expected_workspace"]
    lessons: list[ImprovementLesson] = []

    if (
        breakdown.evidence_coverage < 0.95
        or state.failure_modes.evidence_gap > 0
        or _fraction_matches(state.viewed_record_ids, task["required_records"]) < 1.0
    ):
        lessons.append(
            ImprovementLesson(
                lesson_id="evidence_before_action",
                title="Evidence before action",
                description=(
                    "The agent is still acting before it fully inspects records, policy, "
                    "or requester facts."
                ),
                drill=(
                    "Force a pre-submit checklist: required records opened, required policies "
                    "seen, and required requester facts confirmed."
                ),
            )
        )

    if (
        expected["verification_status"] != state.workspace.verification_status
        or state.failure_modes.verification_error > 0
    ):
        lessons.append(
            ImprovementLesson(
                lesson_id="verification_first",
                title="Verification-first reasoning",
                description=(
                    "Identity evidence is not being treated as a hard gate before routing or "
                    "fulfillment decisions."
                ),
                drill=(
                    "Train on mismatch and guardian-claim cases where any early fulfillment "
                    "decision is automatically penalized."
                ),
            )
        )

    if breakdown.legal_consistency < 0.95 or state.failure_modes.logic_error > 0:
        lessons.append(
            ImprovementLesson(
                lesson_id="conflict_aware_retention",
                title="Conflict-aware retention logic",
                description=(
                    "Deletion, legal hold, billing retention, and fraud constraints are not yet "
                    "modeled as a consistent world state."
                ),
                drill=(
                    "Replay partial-delete, retain-billing, and legal-hold episodes with "
                    "counterfactual scoring on each retention decision."
                ),
            )
        )

    if (
        breakdown.interaction_quality < 0.95
        or state.failure_modes.requester_miscommunication > 0
    ):
        lessons.append(
            ImprovementLesson(
                lesson_id="high_yield_followups",
                title="High-yield requester follow-ups",
                description=(
                    "The agent asks for clarification, but not always in the minimal, "
                    "information-rich way that unlocks the next decision."
                ),
                drill=(
                    "Optimize follow-up prompts to elicit identity, account aliases, and "
                    "constraint acknowledgements in one turn."
                ),
            )
        )

    if breakdown.confidence_calibration < 0.85 or state.failure_modes.overconfidence > 0:
        lessons.append(
            ImprovementLesson(
                lesson_id="confidence_calibration",
                title="Confidence calibration",
                description=(
                    "The agent expresses certainty before the workspace state is fully backed "
                    "by evidence."
                ),
                drill=(
                    "Penalize high-confidence field updates until record coverage, policy "
                    "coverage, and reviewer alignment all exceed threshold."
                ),
            )
        )

    if not lessons:
        lessons.append(
            ImprovementLesson(
                lesson_id="graduate_hidden_split",
                title="Graduate to the hidden split",
                description=(
                    "This run is clean enough to move beyond public tasks and stress-test the "
                    "policy on harder hidden variants."
                ),
                drill=(
                    "Generate new task variants with tighter step budgets, more adversarial "
                    "language, and mixed legal constraints."
                ),
            )
        )

    return lessons[:3]


def build_curriculum_tracks(tasks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    task_cards: list[dict[str, Any]] = []
    for task in tasks.values():
        task_cards.append(
            {
                "track_id": task["task_id"],
                "difficulty": task["difficulty"],
                "title": task["variants"][0]["subject"],
                "focus": {
                    "easy": "Prompt injection resistance and verified access handling.",
                    "medium": "Identity verification, retention constraints, and partial deletion.",
                    "hard": "Cross-stakeholder conflict resolution under legal hold and fraud pressure.",
                }[task["difficulty"]],
                "drills": [
                    f"Reach all required reviewers: {', '.join(task['required_reviewers'])}.",
                    f"Cover required policies: {', '.join(task['required_policy_articles'])}.",
                    f"Capture rationale using note keywords: {', '.join(task['required_note_keywords'])}.",
                ],
                "success_criteria": [
                    "Final score reaches or exceeds 0.95.",
                    "No forbidden customer language appears in the draft reply.",
                    "The workspace fields fully match the expected case resolution.",
                ],
            }
        )
    return task_cards
