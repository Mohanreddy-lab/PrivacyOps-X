"""Typed models for the PrivacyOps-X OpenEnv environment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from openenv.core.env_server.types import Action, Observation, State

WorkspaceFieldName = Literal[
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

MAX_ACTION_TARGET_ID_LENGTH = 128
MAX_ACTION_QUERY_LENGTH = 240
MAX_ACTION_CONTENT_LENGTH = 2_000
MAX_ACTION_FIELD_VALUE_LENGTH = 128


def _sanitize_text(
    value: str,
    *,
    max_length: int,
    allow_newlines: bool,
    field_name: str,
) -> str:
    cleaned = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        raise ValueError(f"{field_name}_empty")
    if len(cleaned) > max_length:
        raise ValueError(f"{field_name}_too_long")
    for character in cleaned:
        if ord(character) < 32 and character not in {"\n", "\t"}:
            raise ValueError(f"{field_name}_contains_control_characters")
    if not allow_newlines and any(character in {"\n", "\t"} for character in cleaned):
        raise ValueError(f"{field_name}_contains_invalid_whitespace")
    return cleaned


class WorkspaceView(BaseModel):
    request_type: Literal[
        "unknown",
        "access",
        "erasure",
        "access_erasure",
        "suppression",
    ] = "unknown"
    verification_status: Literal[
        "unknown",
        "verified",
        "verification_required",
        "rejected_identity",
    ] = "unknown"
    jurisdiction: Literal["unknown", "cpra", "gdpr", "coppa", "other"] = "unknown"
    sla_days: int | None = None
    priority: Literal["low", "medium", "high", "urgent"] | None = None
    routing_queue: Literal[
        "triage",
        "fulfillment",
        "manual_privacy_review",
        "privacy_legal",
        "fraud_privacy_joint",
    ] | None = None
    case_status: Literal[
        "new",
        "pending_verification",
        "approved",
        "partially_fulfilled",
        "escalated",
        "denied",
        "closed",
    ] = "new"
    retention_decision: Literal[
        "none",
        "retain_billing",
        "retain_legal_hold",
        "partial_delete",
        "suppress_marketing",
    ] = "none"
    escalation_required: bool = False
    confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)


class RecordView(BaseModel):
    record_id: str
    record_type: Literal["account", "billing", "fraud", "minor_profile"]
    summary: str
    flags: list[str] = Field(default_factory=list)


class PolicyArticleView(BaseModel):
    article_id: str
    title: str
    excerpt: str


class MessageTurn(BaseModel):
    turn_id: str
    role: Literal["requester", "analyst"]
    channel: Literal["ticket", "email"] = "email"
    message: str
    fact_ids: list[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    reviewer: Literal["compliance", "legal", "audit"]
    severity: Literal["info", "warn", "fail"]
    code: str
    message: str


class StakeholderMessage(BaseModel):
    message_id: str
    sender: Literal["requester", "compliance", "legal", "audit", "critic", "system"]
    severity: Literal["info", "warn", "fail"] = "info"
    message: str
    related_codes: list[str] = Field(default_factory=list)


class MilestoneStatus(BaseModel):
    milestone_id: str
    title: str
    status: Literal["available", "complete", "at_risk"] = "available"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str


class ThemeAlignmentScores(BaseModel):
    multi_agent_interactions: float = Field(default=0.0, ge=0.0, le=1.0)
    long_horizon_planning: float = Field(default=0.0, ge=0.0, le=1.0)
    world_modeling: float = Field(default=0.0, ge=0.0, le=1.0)
    self_improvement: float = Field(default=0.0, ge=0.0, le=1.0)


class ImprovementLesson(BaseModel):
    lesson_id: str
    title: str
    description: str
    drill: str


class FailureModes(BaseModel):
    hallucination: int = 0
    policy_violation: int = 0
    logic_error: int = 0
    unsafe_action: int = 0
    redundancy: int = 0
    verification_error: int = 0
    evidence_gap: int = 0
    overconfidence: int = 0
    requester_miscommunication: int = 0


class BenchmarkBreakdown(BaseModel):
    compliance_accuracy: float = Field(ge=0.0, le=1.0)
    safety_score: float = Field(ge=0.0, le=1.0)
    reasoning_quality: float = Field(ge=0.0, le=1.0)
    efficiency_score: float = Field(ge=0.0, le=1.0)
    legal_consistency: float = Field(ge=0.0, le=1.0)
    robustness_score: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    interaction_quality: float = Field(ge=0.0, le=1.0)
    confidence_calibration: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)


class PrivacyOpsAction(Action):
    action_type: Literal[
        "inspect_case",
        "open_record",
        "search_policy",
        "open_policy_article",
        "set_case_field",
        "add_internal_note",
        "draft_reply",
        "message_requester",
        "request_review",
        "self_review",
        "submit",
    ]
    target_id: str | None = None
    field_name: WorkspaceFieldName | None = None
    field_value: str | int | bool | None = None
    query: str | None = None
    content: str | None = None
    reviewer: Literal["compliance", "legal", "audit"] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = _sanitize_text(
            value,
            max_length=MAX_ACTION_TARGET_ID_LENGTH,
            allow_newlines=False,
            field_name="target_id",
        )
        if not all(character.isalnum() or character in "._:-" for character in cleaned):
            raise ValueError("target_id_contains_invalid_characters")
        return cleaned

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _sanitize_text(
            value,
            max_length=MAX_ACTION_QUERY_LENGTH,
            allow_newlines=False,
            field_name="query",
        )

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _sanitize_text(
            value,
            max_length=MAX_ACTION_CONTENT_LENGTH,
            allow_newlines=True,
            field_name="content",
        )

    @field_validator("field_value")
    @classmethod
    def validate_field_value(cls, value: str | int | bool | None) -> str | int | bool | None:
        if not isinstance(value, str):
            return value
        return _sanitize_text(
            value,
            max_length=MAX_ACTION_FIELD_VALUE_LENGTH,
            allow_newlines=False,
            field_name="field_value",
        )


class PrivacyOpsObservation(Observation):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    ticket_id: str
    ticket_summary: str
    workspace: WorkspaceView
    visible_records: list[RecordView] = Field(default_factory=list)
    visible_policy_articles: list[PolicyArticleView] = Field(default_factory=list)
    requester_thread: list[MessageTurn] = Field(default_factory=list)
    latest_requester_message: str | None = None
    revealed_requester_facts: list[str] = Field(default_factory=list)
    review_findings: list[ReviewFinding] = Field(default_factory=list)
    stakeholder_inbox: list[StakeholderMessage] = Field(default_factory=list)
    milestones: list[MilestoneStatus] = Field(default_factory=list)
    theme_alignment: ThemeAlignmentScores = Field(default_factory=ThemeAlignmentScores)
    explanation_trace: list[str] = Field(default_factory=list)
    last_action_result: str
    warning: str | None = None
    error: str | None = None
    draft_reply: str = ""
    improvement_lessons: list[ImprovementLesson] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0)
    steps_remaining: int = Field(ge=0)
    sla_deadline: int = Field(ge=0)
    urgency_level: Literal["low", "medium", "high"] = "low"
    user_reaction_preview: Literal[
        "unknown",
        "satisfied",
        "confused",
        "escalated",
    ] = "unknown"


class PrivacyOpsState(State):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    seed: int | None = None
    variant_id: str
    case_inspected: bool = False
    workspace: WorkspaceView
    viewed_record_ids: list[str] = Field(default_factory=list)
    viewed_policy_ids: list[str] = Field(default_factory=list)
    visible_policy_ids: list[str] = Field(default_factory=list)
    requester_thread: list[MessageTurn] = Field(default_factory=list)
    last_requester_message: str | None = None
    revealed_requester_facts: list[str] = Field(default_factory=list)
    confidence_history: list[float] = Field(default_factory=list)
    review_history: list[ReviewFinding] = Field(default_factory=list)
    stakeholder_inbox: list[StakeholderMessage] = Field(default_factory=list)
    milestones: list[MilestoneStatus] = Field(default_factory=list)
    theme_alignment: ThemeAlignmentScores = Field(default_factory=ThemeAlignmentScores)
    explanation_tags: list[str] = Field(default_factory=list)
    explanation_trace: list[str] = Field(default_factory=list)
    action_history: list[str] = Field(default_factory=list)
    audit_log: list[str] = Field(default_factory=list)
    note_history: list[str] = Field(default_factory=list)
    draft_reply: str = ""
    improvement_lessons: list[ImprovementLesson] = Field(default_factory=list)
    risk_score: float = Field(ge=0.0, le=1.0)
    sla_window_steps: int = Field(default=0, ge=0)
    sla_deadline: int = Field(default=0, ge=0)
    urgency_level: Literal["low", "medium", "high"] = "low"
    invalid_action_count: int = 0
    redundant_action_count: int = 0
    failure_modes: FailureModes = Field(default_factory=FailureModes)
    submitted: bool = False
    done: bool = False
    user_reaction: Literal["unknown", "satisfied", "confused", "escalated"] = (
        "unknown"
    )
    final_breakdown: BenchmarkBreakdown | None = None
