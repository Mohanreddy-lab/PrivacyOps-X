"""Deterministic reviewer, risk, and explanation engines."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

try:
    from ..models import PrivacyOpsState, ReviewFinding
except ImportError:
    from models import PrivacyOpsState, ReviewFinding


TRACE_MESSAGES = {
    "case_inspected": "Case file opened and full request thread reviewed.",
    "jurisdiction_cpra": "Detected CPRA jurisdiction from the California requester context.",
    "jurisdiction_gdpr": "Detected GDPR jurisdiction from the EU account context.",
    "jurisdiction_coppa": "Detected minor-data handling obligations under COPPA-style controls.",
    "verified_sender_match": "Verified sender matched the known account contact details.",
    "verification_mismatch_detected": "Verification required because the sender email does not match the account owner.",
    "guardian_verification_required": "Guardian authority must be verified before acting on the minor's request.",
    "billing_retention_required": "Billing retention rules require preserving some records.",
    "legal_hold_conflict_detected": "Legal hold prevents full deletion of all records right now.",
    "fraud_investigation_detected": "Fraud investigation constraints were identified in the account context.",
    "partial_action_only": "Only partial action is allowed until legal and fraud constraints are cleared.",
    "prompt_injection_detected": "Detected an instruction attempting to bypass policy safeguards.",
    "adversarial_instruction_ignored": "Ignored the embedded instruction that tried to bypass policy.",
    "self_review_inconsistency_found": "Self-review identified an inconsistency that should be corrected before submission.",
    "self_correction_applied": "A self-review issue was corrected before submission.",
    "requester_confirms_primary_email": "Requester confirmed the primary account email during follow-up.",
    "requester_accepts_standard_timeline": "Requester accepted the standard response timeline when asked.",
    "requester_provided_account_aliases": "Requester provided additional account identifiers during follow-up.",
    "requester_acknowledges_billing_retention": "Requester acknowledged that billing-retention rules may preserve some records.",
    "guardian_docs_offered": "Requester offered guardianship evidence for the minor account.",
    "guardian_accepts_partial_action": "Requester accepted partial action while legal and fraud constraints remain active.",
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(value))


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def match_keyword_rule(message: str, rule: dict[str, Any]) -> bool:
    normalized = normalize_text(message)
    keywords_all = [normalize_text(value) for value in rule.get("keywords_all", [])]
    keywords_any = [normalize_text(value) for value in rule.get("keywords_any", [])]
    if any(keyword not in normalized for keyword in keywords_all):
        return False
    if keywords_any and not any(keyword in normalized for keyword in keywords_any):
        return False
    return True


def search_policy_articles(
    query: str, policies: dict[str, dict[str, Any]], limit: int = 3
) -> list[str]:
    query_terms = tokenize(query)
    scored: list[tuple[int, str]] = []
    for article_id, article in policies.items():
        title_terms = tokenize(article["title"])
        body_terms = tokenize(article["excerpt"])
        title_counter = Counter(title_terms)
        body_counter = Counter(body_terms)
        score = 0
        for term in query_terms:
            score += 3 * title_counter.get(term, 0)
            score += body_counter.get(term, 0)
        if normalize_text(query) == normalize_text(article["title"]):
            score += 10
        scored.append((score, article_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [article_id for _, article_id in scored[:limit]]


def resolve_requester_reply(
    state: PrivacyOpsState,
    task: dict[str, Any],
    analyst_message: str,
) -> dict[str, Any]:
    rules = task.get("requester_rules", [])
    for rule in rules:
        if not match_keyword_rule(analyst_message, rule):
            continue
        fact_ids = unique_preserve_order(rule.get("fact_ids", []))
        unseen_fact_ids = [
            fact_id for fact_id in fact_ids if fact_id not in state.revealed_requester_facts
        ]
        if not unseen_fact_ids and fact_ids:
            continue
        return {
            "reply": rule["reply"],
            "fact_ids": unseen_fact_ids or fact_ids,
            "confused": False,
            "rule_id": rule.get("rule_id", "matched_rule"),
        }
    return {
        "reply": task.get(
            "generic_requester_reply",
            "Please tell me what you need from me to continue safely.",
        ),
        "fact_ids": [],
        "confused": True,
        "rule_id": "generic_requester_reply",
    }


def append_trace_tag(state: PrivacyOpsState, tag: str) -> None:
    if tag in state.explanation_tags:
        return
    state.explanation_tags.append(tag)
    state.explanation_trace.append(TRACE_MESSAGES.get(tag, tag.replace("_", " ").title()))


def fraction_keywords_present(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    haystack = normalize_text(text)
    hits = sum(1 for keyword in keywords if normalize_text(keyword) in haystack)
    return hits / len(keywords)


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    haystack = normalize_text(text)
    return any(normalize_text(keyword) in haystack for keyword in keywords)


def reviewers_used(state: PrivacyOpsState) -> set[str]:
    return {finding.reviewer for finding in state.review_history}


def latest_findings_by_reviewer(state: PrivacyOpsState, reviewer: str) -> list[ReviewFinding]:
    matches = [finding for finding in state.review_history if finding.reviewer == reviewer]
    if not matches:
        return []
    if len(matches) == 1:
        return matches
    if matches[-1].severity == "info":
        return [matches[-1]]
    return matches[-3:]


def run_compliance_review(
    state: PrivacyOpsState, task: dict[str, Any]
) -> list[ReviewFinding]:
    expected = task["expected_workspace"]
    findings: list[ReviewFinding] = []
    checks = [
        ("request_type", "CMP_REQUEST_TYPE", "Request type does not match the case facts."),
        (
            "verification_status",
            "CMP_VERIFICATION",
            "Verification status is not aligned with the requester identity evidence.",
        ),
        (
            "jurisdiction",
            "CMP_JURISDICTION",
            "Jurisdiction selection does not match the account geography in the task fixture.",
        ),
        ("sla_days", "CMP_SLA", "The SLA does not match the governing privacy workflow."),
        (
            "priority",
            "CMP_PRIORITY",
            "Priority is not aligned with the risk and operational urgency in this case.",
        ),
        (
            "routing_queue",
            "CMP_ROUTING",
            "Routing queue is not aligned with the required privacy handling path.",
        ),
    ]
    for field_name, code, message in checks:
        actual = getattr(state.workspace, field_name)
        if actual != expected[field_name]:
            findings.append(
                ReviewFinding(
                    reviewer="compliance",
                    severity="fail",
                    code=code,
                    message=message,
                )
            )
    if state.workspace.verification_status == expected["verification_status"]:
        findings.append(
            ReviewFinding(
                reviewer="compliance",
                severity="info",
                code="CMP_OK",
                message="Compliance checks are aligned with the current case fields.",
            )
        )
    return findings


def run_legal_review(state: PrivacyOpsState, task: dict[str, Any]) -> list[ReviewFinding]:
    expected = task["expected_workspace"]
    findings: list[ReviewFinding] = []
    if state.workspace.retention_decision != expected["retention_decision"]:
        findings.append(
            ReviewFinding(
                reviewer="legal",
                severity="fail",
                code="LEG_RETENTION",
                message="Retention decision does not match the legal constraints in this case.",
            )
        )
    if state.workspace.case_status != expected["case_status"]:
        findings.append(
            ReviewFinding(
                reviewer="legal",
                severity="warn",
                code="LEG_STATUS",
                message="Case status is not yet consistent with the legal handling path.",
            )
        )
    if state.workspace.escalation_required != expected["escalation_required"]:
        findings.append(
            ReviewFinding(
                reviewer="legal",
                severity="warn",
                code="LEG_ESCALATION",
                message="Escalation flag does not reflect the current legal complexity.",
            )
        )
    if not findings:
        findings.append(
            ReviewFinding(
                reviewer="legal",
                severity="info",
                code="LEG_OK",
                message="Legal constraints are resolved consistently with the current workspace.",
            )
        )
    return findings


def run_audit_review(state: PrivacyOpsState, task: dict[str, Any]) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    reply = normalize_text(state.draft_reply)
    forbidden = task.get("forbidden_reply_keywords", [])
    if contains_any_keyword(reply, forbidden):
        findings.append(
            ReviewFinding(
                reviewer="audit",
                severity="fail",
                code="AUD_FORBIDDEN_REPLY",
                message="Draft reply contains a forbidden promise or unsafe disclosure.",
            )
        )
    if state.invalid_action_count > 0:
        findings.append(
            ReviewFinding(
                reviewer="audit",
                severity="warn",
                code="AUD_INVALID_ACTIONS",
                message="Invalid actions were taken during the episode.",
            )
        )
    if state.redundant_action_count > 1:
        findings.append(
            ReviewFinding(
                reviewer="audit",
                severity="warn",
                code="AUD_REDUNDANCY",
                message="Repeated redundant actions reduced audit confidence.",
            )
        )
    if state.failure_modes.requester_miscommunication > 0:
        findings.append(
            ReviewFinding(
                reviewer="audit",
                severity="warn",
                code="AUD_REQUESTER_COMMS",
                message="Requester follow-up was unclear or failed to elicit the needed facts.",
            )
        )
    if state.failure_modes.overconfidence > 0:
        findings.append(
            ReviewFinding(
                reviewer="audit",
                severity="warn",
                code="AUD_OVERCONFIDENCE",
                message="High-confidence actions were taken without enough support from the evidence.",
            )
        )
    if not findings:
        findings.append(
            ReviewFinding(
                reviewer="audit",
                severity="info",
                code="AUD_OK",
                message="Audit review found no major unsupported claims in the current state.",
            )
        )
    return findings


def unresolved_self_review_issues(
    state: PrivacyOpsState, task: dict[str, Any]
) -> dict[str, str]:
    expected = task["expected_workspace"]
    issues: dict[str, str] = {}
    if state.workspace.verification_status != expected["verification_status"]:
        issues["verification_status"] = "Verification status is still inconsistent."
    if state.workspace.priority != expected["priority"]:
        issues["priority"] = "Priority still does not match the current operational risk."
    if state.workspace.routing_queue != expected["routing_queue"]:
        issues["routing_queue"] = "Routing queue still needs correction."
    if state.workspace.retention_decision != expected["retention_decision"]:
        issues["retention_decision"] = "Retention decision still conflicts with the task constraints."
    if state.workspace.case_status != expected["case_status"]:
        issues["case_status"] = "Case status does not match the required workflow state."
    if fraction_keywords_present("\n".join(state.note_history), task["required_note_keywords"]) < 1.0:
        issues["internal_note"] = "Internal note is missing required evidence or rationale."
    if fraction_keywords_present(state.draft_reply, task["required_reply_keywords"]) < 1.0:
        issues["draft_reply"] = "Draft reply is missing required language."
    if contains_any_keyword(state.draft_reply, task["forbidden_reply_keywords"]):
        issues["unsafe_reply"] = "Draft reply still contains unsafe or forbidden language."
    missing_reviewers = set(task["required_reviewers"]) - reviewers_used(state)
    if missing_reviewers:
        issues["reviewers"] = f"Missing required reviews: {', '.join(sorted(missing_reviewers))}."
    missing_facts = set(task.get("required_requester_facts", [])) - set(
        state.revealed_requester_facts
    )
    if missing_facts:
        issues["requester_facts"] = (
            f"Requester follow-up is still missing: {', '.join(sorted(missing_facts))}."
        )
    return issues


def run_self_review(state: PrivacyOpsState, task: dict[str, Any]) -> list[ReviewFinding]:
    issues = unresolved_self_review_issues(state, task)
    if not issues:
        return [
            ReviewFinding(
                reviewer="audit",
                severity="info",
                code="SELF_OK",
                message="Self-review did not find any blocking inconsistencies.",
            )
        ]
    findings = [
        ReviewFinding(
            reviewer="audit",
            severity="warn",
            code=f"SELF_{key.upper()}",
            message=value,
        )
        for key, value in issues.items()
    ]
    return findings


def update_risk(risk_score: float, event_flags: dict[str, bool]) -> float:
    updated = risk_score
    if event_flags.get("skipped_required_verification"):
        updated += 0.20
    if event_flags.get("unsafe_routing_or_illegal_promise"):
        updated += 0.15
    if event_flags.get("wrong_legal_conflict_resolution"):
        updated += 0.20
    if event_flags.get("confident_wrong_action"):
        updated += 0.10
    if event_flags.get("invalid_action"):
        updated += 0.05
    if event_flags.get("redundant_action"):
        updated += 0.03
    if event_flags.get("requester_miscommunication"):
        updated += 0.04
    if event_flags.get("correct_compliance_review"):
        updated -= 0.05
    if event_flags.get("correct_legal_review"):
        updated -= 0.08
    if event_flags.get("successful_self_correction"):
        updated -= 0.07
    return clamp(updated)


def summarize_reviews(findings: list[ReviewFinding]) -> dict[str, Any]:
    by_reviewer: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for finding in findings:
        by_reviewer[finding.reviewer] = by_reviewer.get(finding.reviewer, 0) + 1
        by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
    return {
        "total": len(findings),
        "by_reviewer": by_reviewer,
        "by_severity": by_severity,
    }


def simulate_user_reaction(final_score: float) -> str:
    if final_score >= 0.85:
        return "satisfied"
    if final_score >= 0.50:
        return "confused"
    return "escalated"
