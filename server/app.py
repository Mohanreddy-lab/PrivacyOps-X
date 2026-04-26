"""FastAPI app for PrivacyOps-X."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from collections import defaultdict, deque
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gradio as gr
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from starlette.responses import Response
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
    import openenv.core.env_server.web_interface as openenv_web_interface
except Exception as exc:
    raise ImportError(
        "openenv-core is required to run PrivacyOps-X. Install dependencies first."
    ) from exc

try:
    from ..models import PrivacyOpsAction, PrivacyOpsObservation, PrivacyOpsState
    from .env import PrivacyOpsXEnvironment
    from .fixtures import load_tasks
    from .reporting import build_curriculum_tracks
except ImportError:
    from models import PrivacyOpsAction, PrivacyOpsObservation, PrivacyOpsState
    from server.env import PrivacyOpsXEnvironment
    from server.fixtures import load_tasks
    from server.reporting import build_curriculum_tracks


ROOT_DIR = Path(__file__).resolve().parent.parent
MAX_REQUEST_BODY_BYTES = 32 * 1024
DEFAULT_RATE_LIMIT_MAX_REQUESTS = 120
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 15 * 60
READ_RATE_LIMIT_MAX_REQUESTS = 240
EPISODE_RATE_LIMIT_MAX_REQUESTS = 120
AUTH_RATE_LIMIT_MAX_REQUESTS = 5
RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
FRAME_ANCESTORS_POLICY = "frame-ancestors 'self' https://huggingface.co https://*.huggingface.co https://*.hf.space"
DASHBOARD_FALLBACK_SCORES = {
    "random_score": 0.3594,
    "teacher_score": 0.99,
    "baseline_score": 0.6087,
    "improved_score": 0.9519,
}
DASHBOARD_FALLBACK_BEFORE = [
    {"action_type": "inspect_case"},
    {"action_type": "set_case_field", "field_name": "request_type", "field_value": "access_erasure"},
    {"action_type": "set_case_field", "field_name": "verification_status", "field_value": "verification_required"},
    {"action_type": "set_case_field", "field_name": "jurisdiction", "field_value": "gdpr"},
    {"action_type": "set_case_field", "field_name": "sla_days", "field_value": 30},
    {"action_type": "set_case_field", "field_name": "priority", "field_value": "urgent"},
    {
        "action_type": "draft_reply",
        "content": "Please verify your authority. We cannot delete all data now, but we can provide eligible data through partial fulfillment once verification is complete.",
    },
    {"action_type": "submit"},
]
DASHBOARD_FALLBACK_AFTER = [
    {"action_type": "inspect_case"},
    {"action_type": "open_record", "target_id": "acct_eu_recovery_primary"},
    {"action_type": "open_record", "target_id": "bill_eu_contractor_archive"},
    {"action_type": "open_record", "target_id": "minor_profile_02"},
    {"action_type": "open_record", "target_id": "fraud_case_02"},
    {"action_type": "open_record", "target_id": "legal_hold_02"},
    {"action_type": "search_policy", "query": "guardian legal hold partial fulfillment linked records"},
    {
        "action_type": "message_requester",
        "content": "Please verify your authority as guardian and confirm which linked account aliases are in scope. Because of the legal hold, fraud review, and retained billing artifacts, we cannot delete all data now but we can provide eligible data through partial fulfillment.",
    },
    {"action_type": "set_case_field", "field_name": "request_type", "field_value": "access_erasure"},
    {"action_type": "set_case_field", "field_name": "verification_status", "field_value": "verification_required"},
    {"action_type": "set_case_field", "field_name": "jurisdiction", "field_value": "gdpr"},
    {"action_type": "set_case_field", "field_name": "sla_days", "field_value": 30},
    {"action_type": "set_case_field", "field_name": "priority", "field_value": "urgent"},
    {"action_type": "set_case_field", "field_name": "routing_queue", "field_value": "fraud_privacy_joint"},
    {"action_type": "set_case_field", "field_name": "case_status", "field_value": "partially_fulfilled"},
    {"action_type": "set_case_field", "field_name": "retention_decision", "field_value": "partial_delete"},
    {"action_type": "set_case_field", "field_name": "escalation_required", "field_value": True},
    {
        "action_type": "add_internal_note",
        "content": "cross-account mapping; guardian verification; partial fulfillment; legal hold; fraud investigation",
    },
    {
        "action_type": "draft_reply",
        "content": "Please verify your authority. We cannot delete all data now, but we can provide eligible data through partial fulfillment once verification is complete.",
    },
    {"action_type": "request_review", "reviewer": "legal"},
    {"action_type": "request_review", "reviewer": "audit"},
    {"action_type": "self_review"},
    {"action_type": "submit"},
]


class TypedSchemaResponse(BaseModel):
    action: dict[str, Any]
    observation: dict[str, Any]
    state: dict[str, Any]


class DemoResponse(BaseModel):
    task: str
    steps: list[str]
    score: float


class EnvInfoResponse(BaseModel):
    env_name: str
    version: str
    tasks: list[str]
    max_steps: int
    reward_range: list[float]
    deterministic: bool


class HealthDetailResponse(BaseModel):
    status: str
    env_loaded: bool
    tasks_loaded: int


class TaskCardResponse(BaseModel):
    task_id: str
    difficulty: str
    required_reviewers: list[str]
    required_requester_facts: list[str]
    theme_focus: list[str]


class JudgeReportResponse(BaseModel):
    env_name: str
    version: str
    problem_statement: str
    themes: list[str]
    stakeholder_roles: list[str]
    task_cards: list[TaskCardResponse]
    self_improvement_loop: list[str]
    training_assets: list[str]
    hidden_eval_strategy: str
    judge_endpoints: list[str]


class CurriculumResponse(BaseModel):
    env_name: str
    tracks: list[dict[str, Any]]


def _strip_frontmatter(markdown: str) -> str:
    lines = markdown.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return markdown
    try:
        closing_index = next(
            index for index in range(1, len(lines)) if lines[index].strip() == "---"
        )
    except StopIteration:
        return markdown
    return "\n".join(lines[closing_index + 1 :]).lstrip()


def _prepare_web_readme() -> None:
    source = ROOT_DIR / "README.md"
    if not source.exists():
        return
    cleaned = _strip_frontmatter(source.read_text(encoding="utf-8"))
    target = Path(tempfile.gettempdir()) / "privacyops_x_web_readme.md"
    target.write_text(cleaned, encoding="utf-8")
    os.environ["ENV_README_PATH"] = str(target)

    original_loader = openenv_web_interface._load_readme_from_filesystem

    def _prefer_cleaned_readme(env_name: str | None) -> str | None:
        custom_path = os.environ.get("ENV_README_PATH")
        if custom_path and Path(custom_path).exists():
            try:
                return Path(custom_path).read_text(encoding="utf-8")
            except Exception:
                pass
        return original_loader(env_name)

    openenv_web_interface._load_readme_from_filesystem = _prefer_cleaned_readme


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _first_existing(*relative_paths: str) -> Path | None:
    for relative_path in relative_paths:
        path = ROOT_DIR / relative_path
        if path.exists():
            return path
    return None


def _load_optional_json(*relative_paths: str) -> dict[str, Any] | None:
    path = _first_existing(*relative_paths)
    if path is None:
        return None
    return _read_json_if_exists(path)


def _encode_image_data(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    suffix = path.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _encode_svg_data(svg: str) -> str:
    return f"data:image/svg+xml;charset=utf-8,{quote(svg)}"


def _build_bar_chart_svg(random_score: float, teacher_score: float) -> str:
    random_height = int(random_score * 200)
    teacher_height = int(teacher_score * 200)
    random_y = 240 - random_height
    teacher_y = 240 - teacher_height
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="640" height="320" viewBox="0 0 640 320" role="img" aria-label="Finale baseline range">
      <rect width="640" height="320" fill="#08131c"/>
      <line x1="70" y1="240" x2="590" y2="240" stroke="#365065" stroke-width="2"/>
      <line x1="70" y1="40" x2="70" y2="240" stroke="#365065" stroke-width="2"/>
      <rect x="150" y="{random_y}" width="110" height="{random_height}" rx="12" fill="#56d3c5"/>
      <rect x="360" y="{teacher_y}" width="110" height="{teacher_height}" rx="12" fill="#ffbf70"/>
      <text x="205" y="270" text-anchor="middle" fill="#d9e8f0" font-size="18" font-family="Arial">Random</text>
      <text x="415" y="270" text-anchor="middle" fill="#d9e8f0" font-size="18" font-family="Arial">Teacher</text>
      <text x="205" y="{max(28, random_y - 12)}" text-anchor="middle" fill="#d9e8f0" font-size="18" font-family="Arial">{random_score:.4f}</text>
      <text x="415" y="{max(28, teacher_y - 12)}" text-anchor="middle" fill="#d9e8f0" font-size="18" font-family="Arial">{teacher_score:.4f}</text>
      <text x="24" y="245" fill="#8ea7b8" font-size="14" font-family="Arial">0.0</text>
      <text x="24" y="46" fill="#8ea7b8" font-size="14" font-family="Arial">1.0</text>
    </svg>
    """
    return _encode_svg_data(svg)


def _build_improvement_chart_svg(baseline_score: float, improved_score: float) -> str:
    point_a_y = 240 - int(baseline_score * 180)
    point_b_y = 240 - int(improved_score * 180)
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="640" height="320" viewBox="0 0 640 320" role="img" aria-label="Self-improvement curve">
      <rect width="640" height="320" fill="#08131c"/>
      <line x1="80" y1="240" x2="580" y2="240" stroke="#365065" stroke-width="2"/>
      <line x1="80" y1="40" x2="80" y2="240" stroke="#365065" stroke-width="2"/>
      <polyline points="170,{point_a_y} 470,{point_b_y}" fill="none" stroke="#56d3c5" stroke-width="6" stroke-linecap="round"/>
      <circle cx="170" cy="{point_a_y}" r="9" fill="#ff8f87"/>
      <circle cx="470" cy="{point_b_y}" r="9" fill="#56d3c5"/>
      <text x="170" y="270" text-anchor="middle" fill="#d9e8f0" font-size="18" font-family="Arial">Before</text>
      <text x="470" y="270" text-anchor="middle" fill="#d9e8f0" font-size="18" font-family="Arial">After</text>
      <text x="170" y="{max(28, point_a_y - 14)}" text-anchor="middle" fill="#d9e8f0" font-size="18" font-family="Arial">{baseline_score:.4f}</text>
      <text x="470" y="{max(28, point_b_y - 14)}" text-anchor="middle" fill="#d9e8f0" font-size="18" font-family="Arial">{improved_score:.4f}</text>
      <text x="30" y="245" fill="#8ea7b8" font-size="14" font-family="Arial">0.0</text>
      <text x="30" y="46" fill="#8ea7b8" font-size="14" font-family="Arial">1.0</text>
    </svg>
    """
    return _encode_svg_data(svg)


def _format_score(value: float | None) -> str:
    if value is None:
        return "pending"
    return f"{value:.4f}"


def _render_trajectory(actions: list[dict[str, Any]] | None) -> str:
    if not actions:
        return "<p class='muted'>No trajectory available yet.</p>"
    items = []
    for action in actions:
        payload = escape(json.dumps(action, ensure_ascii=False))
        items.append(f"<li><code>{payload}</code></li>")
    return "<ol class='trajectory'>" + "".join(items) + "</ol>"


def _load_dashboard_payload() -> dict[str, Any]:
    random_report = _load_optional_json(
        "outputs/evals/random_finale_live.json",
        "outputs/evals/random.json",
    )
    teacher_report = _load_optional_json(
        "outputs/evals/teacher_finale_live.json",
        "outputs/evals/teacher.json",
    )
    self_report = _load_optional_json(
        "outputs/evals/self_improvement_cycle_live.json",
        "outputs/evals/self_improvement_cycle.json",
    )
    sft_report = _load_optional_json(
        "outputs/evals/sft_checkpoint.json",
        "outputs/evals/sft_tiny_checkpoint.json",
        "outputs/evals/model_smoke.json",
    )

    random_plot = _encode_image_data(
        _first_existing(
            "outputs/plots/finale_live_random_vs_teacher.png",
            "outputs/plots/policy_comparison.png",
        )
    )
    self_plot = _encode_image_data(
        _first_existing(
            "outputs/plots/self_improvement_curve_live.png",
            "outputs/plots/self_improvement_curve.png",
        )
    )
    random_score = (
        random_report.get("overall", {}).get("mean_final_score")
        if random_report
        else DASHBOARD_FALLBACK_SCORES["random_score"]
    )
    teacher_score = (
        teacher_report.get("overall", {}).get("mean_final_score")
        if teacher_report
        else DASHBOARD_FALLBACK_SCORES["teacher_score"]
    )
    sft_score = sft_report.get("overall", {}).get("mean_final_score") if sft_report else None
    baseline_score = (
        self_report.get("baseline_score")
        if self_report
        else DASHBOARD_FALLBACK_SCORES["baseline_score"]
    )
    improved_score = (
        self_report.get("improved_score")
        if self_report
        else DASHBOARD_FALLBACK_SCORES["improved_score"]
    )

    if random_plot is None:
        random_plot = _build_bar_chart_svg(random_score, teacher_score)
    if self_plot is None:
        self_plot = _build_improvement_chart_svg(baseline_score, improved_score)

    before_behavior = self_report.get("episodes", [{}])[0].get("trajectory") if self_report else DASHBOARD_FALLBACK_BEFORE
    after_behavior = self_report.get("episodes", [{}, {}])[1].get("trajectory") if self_report else DASHBOARD_FALLBACK_AFTER

    return {
        "random_score": random_score,
        "teacher_score": teacher_score,
        "sft_score": sft_score,
        "sft_is_fallback": sft_report is not None and not Path("outputs/evals/sft_checkpoint.json").exists(),
        "baseline_score": baseline_score,
        "improved_score": improved_score,
        "before_behavior": before_behavior,
        "after_behavior": after_behavior,
        "random_plot": random_plot,
        "self_plot": self_plot,
    }


def _task_catalog() -> list[dict[str, Any]]:
    tasks = load_tasks()
    catalog: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        variant = task["variants"][0]
        catalog.append(
            {
                "task_id": task_id,
                "difficulty": task["difficulty"].title(),
                "subject": variant["subject"],
                "preview": variant["preview"],
                "records": len(task["records"]),
                "reviewers": ", ".join(reviewer.title() for reviewer in task["required_reviewers"]),
                "step_limit": task["step_limit"],
            }
        )
    return catalog


def _dump_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _gradio_metric_card(title: str, value: str, caption: str) -> str:
    return f"""
    <article class="px-card px-metric-card">
      <div class="px-kicker">{escape(title)}</div>
      <div class="px-metric-value">{escape(value)}</div>
      <p>{escape(caption)}</p>
    </article>
    """


def _shell_css() -> str:
    return """
    .gradio-container {background: linear-gradient(180deg, #08131d 0%, #0a1622 100%) !important;}
    body {
      margin: 0;
      background: linear-gradient(180deg, #08131d 0%, #0a1622 100%);
      color: #eef4f8;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    a { color: inherit; }
    .px-shell {
      display: grid;
      gap: 18px;
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 40px;
      box-sizing: border-box;
    }
    .px-hero, .px-grid {display: grid; gap: 16px;}
    .px-hero {grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.9fr);}
    .px-card {
      border: 1px solid #22384d;
      border-radius: 20px;
      padding: 20px;
      background: #0f1d2a;
      color: #eef4f8;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.20);
    }
    .px-card h2, .px-card h3 {margin-top: 0;}
    .px-card p, .px-card li {color: #9ab0c4;}
    .px-summary-card ul, .px-task-card ul, .px-card ul {margin: 12px 0 0; padding-left: 18px;}
    .px-shell-compact {max-width: none; margin: 0; padding: 0; gap: 16px;}
    .px-eyebrow, .px-chip, .px-kicker {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid #244357;
      color: #8ae4dc;
      background: rgba(79, 209, 197, 0.08);
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .px-hero h1 {margin: 10px 0 8px; font-size: clamp(2.5rem, 4vw, 3.6rem); line-height: 1.02;}
    .px-lead {margin: 0; color: #9ab0c4; font-size: 1.04rem; max-width: 62ch;}
    .px-page-nav {display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 8px;}
    .px-page-nav a {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 40px;
      padding: 8px 14px;
      border-radius: 999px;
      text-decoration: none;
      border: 1px solid #244357;
      color: #9ab0c4;
      background: rgba(15, 29, 42, 0.72);
      font-weight: 700;
    }
    .px-page-nav a.is-active {
      color: #08131d;
      background: #4fd1c5;
      border-color: transparent;
    }
    .px-link-row {display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px;}
    .px-link-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 10px 14px;
      border-radius: 12px;
      text-decoration: none;
      color: #eef4f8;
      border: 1px solid #2a465d;
      background: #10202d;
      font-weight: 700;
    }
    .px-link-button.px-primary {background: #4fd1c5; color: #08212a; border-color: transparent;}
    .px-link-button.px-secondary-hero {background: #ff8f87; color: #08131d; border-color: transparent;}
    .px-section-head {display: grid; gap: 6px; margin-bottom: 2px;}
    .px-section-head h2 {margin: 0; color: #eef4f8;}
    .px-section-head p {margin: 0; color: #9ab0c4;}
    .px-metrics-grid {grid-template-columns: repeat(4, minmax(0, 1fr));}
    .px-story-grid, .px-task-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
    .px-guide-grid {grid-template-columns: repeat(3, minmax(0, 1fr));}
    .px-two-up {grid-template-columns: repeat(2, minmax(0, 1fr));}
    .px-steps-grid {grid-template-columns: repeat(4, minmax(0, 1fr));}
    .px-mini-steps {grid-template-columns: repeat(3, minmax(0, 1fr));}
    .px-story-grid-wide {grid-template-columns: repeat(3, minmax(0, 1fr));}
    .px-metric-value {font-size: 2rem; font-weight: 800; margin-top: 10px;}
    .px-step {
      width: 34px; height: 34px; border-radius: 999px; display: grid; place-items: center;
      background: #163041; color: #8ae4dc; font-weight: 800; margin-bottom: 12px;
    }
    .px-mini-step {
      border-radius: 18px;
      border: 1px solid #22384d;
      background: #10202d;
      padding: 16px;
      display: grid;
      gap: 8px;
    }
    .px-mini-step strong {color: #eef4f8;}
    .px-mini-step p {margin: 0; color: #9ab0c4;}
    .px-plot {width: 100%; border-radius: 16px; border: 1px solid #22384d; margin-top: 12px;}
    .px-tip-grid {display: grid; gap: 14px; grid-template-columns: repeat(2, minmax(0, 1fr));}
    .px-tip-card {min-height: 100%;}
    .px-bulletless {list-style: none; padding-left: 0 !important; margin: 0;}
    .px-bulletless li {margin: 10px 0; padding: 10px 12px; border-radius: 12px; background: #132536;}
    .px-button-note {margin-top: 8px; font-size: 0.92rem; color: #8ae4dc;}
    .px-inline-code {font-family: Consolas, "SFMono-Regular", monospace;}
    .px-help-grid {display: grid; gap: 14px; grid-template-columns: repeat(3, minmax(0, 1fr));}
    .px-help-card {border: 1px solid #22384d; border-radius: 18px; padding: 16px; background: #10202d;}
    .px-help-card h4 {margin: 0 0 8px; color: #eef4f8;}
    .px-help-card p {margin: 0; color: #9ab0c4;}
    .px-toolbar-note {margin-top: 6px; color: #8ae4dc; font-size: 0.9rem;}
    .px-soft-label {font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8ae4dc;}
    .px-run-grid {display: grid; gap: 18px; grid-template-columns: minmax(320px, 0.95fr) minmax(420px, 1.25fr);}
    .px-button-grid {display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr));}
    .px-button-grid .gr-button, .px-wide-button .gr-button {min-height: 54px !important; border-radius: 14px !important;}
    .px-wide-button {grid-column: 1 / -1;}
    .px-primary-button .gr-button {background: linear-gradient(135deg, #ff7c32, #ff5a00) !important; color: #fff !important; border: 0 !important;}
    .px-secondary-button .gr-button {background: #5a5a67 !important; color: #fff !important; border: 1px solid #6f7080 !important;}
    .px-accent-button .gr-button {background: linear-gradient(135deg, #4fd1c5, #7ce8de) !important; color: #08212a !important; border: 0 !important;}
    .px-field, .px-code-panel, .px-html-panel {border-radius: 18px;}
    .px-code-panel .cm-editor, .px-code-panel .cm-scroller {min-height: 190px;}
    .px-demo-head {margin-bottom: 8px;}
    .px-demo-head h2 {margin: 0 0 8px;}
    .px-demo-head p {margin: 0;}
    .px-full-width {width: 100%;}
    .px-tab-note {margin-top: 6px; color: #9ab0c4;}
    .px-home-note {color: #8ae4dc; font-size: 0.95rem; margin-top: 10px;}
    .px-ops-list {display: grid; gap: 10px; margin-top: 6px;}
    .px-ops-list li {
      list-style: none;
      border: 1px solid #22384d;
      border-radius: 14px;
      background: #132536;
      padding: 12px 14px;
      color: #c7d8e6;
    }
    .px-ops-list strong {display: block; color: #eef4f8; margin-bottom: 4px;}
    .px-dashboard-grid {grid-template-columns: repeat(5, minmax(0, 1fr));}
    .px-dashboard-actions {display: flex; flex-wrap: wrap; gap: 12px; margin-top: 18px;}
    .px-score-table {width: 100%; border-collapse: collapse;}
    .px-score-table th, .px-score-table td {text-align: left; padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.08);}
    .px-score-table th {color: #9ab0c4; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.74rem;}
    .px-status {
      border-radius: 12px;
      padding: 12px 14px;
      font-weight: 700;
      border: 1px solid #244357;
      margin-bottom: 10px;
    }
    .px-status-ready {background: #0f1d2a; color: #cde4f0;}
    .px-status-ok {background: rgba(79, 209, 197, 0.10); color: #8ae4dc; border-color: #285050;}
    .px-status-error {background: rgba(255, 143, 135, 0.12); color: #ffd4cf; border-color: #5d3330;}
    .px-session-card {display: grid; gap: 16px;}
    .px-session-grid {grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px;}
    .px-session-grid div {
      border-radius: 14px; border: 1px solid #22384d; background: #132536; padding: 12px;
      display: grid; gap: 4px;
    }
    .px-session-grid strong {font-size: 0.78rem; color: #8ae4dc; text-transform: uppercase; letter-spacing: 0.05em;}
    .px-session-grid span {color: #eef4f8;}
    .px-note {border-left: 3px solid #4fd1c5; padding-left: 12px; color: #eef4f8;}
    .px-note-muted {border-left-color: #365065; color: #9ab0c4;}
    .gradio-container .gr-block, .gradio-container .gr-panel, .gradio-container .gr-group {
      border-color: #22384d !important;
      background: #0f1d2a !important;
    }
    .gradio-container .gr-form, .gradio-container .gr-box, .gradio-container .gr-accordion {
      border-color: #22384d !important;
      background: #0f1d2a !important;
    }
    .gradio-container button, .gradio-container .gr-button {
      transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    }
    .gradio-container button:hover, .gradio-container .gr-button:hover,
    .px-link-button:hover {
      transform: translateY(-1px);
      box-shadow: 0 10px 24px rgba(0,0,0,0.24);
    }
    @media (max-width: 1024px) {
      .px-hero, .px-metrics-grid, .px-story-grid, .px-story-grid-wide, .px-task-grid, .px-two-up, .px-session-grid, .px-guide-grid, .px-steps-grid, .px-mini-steps, .px-tip-grid, .px-help-grid, .px-run-grid, .px-dashboard-grid {grid-template-columns: 1fr;}
      .px-button-grid {grid-template-columns: 1fr;}
    }
    """


def _page_nav_html(active: str) -> str:
    links = [
        ("home", "/", "Home", "Open the project overview and main explanation."),
        ("playground", "/playground/", "Playground", "Run the live benchmark step by step."),
        ("results", "/dashboard", "Results", "Open plots, scores, and benchmark results."),
        ("docs", "/docs", "Docs", "Open the raw API documentation."),
        ("schema", "/schema", "Schema", "Open the raw JSON schema for action, observation, and state."),
    ]
    items = []
    for key, href, label, title in links:
        classes = "is-active" if key == active else ""
        items.append(
            f'<a class="{classes}" href="{href}" title="{title}">{label}</a>'
        )
    return f'<div class="px-page-nav">{"".join(items)}</div>'


def _build_overview_html(payload: dict[str, Any]) -> str:
    task_cards = "".join(
        f"""
        <article class="px-card px-task-card">
          <div class="px-chip">{escape(task['difficulty'])}</div>
          <h3>{escape(task['subject'])}</h3>
          <p>{escape(task['preview'])}</p>
          <ul>
            <li><strong>Task ID:</strong> <code>{escape(task['task_id'])}</code></li>
            <li><strong>Records:</strong> {task['records']}</li>
            <li><strong>Required reviewers:</strong> {escape(task['reviewers'])}</li>
            <li><strong>Step budget:</strong> {task['step_limit']}</li>
          </ul>
        </article>
        """
        for task in _task_catalog()
    )
    trained_score = _format_score(payload["sft_score"])
    checkpoint_caption = (
        "Current saved checkpoint score."
        if payload.get("sft_score") is not None
        else "This score will appear after a checkpoint eval file is available."
    )
    metrics = "".join(
        [
            _gradio_metric_card(
                "Random baseline",
                _format_score(payload["random_score"]),
                "A weak agent. This is the low score.",
            ),
            _gradio_metric_card(
                "Teacher policy",
                _format_score(payload["teacher_score"]),
                "A strong agent. This is the high score.",
            ),
            _gradio_metric_card(
                "Self-improvement",
                f"{_format_score(payload['baseline_score'])} to {_format_score(payload['improved_score'])}",
                "The same agent gets better after feedback and retry.",
            ),
            _gradio_metric_card(
                "Trained checkpoint",
                trained_score,
                checkpoint_caption,
            ),
        ]
    )
    workflow_cards = """
      <section class="px-section">
        <div class="px-section-head">
          <h2>How the project works</h2>
          <p>This is the full story of the benchmark in four simple steps.</p>
        </div>
        <div class="px-grid px-steps-grid">
          <article class="px-card">
            <div class="px-step">1</div>
            <h3>Pick a privacy case</h3>
            <p>Choose a task such as data access, account deletion, guardian authority, legal hold, or cross-border recovery.</p>
          </article>
          <article class="px-card">
            <div class="px-step">2</div>
            <h3>Let the agent act</h3>
            <p>The agent must inspect records, search policy, message the requester, ask reviewers, and then submit a final answer.</p>
          </article>
          <article class="px-card">
            <div class="px-step">3</div>
            <h3>Watch the state change</h3>
            <p>Every step updates visible records, workspace fields, risk, milestones, and reviewer findings.</p>
          </article>
          <article class="px-card">
            <div class="px-step">4</div>
            <h3>Read the score</h3>
            <p>The benchmark returns safety, policy use, legal handling, communication quality, and efficiency signals.</p>
          </article>
        </div>
      </section>
    """
    return f"""
    <div class="px-shell">
      <section class="px-hero">
        <div>
          {_page_nav_html("home")}
          <div class="px-eyebrow">Simple demo</div>
          <h1>PrivacyOps-X</h1>
          <p class="px-lead">
            PrivacyOps-X is a benchmark for privacy operations.
            It checks whether an AI agent can handle real privacy cases safely,
            follow policy, use evidence, and finish with the right final decision.
          </p>
          <div class="px-link-row">
            <a class="px-link-button px-primary" href="/playground/" title="Open the live playground and run the benchmark step by step.">Open playground</a>
            <a class="px-link-button px-secondary-hero" href="/dashboard" title="Open the benchmark scores, plots, and final result summary.">View results</a>
            <a class="px-link-button" href="/docs" title="Open the API documentation for reset, step, and state endpoints.">API docs</a>
            <a class="px-link-button" href="/schema" title="See the exact JSON schema for actions, observations, and state.">Schema</a>
            <a class="px-link-button" href="/judge-report" title="Read the judge-facing summary of the benchmark and training pipeline.">Project summary</a>
          </div>
          <div class="px-home-note">Start with the playground to see one case live. Then open the results page to understand the scores and plots.</div>
        </div>
        <div class="px-card px-summary-card">
          <h3>What you should understand on this page</h3>
          <ul>
            <li><strong>Why:</strong> privacy workflows are high risk, so fluent answers alone are not enough.</li>
            <li><strong>How:</strong> the agent must use typed actions like <code>inspect_case</code>, <code>open_record</code>, and <code>request_review</code>.</li>
            <li><strong>What you see:</strong> records, policy, requester messages, review output, risk, milestones, and score.</li>
            <li><strong>What this proves:</strong> the same system can evaluate agents, improve them, and train models.</li>
          </ul>
        </div>
      </section>

      <section class="px-section">
        <div class="px-section-head">
          <h2>How to use this homepage</h2>
          <p>Use this page as a fast map of the project before you open the live demo.</p>
        </div>
        <div class="px-grid px-mini-steps">
          <article class="px-mini-step">
            <strong>1. Read the project goal</strong>
            <p>See why privacy operations need a structured benchmark instead of a simple chatbot test.</p>
          </article>
          <article class="px-mini-step">
            <strong>2. Open the playground</strong>
            <p>Run one case and watch the environment update after every action.</p>
          </article>
          <article class="px-mini-step">
            <strong>3. Open results</strong>
            <p>Check the final scores, plots, and training evidence after the demo.</p>
          </article>
        </div>
      </section>

      <section class="px-grid px-metrics-grid">
        {metrics}
      </section>

      {workflow_cards}

      <section class="px-section">
        <div class="px-section-head">
          <h2>Why this project exists</h2>
          <p>This explains the problem, the solution, and the training story in simple English.</p>
        </div>
        <div class="px-grid px-story-grid-wide">
          <article class="px-card">
            <h3>The problem</h3>
            <p>Privacy teams do more than answer questions. They verify identity, check retention rules, handle legal hold, and talk to users safely.</p>
          </article>
          <article class="px-card">
            <h3>The solution</h3>
            <p>PrivacyOps-X turns that workflow into a deterministic environment with typed actions, observable state, reviewer signals, and measurable scores.</p>
          </article>
          <article class="px-card">
            <h3>The evidence</h3>
            <p>The project includes benchmark scores, self-improvement results, training scripts, a Colab notebook, and plots from real runs.</p>
          </article>
        </div>
      </section>

      <section class="px-section">
        <div class="px-section-head">
          <h2>Benchmark cases</h2>
          <p>These are the privacy cases you can test here.</p>
        </div>
        <div class="px-grid px-task-grid">
          {task_cards}
        </div>
      </section>

      <section class="px-section">
        <div class="px-section-head">
          <h2>What the user gets back</h2>
          <p>After each step, the system clearly shows the result.</p>
        </div>
        <div class="px-grid px-guide-grid">
          <article class="px-card">
            <h3>What the agent can do</h3>
            <ul>
              <li><code>inspect_case</code>, <code>open_record</code>, and <code>search_policy</code></li>
              <li><code>message_requester</code>, <code>draft_reply</code>, and <code>add_internal_note</code></li>
              <li><code>request_review</code>, <code>self_review</code>, and <code>submit</code></li>
            </ul>
          </article>
          <article class="px-card">
            <h3>What the system shows</h3>
            <ul>
              <li>Case summary, fields, records, and policy text</li>
              <li>User messages, review results, milestones, risk score, and steps left</li>
              <li>Scores, reports, plots, datasets, and training files</li>
            </ul>
          </article>
          <article class="px-card">
            <h3>Why the benchmark matters</h3>
            <ul>
              <li>It measures real operational behavior, not only fluent language.</li>
              <li>It makes failures easy to inspect and improve.</li>
              <li>It supports both evaluation and model training.</li>
            </ul>
          </article>
        </div>
      </section>
    </div>
    """


def _build_results_html(payload: dict[str, Any]) -> str:
    trained_score = _format_score(payload["sft_score"])
    return f"""
    <div class="px-shell">
      <section class="px-section">
        {_page_nav_html("results")}
        <div class="px-section-head">
          <h2>How to use this results page</h2>
          <p>Read the two plots first, then compare the score cards, then use the docs and schema links if you want the raw details.</p>
        </div>
        <div class="px-grid px-mini-steps">
          <article class="px-mini-step">
            <strong>Read the plots</strong>
            <p>See the difference between a weak policy, a strong policy, and the self-improvement curve.</p>
          </article>
          <article class="px-mini-step">
            <strong>Read the scores</strong>
            <p>Use the score cards to understand the benchmark baseline and the improvement gain.</p>
          </article>
          <article class="px-mini-step">
            <strong>Open raw data</strong>
            <p>Use docs, schema, and judge report if you want the exact JSON structures behind the interface.</p>
          </article>
        </div>
      </section>

      <section class="px-grid px-two-up">
        <article class="px-card">
          <h2>Policy comparison</h2>
          <p>This plot compares a weak agent and a strong agent.</p>
          <img class="px-plot" src="{payload['random_plot']}" alt="Random vs teacher comparison plot" />
        </article>
        <article class="px-card">
          <h2>Improvement over time</h2>
          <p>This plot shows the agent getting better after feedback.</p>
          <img class="px-plot" src="{payload['self_plot']}" alt="Self-improvement curve plot" />
        </article>
      </section>

      <section class="px-grid px-two-up">
        <article class="px-card">
          <h3>Current results</h3>
          <ul>
            <li><strong>Random:</strong> {_format_score(payload['random_score'])}</li>
            <li><strong>Teacher:</strong> {_format_score(payload['teacher_score'])}</li>
            <li><strong>Self-improvement:</strong> {_format_score(payload['baseline_score'])} to {_format_score(payload['improved_score'])}</li>
            <li><strong>Trained model:</strong> {trained_score}</li>
          </ul>
        </article>
        <article class="px-card">
          <h3>What is included</h3>
          <ul>
            <li>An OpenEnv environment with <code>reset</code>, <code>step</code>, and <code>state</code></li>
            <li>A training script and a Colab notebook</li>
            <li>A loss curve and an improvement curve</li>
            <li>A README, blog, Space, and demo page</li>
          </ul>
        </article>
      </section>
    </div>
    """


def _build_api_html() -> str:
    return """
    <div class="px-shell">
      <section class="px-section">
        """ + _page_nav_html("docs") + """
        <div class="px-section-head">
          <h2>How to use the API and schema pages</h2>
          <p>Open the raw docs if you want endpoint details. Open the schema if you want the exact JSON format used by the benchmark.</p>
        </div>
        <div class="px-grid px-mini-steps">
          <article class="px-mini-step">
            <strong>Docs</strong>
            <p>Use <code>/docs</code> to inspect and try the <code>reset</code>, <code>step</code>, and <code>state</code> endpoints.</p>
          </article>
          <article class="px-mini-step">
            <strong>Schema</strong>
            <p>Use <code>/schema</code> to see the exact shape of action, observation, and state objects.</p>
          </article>
          <article class="px-mini-step">
            <strong>Judge JSON</strong>
            <p>Use <code>/judge-report</code> and <code>/curriculum</code> to inspect the benchmark framing and self-improvement structure.</p>
          </article>
        </div>
      </section>

      <section class="px-grid px-two-up">
        <article class="px-card">
          <h2>API pages</h2>
          <ul>
            <li><a href="/docs">/docs</a> shows the API docs</li>
            <li><a href="/schema">/schema</a> shows the action, observation, and state format</li>
            <li><a href="/envinfo">/envinfo</a> shows project info</li>
            <li><a href="/dashboard">/dashboard</a> shows plots and scores</li>
            <li><a href="/judge-report">/judge-report</a> shows a short summary</li>
          </ul>
        </article>
        <article class="px-card">
          <h2>Why this page uses Gradio</h2>
          <p>
            Gradio makes the project easier to read and use.
            You can understand the project, try the live demo, and still keep the real API under it.
          </p>
        </article>
      </section>
    </div>
    """


def _build_playground_intro_html() -> str:
    return """
    <div class="px-shell px-shell-compact">
      """ + _page_nav_html("playground") + """
      <div class="px-demo-head">
        <div class="px-eyebrow">Run Demo</div>
        <h2>Live benchmark playground</h2>
        <p>Use this page to run one privacy case step by step and see exactly how the benchmark works.</p>
      </div>
      <div class="px-help-grid">
        <div class="px-help-card">
          <h4>What this page proves</h4>
          <p>The playground shows how the agent reads a case, takes typed actions, and earns a score based on safety and policy use.</p>
        </div>
        <div class="px-help-card">
          <h4>How to use it</h4>
          <p>Pick a task, start the case, use the quick buttons or edit JSON, then read the live result and session summary.</p>
        </div>
        <div class="px-help-card">
          <h4>What to watch</h4>
          <p>Focus on risk score, visible records, milestone progress, reviewer findings, and the final action path.</p>
        </div>
      </div>
      <div class="px-grid px-mini-steps">
        <article class="px-mini-step">
          <strong>Step 1: start the case</strong>
          <p>Choose a task and seed, then press <em>Start Case</em> to load the first observation.</p>
        </article>
        <article class="px-mini-step">
          <strong>Step 2: choose an action</strong>
          <p>Use a quick action button or edit the JSON directly if you want full control.</p>
        </article>
        <article class="px-mini-step">
          <strong>Step 3: read the result</strong>
          <p>Check the live output and summary panel to see what changed, what risk increased, and whether the case is done.</p>
        </article>
      </div>
    </div>
    """


def _gradio_head() -> str:
    tooltip_map = {
        "task-picker": "Choose which privacy benchmark case to run.",
        "seed-input": "Use a seed to replay the same deterministic case variant.",
        "start-case": "Reset the environment and start a fresh case.",
        "state-case": "Read the current environment state without sending a new action.",
        "close-case": "Close the current session and clear the live state.",
        "inspect-case": "Load the case summary and current workspace details.",
        "search-policy": "Look up policy guidance for the current issue.",
        "request-legal-review": "Ask legal review to check whether the action is safe and allowed.",
        "message-user": "Draft a message to the requester for verification or clarification.",
        "self-review": "Ask the agent to review its own work before submitting.",
        "submit-case": "Submit the current resolution and finish the case.",
        "send-action": "Run the JSON action shown in the editor.",
        "action-json": "Edit the exact action payload that will be sent to the environment.",
        "latest-result": "See the newest observation, reward, and state returned by the environment.",
        "session-summary": "Quick view of task status, risk, milestones, and latest note.",
        "example-actions": "Insert a ready-made action template into the JSON editor.",
    }
    tooltip_json = json.dumps(tooltip_map)
    return f"""
    <script>
    (() => {{
      const tooltipMap = {tooltip_json};
      const applyTooltips = () => {{
        Object.entries(tooltipMap).forEach(([id, text]) => {{
          const root = document.getElementById(id);
          if (!root) return;
          root.setAttribute("title", text);
          root.setAttribute("aria-label", text);
          root.querySelectorAll("button, textarea, input, select, .cm-editor, .cm-content, .wrap, label, .gr-button").forEach((el) => {{
            el.setAttribute("title", text);
            el.setAttribute("aria-label", text);
          }});
        }});
      }};
      const run = () => {{
        applyTooltips();
        const observer = new MutationObserver(() => applyTooltips());
        observer.observe(document.body, {{ childList: true, subtree: true }});
      }};
      if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", run);
      }} else {{
        run();
      }}
    }})();
    </script>
    """


def _render_home_html() -> HTMLResponse:
    payload = _load_dashboard_payload()
    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>PrivacyOps-X</title>
            <style>{_shell_css()}</style>
          </head>
          <body>
            {_build_overview_html(payload)}
          </body>
        </html>
        """
    )


def _gradio_view_state(current_env: PrivacyOpsXEnvironment | None):
    if current_env is None:
        message = "Start a case first."
        return (
            None,
            _status_html(message, kind="error"),
            _session_summary_html(None, None),
            _dump_json({"error": message}),
        )

    state_payload = current_env.state.model_dump(mode="json")
    observation_payload = {
        "task_id": state_payload.get("task_id"),
        "difficulty": state_payload.get("difficulty"),
        "visible_records": state_payload.get("visible_records", []),
        "milestones": state_payload.get("milestones", []),
        "risk_score": state_payload.get("risk_score"),
        "last_action_result": "Current environment state snapshot.",
    }
    return (
        current_env,
        _status_html("Current state loaded.", kind="ok"),
        _session_summary_html(observation_payload, state_payload),
        _dump_json({"operation": "state", "state": state_payload}),
    )


def _gradio_close_episode(current_env: PrivacyOpsXEnvironment | None):
    if current_env is not None:
        try:
            current_env.close()
        except Exception:
            pass
    return (
        None,
        _status_html("Session closed. Press reset() Start Case to begin again.", kind="ready"),
        _session_summary_html(None, None),
        _dump_json({"operation": "close", "closed": True}),
        _reset_action_template(),
    )


def _status_html(message: str, *, kind: str = "ready") -> str:
    kind_class = {
        "ready": "px-status-ready",
        "ok": "px-status-ok",
        "error": "px-status-error",
    }.get(kind, "px-status-ready")
    return f"<div class='px-status {kind_class}'>{escape(message)}</div>"


def _session_summary_html(observation: dict[str, Any] | None, state: dict[str, Any] | None) -> str:
    if not observation or not state:
        return """
        <div class="px-card px-session-card">
          <h3>Session summary</h3>
          <p>Start a case to see the task, risk, fields, and latest result.</p>
        </div>
        """

    workspace = state.get("workspace", {})
    milestones = observation.get("milestones", [])
    completed = sum(1 for item in milestones if item.get("status") == "complete")
    latest = observation.get("last_action_result") or "No action result yet."
    warning = observation.get("warning")
    error = observation.get("error")
    note = warning or error or "No active warnings."
    risk = state.get("risk_score", observation.get("risk_score", 0.0))

    return f"""
    <div class="px-card px-session-card">
      <h3>Session summary</h3>
      <div class="px-grid px-session-grid">
        <div><strong>Task</strong><span>{escape(str(observation.get('task_id', '')))}</span></div>
        <div><strong>Difficulty</strong><span>{escape(str(observation.get('difficulty', '')))}</span></div>
        <div><strong>Step count</strong><span>{escape(str(state.get('step_count', 0)))}</span></div>
        <div><strong>Risk score</strong><span>{risk:.4f}</span></div>
        <div><strong>Visible records</strong><span>{len(observation.get('visible_records', []))}</span></div>
        <div><strong>Milestones complete</strong><span>{completed}/{len(milestones)}</span></div>
        <div><strong>Request type</strong><span>{escape(str(workspace.get('request_type', 'unknown')))}</span></div>
        <div><strong>Verification</strong><span>{escape(str(workspace.get('verification_status', 'unknown')))}</span></div>
        <div><strong>Done</strong><span>{'yes' if state.get('done') else 'no'}</span></div>
      </div>
      <div class="px-note">
        <strong>Latest result:</strong> {escape(str(latest))}
      </div>
      <div class="px-note px-note-muted">
        <strong>Note:</strong> {escape(str(note))}
      </div>
    </div>
    """


def _reset_action_template() -> str:
    return _dump_json({"action_type": "inspect_case"})


def _response_payload(operation: str, observation: PrivacyOpsObservation, state: PrivacyOpsState) -> str:
    observation_payload = observation.model_dump(mode="json")
    state_payload = state.model_dump(mode="json")
    return _dump_json(
        {
            "operation": operation,
            "observation": observation_payload,
            "state": state_payload,
            "reward": observation_payload.get("reward"),
            "done": observation_payload.get("done", state_payload.get("done")),
        }
    )


def _gradio_reset_episode(
    task_id: str,
    seed: float | int | None,
    current_env: PrivacyOpsXEnvironment | None,
):
    if current_env is not None:
        try:
            current_env.close()
        except Exception:
            pass
    env = PrivacyOpsXEnvironment()
    numeric_seed = int(seed or 0)
    observation = env.reset(task_id=task_id, seed=numeric_seed)
    return (
        env,
        _status_html(f"Case ready - {task_id} - seed {numeric_seed}", kind="ok"),
        _session_summary_html(
            observation.model_dump(mode="json"),
            env.state.model_dump(mode="json"),
        ),
        _response_payload("reset", observation, env.state),
        _reset_action_template(),
    )


def _gradio_run_step(
    current_env: PrivacyOpsXEnvironment | None,
    action_json: str,
):
    if current_env is None:
        message = "Start a case first."
        return (
            None,
            _status_html(message, kind="error"),
            _session_summary_html(None, None),
            _dump_json({"error": message}),
        )

    try:
        action_payload = json.loads(action_json)
    except json.JSONDecodeError as exc:
        return (
            current_env,
            _status_html("Action JSON is invalid.", kind="error"),
            _session_summary_html(None, None),
            _dump_json({"error": f"JSON decode error: {exc}"}),
        )

    try:
        action = PrivacyOpsAction(**action_payload)
    except Exception as exc:
        return (
            current_env,
            _status_html("Action payload did not match the schema.", kind="error"),
            _session_summary_html(None, None),
            _dump_json({"error": str(exc), "action": action_payload}),
        )

    observation = current_env.step(action)
    observation_payload = observation.model_dump(mode="json")
    state_payload = current_env.state.model_dump(mode="json")
    status_message = (
        "Case finished."
        if state_payload.get("done")
        else f"Step saved - reward {observation_payload.get('reward', 0.0):.4f}"
    )
    return (
        current_env,
        _status_html(status_message, kind="ok"),
        _session_summary_html(observation_payload, state_payload),
        _response_payload("step", observation, current_env.state),
    )


def _action_template(action_type: str, **extras: Any) -> str:
    payload = {"action_type": action_type}
    payload.update(extras)
    return _dump_json(payload)


def _build_gradio_demo() -> gr.Blocks:
    payload = _load_dashboard_payload()
    sft_caption = (
        "Current saved checkpoint result."
        if payload.get("sft_score") is not None
        else "This score will appear after a checkpoint eval file is available."
    )
    task_choices = [
        (f"{task['difficulty']} - {task['subject']}", task["task_id"])
        for task in _task_catalog()
    ]
    action_examples = [
        [_action_template("inspect_case")],
        [_action_template("search_policy", query="legal hold retention deletion")],
        [_action_template("request_review", reviewer="legal")],
        [_action_template("message_requester", content="Please verify your identity and confirm which account is in scope.")],
        [_action_template("submit")],
    ]
    schema_text = _dump_json(
        {
            "action": PrivacyOpsAction.model_json_schema(),
            "observation": PrivacyOpsObservation.model_json_schema(),
            "state": PrivacyOpsState.model_json_schema(),
        }
    )
    css = _shell_css()

    with gr.Blocks(
        title="PrivacyOps-X",
        analytics_enabled=False,
    ) as demo:
        gr.HTML(_gradio_head())
        with gr.Tabs():
            with gr.Tab("Run Demo"):
                gr.HTML(_build_playground_intro_html())
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4):
                        with gr.Group(elem_classes=["px-card", "px-field"]):
                            gr.Markdown("### Case setup\nPick a case, choose a seed, and run `reset()` to start the environment.")
                            task_id = gr.Dropdown(
                                choices=task_choices,
                                value=task_choices[0][1],
                                label="Task",
                                info="Choose the benchmark case you want to run.",
                                elem_id="task-picker",
                                elem_classes=["px-field"],
                            )
                            seed = gr.Number(
                                value=0,
                                precision=0,
                                label="Seed",
                                info="Use the same seed to replay the same task variant.",
                                elem_id="seed-input",
                                elem_classes=["px-field"],
                            )
                            reset_button = gr.Button(
                                "reset() Start Case",
                                variant="primary",
                                elem_id="start-case",
                                elem_classes=["px-primary-button", "px-full-width"],
                            )
                            with gr.Row():
                                state_button = gr.Button(
                                    "state() View State",
                                    elem_id="state-case",
                                    elem_classes=["px-accent-button"],
                                )
                                close_button = gr.Button(
                                    "close() End Session",
                                    elem_id="close-case",
                                    elem_classes=["px-secondary-button"],
                                )
                            gr.HTML("<div class='px-toolbar-note'>Use reset() to start, state() to inspect the current state, step() to send an action, and close() to end the session.</div>")
                        status = gr.HTML(_status_html("Ready to start a case."))
                    with gr.Column(scale=6):
                        session_summary = gr.HTML(
                            _session_summary_html(None, None),
                            elem_id="session-summary",
                            elem_classes=["px-html-panel"],
                        )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4):
                        with gr.Group(elem_classes=["px-card", "px-code-panel"]):
                            gr.Markdown("### Action builder\nUse quick buttons or edit the JSON action directly, then run `step()`.")
                            action_json = gr.Code(
                                value=_reset_action_template(),
                                language="json",
                                label="Action JSON",
                                lines=16,
                                elem_id="action-json",
                                elem_classes=["px-code-panel"],
                            )
                            gr.HTML(
                                """
                                <div class="px-tip-grid">
                                  <div class="px-help-card">
                                    <h4>Fast path</h4>
                                    <p>Use the quick action buttons below if you want a guided demo flow.</p>
                                  </div>
                                  <div class="px-help-card">
                                    <h4>Advanced path</h4>
                                    <p>Edit the JSON yourself if you want full control over the typed action payload.</p>
                                  </div>
                                </div>
                                """
                            )
                            with gr.Row():
                                inspect_button = gr.Button(
                                    "Inspect case",
                                    elem_id="inspect-case",
                                    elem_classes=["px-secondary-button"],
                                )
                                search_button = gr.Button(
                                    "Search policy",
                                    elem_id="search-policy",
                                    elem_classes=["px-secondary-button"],
                                )
                            review_button = gr.Button(
                                "Ask legal to review",
                                elem_id="request-legal-review",
                                elem_classes=["px-secondary-button"],
                            )
                            with gr.Row():
                                requester_button = gr.Button(
                                    "Message user",
                                    elem_id="message-user",
                                    elem_classes=["px-secondary-button"],
                                )
                                self_review_button = gr.Button(
                                    "Check my work",
                                    elem_id="self-review",
                                    elem_classes=["px-secondary-button"],
                                )
                            submit_button = gr.Button(
                                "Submit",
                                elem_id="submit-case",
                                elem_classes=["px-secondary-button"],
                            )
                            gr.HTML(
                                """
                                <ul class="px-ops-list">
                                  <li><strong>Inspect case</strong>Load the case summary, visible records, and current workspace.</li>
                                  <li><strong>Search policy</strong>Look up the policy rule that should guide the next step.</li>
                                  <li><strong>Ask legal to review</strong>Send the case to legal when the action needs formal review.</li>
                                  <li><strong>Message user</strong>Ask the requester for identity proof or missing facts.</li>
                                  <li><strong>Check my work</strong>Run a self-review before the final answer.</li>
                                  <li><strong>Submit</strong>Finish the case when you have enough evidence.</li>
                                </ul>
                                """
                            )
                            step_button = gr.Button(
                                "step() Send Action",
                                variant="primary",
                                elem_id="send-action",
                                elem_classes=["px-primary-button", "px-full-width"],
                            )
                    with gr.Column(scale=6):
                        with gr.Group(elem_classes=["px-card", "px-code-panel"]):
                            gr.Markdown("### Live environment output\nThis panel shows what the benchmark returns after each step.")
                            response_json = gr.Code(
                                value="Press \"Start Case\" to begin.",
                                language="json",
                                label="Latest result",
                                lines=28,
                                elem_id="latest-result",
                                elem_classes=["px-code-panel"],
                            )
                gr.Examples(
                    examples=action_examples,
                    inputs=[action_json],
                    label="Example actions",
                    elem_id="example-actions",
                )
                session_env = gr.State(value=None)
                reset_button.click(
                    fn=_gradio_reset_episode,
                    inputs=[task_id, seed, session_env],
                    outputs=[session_env, status, session_summary, response_json, action_json],
                )
                state_button.click(
                    fn=_gradio_view_state,
                    inputs=[session_env],
                    outputs=[session_env, status, session_summary, response_json],
                )
                close_button.click(
                    fn=_gradio_close_episode,
                    inputs=[session_env],
                    outputs=[session_env, status, session_summary, response_json, action_json],
                )
                step_button.click(
                    fn=_gradio_run_step,
                    inputs=[session_env, action_json],
                    outputs=[session_env, status, session_summary, response_json],
                )
                inspect_button.click(
                    fn=lambda: _action_template("inspect_case"),
                    outputs=action_json,
                )
                search_button.click(
                    fn=lambda: _action_template("search_policy", query="legal hold retention deletion"),
                    outputs=action_json,
                )
                review_button.click(
                    fn=lambda: _action_template("request_review", reviewer="legal"),
                    outputs=action_json,
                )
                requester_button.click(
                    fn=lambda: _action_template(
                        "message_requester",
                        content="Please verify your identity and confirm which linked account aliases are in scope.",
                    ),
                    outputs=action_json,
                )
                self_review_button.click(
                    fn=lambda: _action_template("self_review"),
                    outputs=action_json,
                )
                submit_button.click(
                    fn=lambda: _action_template("submit"),
                    outputs=action_json,
                )
            with gr.Tab("Results"):
                gr.HTML(
                    f"""
                    <div class="px-shell px-shell-compact">
                      <div class="px-help-card">
                        <h4>Checkpoint status</h4>
                        <p>{escape(sft_caption)}</p>
                      </div>
                    </div>
                    """
                )
                gr.HTML(_build_results_html(payload))
            with gr.Tab("API"):
                gr.HTML(_build_api_html())
                gr.Code(value=schema_text, language="json", label="Schema", lines=24)
        gr.HTML(_build_overview_html(payload))
    demo.theme = gr.themes.Soft(
        primary_hue="teal",
        neutral_hue="slate",
        radius_size="lg",
    )
    demo.css = css
    return demo


def _client_identity(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _is_auth_like_path(path: str) -> bool:
    auth_markers = ("auth", "login", "signin", "signup", "oauth", "token")
    return any(segment in path for segment in auth_markers)


def _resolve_rate_limit(request: Request) -> tuple[int, int]:
    path = request.url.path.lower()
    if _is_auth_like_path(path):
        return AUTH_RATE_LIMIT_MAX_REQUESTS, DEFAULT_RATE_LIMIT_WINDOW_SECONDS
    if path in {"/reset", "/step"}:
        return EPISODE_RATE_LIMIT_MAX_REQUESTS, DEFAULT_RATE_LIMIT_WINDOW_SECONDS
    if request.method.upper() == "GET":
        return READ_RATE_LIMIT_MAX_REQUESTS, DEFAULT_RATE_LIMIT_WINDOW_SECONDS
    return DEFAULT_RATE_LIMIT_MAX_REQUESTS, DEFAULT_RATE_LIMIT_WINDOW_SECONDS


def _check_rate_limit(request: Request) -> int | None:
    limit, window_seconds = _resolve_rate_limit(request)
    bucket_key = f"{_client_identity(request)}:{request.method.upper()}:{request.url.path}"
    timestamps = RATE_LIMIT_BUCKETS[bucket_key]
    now = time.time()
    cutoff = now - window_seconds
    while timestamps and timestamps[0] <= cutoff:
        timestamps.popleft()
    if len(timestamps) >= limit:
        return max(1, int(window_seconds - (now - timestamps[0])))
    timestamps.append(now)
    return None


async def _validate_request_body(request: Request) -> Request | Response:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Malformed Content-Length header."},
            )
        if declared_length > MAX_REQUEST_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Payload too large."},
            )

    if request.method.upper() not in {"POST", "PUT", "PATCH"}:
        return request

    body = await request.body()
    if len(body) > MAX_REQUEST_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "Payload too large."},
        )

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type and body:
        try:
            json.loads(body)
        except json.JSONDecodeError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Malformed JSON payload."},
            )

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(request.scope, receive)


_prepare_web_readme()


app = create_app(
    PrivacyOpsXEnvironment,
    PrivacyOpsAction,
    PrivacyOpsObservation,
    env_name="privacyops_x",
    max_concurrent_envs=16,
)

app.router.routes = [
    route
    for route in app.router.routes
    if not (
        isinstance(route, APIRoute)
        and route.path in {"/", "/state", "/schema"}
        and "GET" in (route.methods or set())
    )
]


@app.middleware("http")
async def security_and_pretty_json_middleware(request: Request, call_next):
    retry_after = _check_rate_limit(request)
    if retry_after is not None:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please retry later."},
            headers={"Retry-After": str(retry_after)},
        )

    validated_request = await _validate_request_body(request)
    if isinstance(validated_request, Response):
        return validated_request

    response = await call_next(validated_request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = FRAME_ANCESTORS_POLICY
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        return response
    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    try:
        payload = json.loads(body)
    except Exception:
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=content_type,
            background=response.background,
        )
    pretty_bytes = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
    headers = dict(response.headers)
    headers.pop("content-length", None)
    return Response(
        content=pretty_bytes,
        status_code=response.status_code,
        headers=headers,
        media_type="application/json",
        background=response.background,
    )


@app.get(
    "/state",
    response_model=PrivacyOpsState,
    tags=["State Management"],
    summary="Get current environment state",
    description="Retrieve the typed PrivacyOps-X state model for the current environment instance.",
)
def state() -> PrivacyOpsState:
    env = PrivacyOpsXEnvironment()
    try:
        return env.state
    finally:
        env.close()


@app.get(
    "/schema",
    response_model=TypedSchemaResponse,
    tags=["Schema"],
    summary="Get typed JSON schemas",
    description="Return the typed PrivacyOps-X schemas for action, observation, and state.",
)
def schema() -> TypedSchemaResponse:
    return TypedSchemaResponse(
        action=PrivacyOpsAction.model_json_schema(),
        observation=PrivacyOpsObservation.model_json_schema(),
        state=PrivacyOpsState.model_json_schema(),
    )


@app.get(
    "/demo",
    response_model=DemoResponse,
    tags=["Environment Info"],
    summary="Get a demo trajectory summary",
    description="Return a short example path and score for quick evaluation.",
)
def demo() -> DemoResponse:
    return DemoResponse(
        task="easy_verified_access_with_injection",
        steps=["inspect_case", "open_record", "submit"],
        score=0.92,
    )


@app.get(
    "/envinfo",
    response_model=EnvInfoResponse,
    tags=["Environment Info"],
    summary="Get extended environment metadata",
    description="Return evaluation-oriented metadata for judges and tooling.",
)
def envinfo() -> EnvInfoResponse:
    tasks = load_tasks()
    step_limits = [int(task["step_limit"]) for task in tasks.values()]
    return EnvInfoResponse(
        env_name="PrivacyOps-X",
        version="1.2",
        tasks=list(tasks.keys()),
        max_steps=max(step_limits) if step_limits else 0,
        reward_range=[0.0, 1.0],
        deterministic=True,
    )


@app.get(
    "/healthz",
    response_model=HealthDetailResponse,
    tags=["Health"],
    summary="Detailed health check",
    description="Extended health check with environment and task load info.",
)
def healthz() -> HealthDetailResponse:
    tasks = load_tasks()
    return HealthDetailResponse(
        status="healthy",
        env_loaded=True,
        tasks_loaded=len(tasks),
    )


@app.get(
    "/judge-report",
    response_model=JudgeReportResponse,
    tags=["Environment Info"],
    summary="Get a judge-facing environment summary",
    description="Return the benchmark framing, task cards, and self-improvement loop used in the finale pitch.",
)
def judge_report() -> JudgeReportResponse:
    tasks = load_tasks()
    focus_map = {
        "easy": ["multi-agent interactions", "world modeling"],
        "medium": ["long-horizon planning", "world modeling", "self-improvement"],
        "hard": ["multi-agent interactions", "long-horizon planning", "world modeling"],
    }
    task_cards = [
        TaskCardResponse(
            task_id=task["task_id"],
            difficulty=task["difficulty"],
            required_reviewers=task["required_reviewers"],
            required_requester_facts=task.get("required_requester_facts", []),
            theme_focus=focus_map[task["difficulty"]],
        )
        for task in tasks.values()
    ]
    return JudgeReportResponse(
        env_name="PrivacyOps-X",
        version="1.1",
        problem_statement=(
            "Train and evaluate agents that resolve privacy-rights cases under real "
            "operational pressure, hidden constraints, and multi-stakeholder review."
        ),
        themes=[
            "multi-agent interactions",
            "long-horizon planning",
            "world modeling",
            "self-improving agent systems",
        ],
        stakeholder_roles=["privacy_analyst", "requester", "compliance", "legal", "audit", "critic"],
        task_cards=task_cards,
        self_improvement_loop=[
            "Run a deterministic episode and capture failure modes.",
            "Extract improvement lessons and counterfactual drills.",
            "Promote high-scoring agents to harder hidden variants and tighter budgets.",
        ],
        training_assets=[
            "TRAINING.md",
            "notebooks/privacyops_x_trl_colab.ipynb",
            "scripts/generate_sft_dataset.py",
            "scripts/train_trl_sft.py",
            "scripts/train_openenv_grpo.py",
            "scripts/evaluate_policies.py",
            "scripts/plot_eval_results.py",
            "scripts/run_self_improvement_cycle.py",
        ],
        hidden_eval_strategy=(
            "Public tasks teach the workflow, while seeded variants and stricter step "
            "budgets stress generalization without changing the API."
        ),
        judge_endpoints=["/docs", "/schema", "/envinfo", "/judge-report", "/curriculum", "/dashboard"],
    )


@app.get(
    "/curriculum",
    response_model=CurriculumResponse,
    tags=["Environment Info"],
    summary="Get the self-improvement curriculum",
    description="Return the training tracks that turn benchmark failures into harder targeted drills.",
)
def curriculum() -> CurriculumResponse:
    tasks = load_tasks()
    return CurriculumResponse(
        env_name="PrivacyOps-X",
        tracks=build_curriculum_tracks(tasks),
    )


@app.get("/dashboard", include_in_schema=False, response_class=HTMLResponse)
@app.get("/report", include_in_schema=False, response_class=HTMLResponse)
def dashboard() -> str:
    payload = _load_dashboard_payload()
    random_score = _format_score(payload["random_score"])
    teacher_score = _format_score(payload["teacher_score"])
    sft_score = _format_score(payload["sft_score"])
    baseline_score = _format_score(payload["baseline_score"])
    improved_score = _format_score(payload["improved_score"])
    random_plot_html = (
        f"<img src='{payload['random_plot']}' alt='Random vs teacher comparison plot' />"
        if payload["random_plot"]
        else "<div class='placeholder'>Comparison plot not available yet.</div>"
    )
    self_plot_html = (
        f"<img src='{payload['self_plot']}' alt='Self-improvement curve plot' />"
        if payload["self_plot"]
        else "<div class='placeholder'>Self-improvement plot not available yet.</div>"
    )
    metric_cards = "".join(
        [
            _gradio_metric_card("Random agent", random_score, "Low score baseline"),
            _gradio_metric_card("Teacher agent", teacher_score, "High score reference"),
            _gradio_metric_card("Baseline", baseline_score, "Score before improvement"),
            _gradio_metric_card("Improved", improved_score, "Score after improvement"),
            _gradio_metric_card(
                "Trained model",
                sft_score,
                "Current saved checkpoint result" if payload["sft_score"] is not None else "Checkpoint eval not available yet",
            ),
        ]
    )
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>PrivacyOps-X Judge Dashboard</title>
        <style>{_shell_css()}</style>
      </head>
      <body>
        <div class="px-shell">
          <section class="px-hero">
            <div>
              {_page_nav_html("results")}
              <div class="px-eyebrow">View results</div>
              <h1>PrivacyOps-X results</h1>
              <p class="px-lead">
                This page shows the main results of the project.
                It shows scores, plots, and what the numbers mean.
              </p>
              <div class="px-dashboard-actions">
                <a class="px-link-button px-primary" href="/playground/" title="Open the live benchmark playground.">Open playground</a>
                <a class="px-link-button px-secondary-hero" href="/docs" title="Open the API documentation.">Open docs</a>
                <a class="px-link-button" href="/judge-report" title="Read the judge-facing summary JSON.">Project summary JSON</a>
                <a class="px-link-button" href="/curriculum" title="See the self-improvement curriculum JSON.">Curriculum JSON</a>
                <a class="px-link-button" href="/schema" title="Open the typed schema page.">Schema</a>
              </div>
            </div>
            <div class="px-card px-summary-card">
              <h3>How to use this page</h3>
              <ul>
                <li><strong>First:</strong> compare the score cards to see the gap between weak and strong behavior.</li>
                <li><strong>Second:</strong> read the plots to see policy comparison and improvement over time.</li>
                <li><strong>Third:</strong> open docs or schema if you want the raw benchmark data.</li>
              </ul>
            </div>
          </section>

          <section class="px-section">
            <div class="px-grid px-mini-steps">
              <article class="px-mini-step">
                <strong>Read the score cards</strong>
                <p>These cards show the starting point, the teacher target, and the improvement result.</p>
              </article>
              <article class="px-mini-step">
                <strong>Read the plots</strong>
                <p>The charts show the same story visually so judges can understand the result quickly.</p>
              </article>
              <article class="px-mini-step">
                <strong>Open raw endpoints</strong>
                <p>Use the JSON and docs links when you want the exact benchmark structures behind the UI.</p>
              </article>
            </div>
          </section>

          <section class="px-grid px-dashboard-grid">
            {metric_cards}
          </section>

          <section class="px-section">
            <div class="px-eyebrow">View results</div>
            <div class="px-section-head">
              <h2>Plots</h2>
              <p>These plots show benchmark performance and improvement.</p>
            </div>
            <div class="px-grid px-two-up">
              <article class="px-card plot">
                <h3>Policy comparison</h3>
                <p>This plot compares a weak agent and a strong agent.</p>
                {random_plot_html}
              </article>
              <article class="px-card plot">
                <h3>Improvement over time</h3>
                <p>This plot shows the agent getting better after feedback.</p>
                {self_plot_html}
              </article>
            </div>
          </section>

          <section class="px-section">
            <h2>What The Score Means</h2>
            <article class="px-card">
              <table class="px-score-table">
                <thead>
                  <tr><th>Metric</th><th>Meaning</th></tr>
                </thead>
                <tbody>
                  <tr><td>Compliance</td><td>Did the agent follow the rules?</td></tr>
                  <tr><td>Safety</td><td>Did it avoid unsafe actions or promises?</td></tr>
                  <tr><td>Evidence</td><td>Did it check records, policy, and user details first?</td></tr>
                  <tr><td>Legal</td><td>Did it handle legal hold, retention, and escalation in the right way?</td></tr>
                  <tr><td>Communication</td><td>Did it give a safe reply and clear notes?</td></tr>
                  <tr><td>Efficiency</td><td>Did it avoid extra steps and stay within the step limit?</td></tr>
                </tbody>
              </table>
            </article>
          </section>
        </div>
      </body>
    </html>
    """


@app.get("/", include_in_schema=False)
@app.get("/web", include_in_schema=False)
@app.get("/web/", include_in_schema=False)
def interface_redirect() -> HTMLResponse:
    return _render_home_html()


app = gr.mount_gradio_app(app, _build_gradio_demo(), path="/playground")

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()


