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
    sft_report = _load_optional_json("outputs/evals/sft_checkpoint.json")

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
    .px-section-head {display: grid; gap: 6px; margin-bottom: 2px;}
    .px-section-head h2 {margin: 0; color: #eef4f8;}
    .px-section-head p {margin: 0; color: #9ab0c4;}
    .px-metrics-grid {grid-template-columns: repeat(4, minmax(0, 1fr));}
    .px-story-grid, .px-task-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}
    .px-two-up {grid-template-columns: repeat(2, minmax(0, 1fr));}
    .px-metric-value {font-size: 2rem; font-weight: 800; margin-top: 10px;}
    .px-step {
      width: 34px; height: 34px; border-radius: 999px; display: grid; place-items: center;
      background: #163041; color: #8ae4dc; font-weight: 800; margin-bottom: 12px;
    }
    .px-plot {width: 100%; border-radius: 16px; border: 1px solid #22384d; margin-top: 12px;}
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
    @media (max-width: 1024px) {
      .px-hero, .px-metrics-grid, .px-story-grid, .px-task-grid, .px-two-up, .px-session-grid {grid-template-columns: 1fr;}
    }
    """


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
                "The same agent gets better after feedback.",
            ),
            _gradio_metric_card(
                "Trained checkpoint",
                trained_score,
                "This shows the trained model score when it is ready.",
            ),
        ]
    )
    return f"""
    <div class="px-shell">
      <section class="px-hero">
        <div>
          <div class="px-eyebrow">Simple demo</div>
          <h1>PrivacyOps-X</h1>
          <p class="px-lead">
            PrivacyOps-X is a test environment for privacy tasks.
            It checks if an AI agent can read a case, look at records, check policy,
            ask for review, and give a safe final answer.
          </p>
          <div class="px-link-row">
            <a class="px-link-button px-primary" href="/playground/">Open playground</a>
            <a class="px-link-button px-primary" href="/dashboard">View results</a>
            <a class="px-link-button" href="/docs">API docs</a>
            <a class="px-link-button" href="/schema">Schema</a>
            <a class="px-link-button" href="/judge-report">Project summary</a>
          </div>
        </div>
        <div class="px-card px-summary-card">
          <h3>What this project does</h3>
          <ul>
            <li>This is a privacy workflow demo, not just a chatbot.</li>
            <li>The agent uses clear actions like <code>inspect_case</code>, <code>open_record</code>, and <code>request_review</code>.</li>
            <li>Each step shows what changed, what risk was found, and what score the agent got.</li>
            <li>The same system can test agents, improve them, and train models.</li>
          </ul>
        </div>
      </section>

      <section class="px-grid px-metrics-grid">
        {metrics}
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
        <div class="px-grid px-two-up">
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
        </div>
      </section>
    </div>
    """


def _build_results_html(payload: dict[str, Any]) -> str:
    trained_score = _format_score(payload["sft_score"])
    return f"""
    <div class="px-shell">
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
        with gr.Tabs():
            with gr.Tab("Run Demo"):
                gr.Markdown(
                    """
                    ### Live benchmark playground
                    Use this tab to run the live demo.
                    Pick a task, start the case, send one action, and read the JSON result.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        status = gr.HTML(_status_html("Ready to start a case."))
                        task_id = gr.Dropdown(
                            choices=task_choices,
                            value=task_choices[0][1],
                            label="Task",
                        )
                        seed = gr.Number(value=0, precision=0, label="Seed")
                        reset_button = gr.Button("Start Case", variant="primary")
                        action_json = gr.Code(
                            value=_reset_action_template(),
                            language="json",
                            label="Action JSON",
                            lines=16,
                        )
                        with gr.Row():
                            inspect_button = gr.Button("Inspect case")
                            search_button = gr.Button("Search policy")
                            review_button = gr.Button("Ask legal to review")
                        with gr.Row():
                            requester_button = gr.Button("Message user")
                            self_review_button = gr.Button("Check my work")
                            submit_button = gr.Button("Submit")
                        step_button = gr.Button("Send Action", variant="primary")
                    with gr.Column(scale=6):
                        session_summary = gr.HTML(_session_summary_html(None, None))
                        response_json = gr.Code(
                            value="Press \"Start Case\" to begin.",
                            language="json",
                            label="Latest result",
                            lines=28,
                        )
                gr.Examples(
                    examples=action_examples,
                    inputs=[action_json],
                    label="Example actions",
                )
                session_env = gr.State(value=None)
                reset_button.click(
                    fn=_gradio_reset_episode,
                    inputs=[task_id, seed, session_env],
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
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>PrivacyOps-X Judge Dashboard</title>
        <style>
          :root {{
            --bg: #07131f;
            --panel: rgba(8, 20, 32, 0.9);
            --panel-2: rgba(14, 30, 45, 0.95);
            --ink: #eef4f8;
            --muted: #9db2c4;
            --teal: #72e6d1;
            --amber: #ffbf70;
            --rose: #ff8a80;
            --blue: #7db8ff;
            --line: rgba(132, 171, 196, 0.22);
          }}
          body {{
            margin: 0;
            color: var(--ink);
            font-family: "Trebuchet MS", "Lucida Sans Unicode", sans-serif;
            background:
              radial-gradient(circle at top left, rgba(114, 230, 209, 0.14), transparent 30%),
              radial-gradient(circle at 88% 10%, rgba(125, 184, 255, 0.14), transparent 24%),
              linear-gradient(160deg, #051019 0%, #091827 45%, #11263a 100%);
          }}
          * {{ box-sizing: border-box; }}
          .shell {{
            width: min(1220px, calc(100vw - 28px));
            margin: 0 auto;
            padding: 28px 0 42px;
          }}
          .hero, .card {{
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 24px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
          }}
          .hero {{
            padding: 28px;
          }}
          .eyebrow {{
            display: inline-flex;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(114, 230, 209, 0.22);
            background: rgba(114, 230, 209, 0.08);
            color: var(--teal);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.8rem;
          }}
          h1 {{
            margin: 14px 0 10px;
            font-size: clamp(2rem, 4vw, 3rem);
            font-family: Georgia, "Times New Roman", serif;
          }}
          p {{
            color: var(--muted);
            line-height: 1.7;
          }}
          .actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-top: 18px;
          }}
          .button {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 16px;
            border-radius: 14px;
            text-decoration: none;
            color: var(--ink);
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.12);
            font-weight: 700;
          }}
          .button.primary {{
            background: linear-gradient(135deg, var(--teal), #95f4e4);
            color: #061018;
            border-color: transparent;
          }}
          .grid {{
            display: grid;
            gap: 18px;
            margin-top: 22px;
          }}
          .stats {{
            grid-template-columns: repeat(5, minmax(0, 1fr));
          }}
          .card {{
            padding: 20px;
          }}
          .label {{
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.76rem;
          }}
          .value {{
            margin-top: 8px;
            font-size: 2rem;
            font-weight: 800;
          }}
          .value small {{
            display: block;
            font-size: 0.92rem;
            color: var(--muted);
            margin-top: 6px;
          }}
          .section {{
            margin-top: 30px;
          }}
          .section h2 {{
            margin: 0 0 14px;
            color: var(--amber);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.9rem;
          }}
          .split {{
            display: grid;
            gap: 18px;
            grid-template-columns: 1fr 1fr;
          }}
          .stack {{
            display: grid;
            gap: 18px;
            grid-template-columns: 1fr 1fr;
          }}
          .plot {{
            overflow: hidden;
          }}
          .plot img {{
            display: block;
            width: 100%;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: #051019;
          }}
          .placeholder {{
            min-height: 240px;
            display: grid;
            place-items: center;
            border-radius: 18px;
            border: 1px dashed rgba(255,255,255,0.16);
            color: var(--muted);
            background: var(--panel-2);
          }}
          .trajectory {{
            margin: 0;
            padding-left: 22px;
          }}
          .trajectory li {{
            margin-bottom: 10px;
            color: var(--ink);
          }}
          code {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 8px;
            background: rgba(125, 184, 255, 0.08);
            color: #d8ebff;
            white-space: pre-wrap;
            word-break: break-word;
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
          }}
          th, td {{
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
          }}
          th {{
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.74rem;
          }}
          .muted {{ color: var(--muted); }}
          @media (max-width: 980px) {{
            .stats, .split, .stack {{
              grid-template-columns: 1fr;
            }}
          }}
        </style>
      </head>
      <body>
        <div class="shell">
          <section class="hero">
            <div class="eyebrow">View results</div>
            <h1>PrivacyOps-X results</h1>
            <p>
              This page shows the main results of the project.
              It shows scores, plots, and what the numbers mean.
            </p>
            <div class="actions">
              <a class="button primary" href="/docs">Open docs</a>
              <a class="button" href="/judge-report">Project summary JSON</a>
              <a class="button" href="/curriculum">Curriculum JSON</a>
              <a class="button" href="/schema">Schema</a>
            </div>
          </section>

          <section class="grid stats">
            <article class="card">
              <div class="label">Random agent</div>
              <div class="value">{random_score}<small>low score</small></div>
            </article>
            <article class="card">
              <div class="label">Teacher agent</div>
              <div class="value">{teacher_score}<small>high score</small></div>
            </article>
            <article class="card">
              <div class="label">Baseline</div>
              <div class="value">{baseline_score}<small>score before improvement</small></div>
            </article>
            <article class="card">
              <div class="label">Improved</div>
              <div class="value">{improved_score}<small>score after improvement</small></div>
            </article>
            <article class="card">
              <div class="label">Trained Model</div>
              <div class="value">{sft_score}<small>{"not ready yet" if payload["sft_score"] is None else "trained model result"}</small></div>
            </article>
          </section>

          <section class="section">
            <h2>Plots</h2>
            <div class="split">
              <article class="card plot">
                <h3>Policy comparison</h3>
                <p class="muted">This plot compares a weak agent and a strong agent.</p>
                {random_plot_html}
              </article>
              <article class="card plot">
                <h3>Improvement over time</h3>
                <p class="muted">This plot shows the agent getting better after feedback.</p>
                {self_plot_html}
              </article>
            </div>
          </section>

          <section class="section">
            <h2>What The Score Means</h2>
            <article class="card">
              <table>
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


