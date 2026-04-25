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

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
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
        else None
    )
    teacher_score = (
        teacher_report.get("overall", {}).get("mean_final_score")
        if teacher_report
        else None
    )
    sft_score = sft_report.get("overall", {}).get("mean_final_score") if sft_report else None
    baseline_score = self_report.get("baseline_score") if self_report else None
    improved_score = self_report.get("improved_score") if self_report else None

    return {
        "random_score": random_score,
        "teacher_score": teacher_score,
        "sft_score": sft_score,
        "baseline_score": baseline_score,
        "improved_score": improved_score,
        "before_behavior": self_report.get("before_behavior") if self_report else None,
        "after_behavior": self_report.get("after_behavior") if self_report else None,
        "random_plot": random_plot,
        "self_plot": self_plot,
    }


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
    pretty = request.query_params.get("pretty")
    if not pretty or pretty.lower() in {"0", "false", "no"}:
        return response
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
    before_html = _render_trajectory(payload["before_behavior"])
    after_html = _render_trajectory(payload["after_behavior"])
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
            <div class="eyebrow">Judge dashboard</div>
            <h1>PrivacyOps-X report view</h1>
            <p>
              This page is the fastest path for judges to understand the benchmark:
              baseline range, oracle upper bound, self-improvement evidence, plots,
              and one before/after trajectory on the finale case.
            </p>
            <div class="actions">
              <a class="button primary" href="/docs">Open API docs</a>
              <a class="button" href="/judge-report">Judge report JSON</a>
              <a class="button" href="/curriculum">Curriculum JSON</a>
              <a class="button" href="/schema">Typed schema</a>
            </div>
          </section>

          <section class="grid stats">
            <article class="card">
              <div class="label">Random finale</div>
              <div class="value">{random_score}<small>lower-bound baseline</small></div>
            </article>
            <article class="card">
              <div class="label">Teacher finale</div>
              <div class="value">{teacher_score}<small>oracle upper bound</small></div>
            </article>
            <article class="card">
              <div class="label">Self-improve before</div>
              <div class="value">{baseline_score}<small>adaptive policy start</small></div>
            </article>
            <article class="card">
              <div class="label">Self-improve after</div>
              <div class="value">{improved_score}<small>adaptive policy end</small></div>
            </article>
            <article class="card">
              <div class="label">GPU SFT checkpoint</div>
              <div class="value">{sft_score}<small>{"pending run" if payload["sft_score"] is None else "trained checkpoint"}</small></div>
            </article>
          </section>

          <section class="section">
            <h2>Plots</h2>
            <div class="split">
              <article class="card plot">
                <h3>Finale baseline range</h3>
                <p class="muted">Random policy versus teacher policy on the showcase task.</p>
                {random_plot_html}
              </article>
              <article class="card plot">
                <h3>Self-improvement curve</h3>
                <p class="muted">Failure-aware adaptive policy improving over repeated episodes.</p>
                {self_plot_html}
              </article>
            </div>
          </section>

          <section class="section">
            <h2>Before vs after</h2>
            <div class="split">
              <article class="card">
                <h3>Before improvement</h3>
                <p class="muted">Short, shallow trajectory that submits with limited evidence and no reviewer coordination.</p>
                {before_html}
              </article>
              <article class="card">
                <h3>After improvement</h3>
                <p class="muted">Full trajectory with record inspection, policy grounding, requester follow-up, review, and self-check.</p>
                {after_html}
              </article>
            </div>
          </section>

          <section class="section">
            <h2>Reward breakdown</h2>
            <article class="card">
              <table>
                <thead>
                  <tr><th>Metric</th><th>Meaning</th></tr>
                </thead>
                <tbody>
                  <tr><td>Compliance</td><td>Follows privacy workflow and policy requirements.</td></tr>
                  <tr><td>Safety</td><td>Avoids harmful disclosure, unsafe routing, or false promises.</td></tr>
                  <tr><td>Evidence</td><td>Opens the right records, policy, and requester facts before acting.</td></tr>
                  <tr><td>Legal</td><td>Handles legal hold, retention, and escalation consistently.</td></tr>
                  <tr><td>Communication</td><td>Produces safe requester replies and clear internal notes.</td></tr>
                  <tr><td>Efficiency</td><td>Avoids redundant or wasteful steps while staying within budget.</td></tr>
                </tbody>
              </table>
            </article>
          </section>
        </div>
      </body>
    </html>
    """


@app.get("/playground", include_in_schema=False, response_class=HTMLResponse)
@app.get("/playground/", include_in_schema=False, response_class=HTMLResponse)
@app.get("/web", include_in_schema=False, response_class=HTMLResponse)
@app.get("/web/", include_in_schema=False, response_class=HTMLResponse)
def playground() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>PrivacyOps-X Playground</title>
        <style>
          :root {
            --page: #08131c;
            --panel: #0f1c27;
            --panel-soft: #132331;
            --field: #0a141d;
            --line: #21384a;
            --ink: #edf4f8;
            --muted: #9fb0bd;
            --accent: #56d3c5;
            --accent-ink: #072028;
            --danger-soft: rgba(255, 143, 135, 0.10);
            --ok-soft: rgba(86, 211, 197, 0.12);
            --radius: 18px;
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            background: var(--page);
            color: var(--ink);
            font: 16px/1.5 Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }
          a { color: inherit; }
          .shell {
            width: min(1380px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 28px 0 40px;
          }
          .hero {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 18px;
            align-items: end;
            margin-bottom: 18px;
          }
          .brand {
            display: flex;
            align-items: flex-start;
            gap: 14px;
          }
          .logo {
            width: 52px;
            height: 52px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: #163041;
            color: var(--accent);
            font-weight: 800;
            letter-spacing: 0.06em;
          }
          .eyebrow {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid #244354;
            background: #10202c;
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          h1 {
            margin: 10px 0 6px;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.05;
            font-family: Georgia, "Times New Roman", serif;
          }
          .subtitle {
            margin: 0;
            max-width: 760px;
            color: var(--muted);
          }
          .top-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: flex-end;
          }
          .button,
          .chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 42px;
            padding: 10px 14px;
            border-radius: 12px;
            border: 1px solid var(--line);
            background: var(--panel-soft);
            color: var(--ink);
            font-weight: 700;
            text-decoration: none;
            cursor: pointer;
          }
          .button.primary {
            background: var(--accent);
            color: var(--accent-ink);
            border-color: transparent;
          }
          .layout {
            display: grid;
            grid-template-columns: 420px minmax(0, 1fr);
            gap: 18px;
            align-items: start;
          }
          .stack {
            display: grid;
            gap: 18px;
          }
          .card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 20px;
          }
          .section-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 14px;
          }
          .section-head p,
          .muted,
          .helper {
            margin: 0;
            color: var(--muted);
          }
          .status {
            display: inline-flex;
            align-items: center;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid #285050;
            background: var(--ok-soft);
            color: var(--accent);
            font-weight: 700;
            white-space: nowrap;
          }
          .field {
            margin-bottom: 14px;
          }
          label {
            display: block;
            margin-bottom: 8px;
            color: #dce8ef;
            font-size: 0.9rem;
            font-weight: 700;
          }
          input,
          select,
          textarea {
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--line);
            background: var(--field);
            color: var(--ink);
            padding: 12px 14px;
            font: inherit;
          }
          textarea {
            min-height: 260px;
            resize: vertical;
            font-family: Consolas, "Courier New", monospace;
            line-height: 1.55;
          }
          .inline-fields {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 120px;
            gap: 12px;
          }
          .button-row,
          .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
          }
          .response {
            display: grid;
            gap: 16px;
          }
          pre {
            margin: 0;
            min-height: 520px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-word;
            border-radius: 14px;
            border: 1px solid var(--line);
            background: var(--field);
            color: #dcebf4;
            padding: 18px;
            font: 0.95rem/1.6 Consolas, "Courier New", monospace;
          }
          .mini-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
          }
          .mini {
            border-radius: 14px;
            border: 1px solid var(--line);
            background: var(--panel-soft);
            padding: 14px;
          }
          .mini strong {
            display: block;
            margin-bottom: 6px;
            color: #dce8ef;
            font-size: 0.82rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
          }
          .mini code {
            color: #d5f7f1;
          }
          .inline-link {
            color: #cfeef5;
            text-decoration: none;
            border-bottom: 1px dotted #527387;
          }
          .kicker {
            display: grid;
            gap: 12px;
            margin-bottom: 18px;
          }
          .tip-box {
            border-left: 3px solid var(--accent);
            background: #0e1b26;
            border-radius: 14px;
            padding: 14px 16px;
          }
          @media (max-width: 1080px) {
            .hero,
            .layout,
            .mini-grid {
              grid-template-columns: 1fr;
            }
            .top-actions {
              justify-content: flex-start;
            }
          }
        </style>
      </head>
      <body>
        <div class="shell">
          <section class="hero">
            <div class="brand">
              <div class="logo">PX</div>
              <div>
                <div class="eyebrow">Interactive Playground</div>
                <h1>PrivacyOps-X</h1>
                <p class="subtitle">
                  Use this page to run real benchmark episodes against the live environment.
                  Reset a case, inspect the observation, send a typed action, and watch the
                  exact response that the evaluator sees.
                </p>
              </div>
            </div>
            <div class="top-actions">
              <a class="button" href="/">Home</a>
              <a class="button" href="/dashboard">Judge Dashboard</a>
              <a class="button" href="/docs">API Docs</a>
              <a class="button" href="/schema">Schema</a>
            </div>
          </section>

          <div class="layout">
            <div class="stack">
              <section class="card">
                <div class="section-head">
                  <div>
                    <h2>Episode control</h2>
                    <p class="muted">Choose a task, set the seed, then call <code>/reset</code>.</p>
                  </div>
                  <div id="status" class="status">Ready</div>
                </div>

                <div class="field">
                  <label for="task-id">Task</label>
                  <select id="task-id">
                    <option value="easy_verified_access_with_injection">Easy - verified access with prompt injection</option>
                    <option value="medium_unverified_erasure_multi_account">Medium - multi-account erasure with billing retention</option>
                    <option value="hard_guardian_minor_legal_hold_fraud">Hard - guardian request under legal hold and fraud</option>
                    <option value="finale_cross_border_recovery_cascade">Finale - cross-border recovery cascade</option>
                  </select>
                </div>

                <div class="inline-fields">
                  <div class="field">
                    <label for="seed">Seed</label>
                    <input id="seed" type="number" value="0" min="0">
                  </div>
                  <div class="field">
                    <label for="pretty">Pretty</label>
                    <select id="pretty">
                      <option value="1">On</option>
                      <option value="0">Off</option>
                    </select>
                  </div>
                </div>

                <div class="button-row">
                  <button class="button primary" type="button" onclick="resetEnv()">Reset episode</button>
                  <button class="button" type="button" onclick="loadHealth()">Health</button>
                  <button class="button" type="button" onclick="loadSchema()">Schema</button>
                </div>
              </section>

              <section class="card">
                <div class="section-head">
                  <div>
                    <h2>Action JSON</h2>
                    <p class="muted">Paste only the action body. The page wraps it as <code>{"action": ...}</code>.</p>
                  </div>
                </div>

                <div class="field">
                  <label for="action-json">Next action</label>
                  <textarea id="action-json">{
  "action_type": "inspect_case"
}</textarea>
                </div>

                <div class="chip-row">
                  <button class="chip" type="button" onclick="setAction('inspect_case')">Inspect case</button>
                  <button class="chip" type="button" onclick="setAction('open_record', { target_id: 'core_profile' })">Open record</button>
                  <button class="chip" type="button" onclick="setAction('search_policy', { query: 'legal hold retention deletion' })">Search policy</button>
                  <button class="chip" type="button" onclick="setAction('self_review')">Self review</button>
                  <button class="chip" type="button" onclick="setAction('submit')">Submit</button>
                </div>

                <div class="button-row" style="margin-top:12px;">
                  <button class="button primary" type="button" onclick="runStep()">Send step</button>
                </div>
              </section>
            </div>

            <section class="card response">
              <div class="section-head">
                <div>
                  <h2>Live response</h2>
                  <p class="muted">Every request and response stays visible here so judges can see the exact environment behavior.</p>
                </div>
              </div>

              <div class="kicker">
                <div class="tip-box">
                  Start with <code>inspect_case</code>, then open records and policy before sending requester or reviewer actions.
                </div>
              </div>

              <pre id="output">Press "Reset episode" to begin.</pre>

              <div class="mini-grid">
                <div class="mini">
                  <strong>Suggested flow</strong>
                  Reset -> inspect -> open records -> search policy -> review -> submit
                </div>
                <div class="mini">
                  <strong>Useful links</strong>
                  <a class="inline-link" href="/envinfo">/envinfo</a><br>
                  <a class="inline-link" href="/judge-report">/judge-report</a><br>
                  <a class="inline-link" href="/curriculum">/curriculum</a>
                </div>
                <div class="mini">
                  <strong>Common actions</strong>
                  <code>inspect_case</code><br>
                  <code>open_record</code><br>
                  <code>request_review</code>
                </div>
              </div>
            </section>
          </div>
        </div>

        <script>
          const output = document.getElementById("output");
          const statusChip = document.getElementById("status");

          function setStatus(text, kind = "ok") {
            statusChip.textContent = text;
            if (kind === "error") {
              statusChip.style.color = "#ffd4cf";
              statusChip.style.borderColor = "#5d3330";
              statusChip.style.background = "var(--danger-soft)";
              return;
            }
            statusChip.style.color = "var(--accent)";
            statusChip.style.borderColor = "#285050";
            statusChip.style.background = "var(--ok-soft)";
          }

          function prettyEnabled() {
            return document.getElementById("pretty").value === "1";
          }

          function render(title, payload) {
            output.textContent = title + "\\n\\n" + JSON.stringify(payload, null, 2);
          }

          async function request(path, options = {}) {
            const url = prettyEnabled() ? path + (path.includes("?") ? "&pretty=1" : "?pretty=1") : path;
            const response = await fetch(url, options);
            const text = await response.text();
            let body;
            try {
              body = JSON.parse(text);
            } catch {
              body = text;
            }
            if (!response.ok) {
              throw new Error(typeof body === "string" ? body : JSON.stringify(body, null, 2));
            }
            return body;
          }

          async function resetEnv() {
            try {
              setStatus("Resetting...");
              const taskId = document.getElementById("task-id").value;
              const seed = Number(document.getElementById("seed").value || 0);
              const body = await request("/reset", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ task_id: taskId, seed })
              });
              render("POST /reset", body);
              setStatus("Episode ready");
            } catch (error) {
              render("Reset failed", { error: String(error) });
              setStatus("Reset failed", "error");
            }
          }

          async function runStep() {
            try {
              setStatus("Running step...");
              const action = JSON.parse(document.getElementById("action-json").value);
              const body = await request("/step", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action })
              });
              render("POST /step", body);
              setStatus(body.done ? "Episode finished" : "Step accepted");
            } catch (error) {
              render("Step failed", { error: String(error) });
              setStatus("Step failed", "error");
            }
          }

          async function loadSchema() {
            try {
              setStatus("Loading schema...");
              const body = await request("/schema");
              render("GET /schema", body);
              setStatus("Schema loaded");
            } catch (error) {
              render("Schema failed", { error: String(error) });
              setStatus("Schema failed", "error");
            }
          }

          async function loadHealth() {
            try {
              setStatus("Checking health...");
              const body = await request("/healthz");
              render("GET /healthz", body);
              setStatus("Healthy");
            } catch (error) {
              render("Health check failed", { error: String(error) });
              setStatus("Health failed", "error");
            }
          }

          function setAction(actionType, extras = {}) {
            document.getElementById("action-json").value = JSON.stringify({ action_type: actionType, ...extras }, null, 2);
          }
        </script>
      </body>
    </html>
    """


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>PrivacyOps-X</title>
        <style>
          :root {
            --bg: #08131d;
            --panel: #0f1d2a;
            --panel-soft: #132536;
            --panel-2: #0c1723;
            --ink: #eef4f8;
            --muted: #9ab0c4;
            --line: #22384d;
            --accent: #4fd1c5;
            --accent-2: #8ae4dc;
            --warning: #ffbf70;
            --shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            background: linear-gradient(180deg, #09131d 0%, #0a1622 100%);
            color: var(--ink);
            font: 16px/1.55 Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }
          .page {
            width: min(1360px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 28px 0 44px;
          }
          .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin-bottom: 18px;
          }
          .brand {
            display: flex;
            align-items: center;
            gap: 14px;
          }
          .logo {
            width: 48px;
            height: 48px;
            border-radius: 16px;
            display: grid;
            place-items: center;
            background: #173044;
            color: var(--accent-2);
            font-weight: 800;
            letter-spacing: 0.06em;
          }
          h1, h2, h3 { margin: 0; }
          h1 {
            font-size: clamp(2.2rem, 4vw, 3.4rem);
            font-family: Georgia, "Times New Roman", serif;
          }
          .eyebrow {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid #244357;
            color: var(--accent-2);
            background: rgba(79, 209, 197, 0.08);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.75rem;
            font-weight: 700;
          }
          .hero {
            display: grid;
            grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.95fr);
            gap: 20px;
            padding: 28px;
            border-radius: 24px;
            border: 1px solid var(--line);
            background: var(--panel);
            box-shadow: var(--shadow);
          }
          .hero-copy p,
          .muted {
            color: var(--muted);
            margin: 0;
          }
          .hero-copy p {
            margin-top: 14px;
            max-width: 62ch;
            font-size: 1.05rem;
          }
          .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 22px;
          }
          .button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 46px;
            padding: 10px 16px;
            border-radius: 12px;
            text-decoration: none;
            font-weight: 700;
            border: 1px solid #2a465d;
            background: #10202d;
            color: var(--ink);
          }
          .button.primary {
            background: var(--accent);
            color: #08212a;
            border-color: transparent;
          }
          .aside {
            display: grid;
            gap: 14px;
          }
          .mini-panel,
          .card {
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--panel-soft);
            padding: 18px;
          }
          .mini-panel h2,
          .section-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--warning);
            margin-bottom: 10px;
          }
          .mini-panel ul {
            margin: 0;
            padding-left: 18px;
            color: var(--muted);
          }
          .mini-panel li { margin-bottom: 8px; }
          .stats,
          .link-grid,
          .case-grid,
          .verify-grid {
            display: grid;
            gap: 16px;
            margin-top: 18px;
          }
          .stats { grid-template-columns: repeat(4, minmax(0, 1fr)); }
          .link-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
          .case-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
          .verify-grid { grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr); }
          .stat strong,
          .link-card h3,
          .case-card h3 {
            display: block;
            margin-bottom: 8px;
          }
          .stat strong {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
          }
          .stat .value {
            font-size: 2rem;
            font-weight: 800;
          }
          .stat .sub {
            color: var(--muted);
            margin-top: 6px;
          }
          .section-title {
            margin-top: 28px;
          }
          .link-card a,
          .case-card a,
          a.inline-link {
            color: var(--accent-2);
            text-decoration: none;
          }
          .link-card p,
          .case-card p {
            margin: 0;
            color: var(--muted);
          }
          .tag {
            display: inline-flex;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(138, 228, 220, 0.08);
            border: 1px solid #244357;
            color: var(--accent-2);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
          }
          pre {
            margin: 0;
            padding: 18px;
            border-radius: 16px;
            border: 1px solid #20384c;
            background: #09131d;
            color: #dceaf5;
            overflow: auto;
            font: 0.95rem/1.6 Consolas, "Courier New", monospace;
          }
          ul.clean {
            margin: 0;
            padding-left: 18px;
            color: var(--muted);
          }
          ul.clean li { margin-bottom: 8px; }
          code {
            padding: 2px 6px;
            border-radius: 7px;
            background: rgba(138, 228, 220, 0.08);
            color: #d7f7f1;
          }
          @media (max-width: 1080px) {
            .hero,
            .stats,
            .link-grid,
            .case-grid,
            .verify-grid {
              grid-template-columns: 1fr;
            }
            .page {
              width: min(100vw - 20px, 1360px);
            }
          }
        </style>
      </head>
      <body>
        <div class="page">
          <div class="topbar">
            <div class="brand">
              <div class="logo">PX</div>
              <div>
                <div class="eyebrow">OpenEnv benchmark</div>
                <h1>PrivacyOps-X</h1>
              </div>
            </div>
          </div>

          <section class="hero">
            <div class="hero-copy">
              <div class="eyebrow">Safety-critical privacy operations</div>
              <p>
                PrivacyOps-X trains and evaluates agents on real privacy-rights workflows:
                identity checks, deletion constraints, legal hold, fraud review, requester
                messaging, and final case submission under measurable reward.
              </p>
              <div class="actions">
                <a class="button primary" href="/playground">Open Playground</a>
                <a class="button" href="/dashboard">Judge Dashboard</a>
                <a class="button" href="/docs">API Docs</a>
                <a class="button" href="/judge-report">Judge Report</a>
                <a class="button" href="/schema">Typed Schema</a>
                <a class="button" href="/curriculum">Curriculum</a>
              </div>
            </div>
            <div class="aside">
              <div class="mini-panel">
                <h2>What judges can verify</h2>
                <ul>
                  <li>Typed <code>reset</code>, <code>step</code>, and <code>state</code> endpoints</li>
                  <li>Deterministic compliance, legal, and audit reviewers</li>
                  <li>Multi-turn requester interaction with revealed facts</li>
                  <li>Dense rewards with benchmark-grade final scoring</li>
                </ul>
              </div>
              <div class="mini-panel">
                <h2>Best demo flow</h2>
                <ul>
                  <li>Start in <a class="inline-link" href="/playground">/playground</a></li>
                  <li>Reset the medium or hard task</li>
                  <li>Run <code>inspect_case</code> first</li>
                  <li>Show dashboard and API docs after one step</li>
                </ul>
              </div>
            </div>
          </section>

          <section class="stats">
            <article class="card stat">
              <strong>Scenarios</strong>
              <div class="value">4</div>
              <div class="sub">easy, medium, hard, finale</div>
            </article>
            <article class="card stat">
              <strong>Reviewers</strong>
              <div class="value">3</div>
              <div class="sub">compliance, legal, audit</div>
            </article>
            <article class="card stat">
              <strong>Self-improvement</strong>
              <div class="value">0.95</div>
              <div class="sub">from 0.61 on the finale task</div>
            </article>
            <article class="card stat">
              <strong>Deployment</strong>
              <div class="value">HF Space</div>
              <div class="sub">Dockerized + OpenEnv-compatible</div>
            </article>
          </section>

          <div class="section-title">Quick links</div>
          <section class="link-grid">
            <article class="card link-card">
              <h3><a href="/playground">Interactive playground</a></h3>
              <p>Reset episodes, send actions, and inspect raw JSON responses live.</p>
            </article>
            <article class="card link-card">
              <h3><a href="/dashboard">Judge dashboard</a></h3>
              <p>See baseline, oracle, self-improvement, and judge-facing benchmark context.</p>
            </article>
            <article class="card link-card">
              <h3><a href="/docs">Swagger + schema</a></h3>
              <p>Inspect the typed OpenAPI surface and model contracts without guessing.</p>
            </article>
          </section>

          <div class="section-title">Benchmark cases</div>
          <section class="case-grid">
            <article class="card case-card">
              <span class="tag">Easy</span>
              <h3>Verified access with prompt injection</h3>
              <p>A matched requester asks for access while trying to pressure the analyst into unsafe handling.</p>
            </article>
            <article class="card case-card">
              <span class="tag">Medium</span>
              <h3>Multi-account erasure with billing retention</h3>
              <p>A GDPR deletion request arrives from a mismatched sender and conflicts with invoice retention rules.</p>
            </article>
            <article class="card case-card">
              <span class="tag">Hard + Finale</span>
              <h3>Guardian, legal hold, fraud, and cross-border conflict</h3>
              <p>High-risk workflows force the agent to balance authority, fraud review, retention, and partial fulfillment.</p>
            </article>
          </section>

          <div class="section-title">Quick verification</div>
          <section class="verify-grid">
            <article class="card">
              <pre>curl -X POST /reset

curl -X POST /reset \\
  -H "Content-Type: application/json" \\
  -d '{"task_id":"medium_unverified_erasure_multi_account","seed":0}'

curl -X POST /step \\
  -H "Content-Type: application/json" \\
  -d '{"action":{"action_type":"inspect_case"}}'</pre>
            </article>
            <article class="card">
              <ul class="clean">
                <li><a class="inline-link" href="/health">/health</a> confirms runtime readiness</li>
                <li><a class="inline-link" href="/metadata">/metadata</a> exposes environment identity</li>
                <li><a class="inline-link" href="/envinfo">/envinfo</a> provides judge-friendly metadata</li>
                <li><a class="inline-link" href="/healthz">/healthz</a> returns detailed health info</li>
                <li><a class="inline-link" href="/playground">/playground</a> opens the built-in interactive UI</li>
                <li><a class="inline-link" href="/docs">/docs</a> provides the FastAPI reference surface</li>
              </ul>
            </article>
          </section>
        </div>
      </body>
    </html>
    """


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
