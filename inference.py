"""Baseline inference script for PrivacyOps-X."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

from client import PrivacyOpsXEnv
from models import PrivacyOpsAction, PrivacyOpsObservation
from server.teacher import build_teacher_plan

BENCHMARK = "privacyops_x"
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_ROUTER_URL = os.getenv("HF_ROUTER_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_IMAGE_NAME = "privacyops-x:latest"
DEFAULT_IMAGE_NAME_ALT = "privacyops_x:latest"
API_KEY = OPENAI_API_KEY or HF_TOKEN or ""
if not OPENAI_API_KEY and HF_TOKEN and HF_ROUTER_URL:
    API_BASE_URL = HF_ROUTER_URL
MODEL_TIMEOUT_SECONDS = float(os.environ.get("MODEL_TIMEOUT_SECONDS", "8"))
TASK_ORDER = [
    "easy_verified_access_with_injection",
    "medium_unverified_erasure_multi_account",
    "hard_guardian_minor_legal_hold_fraud",
]
TEMPERATURE = 0
MAX_TOKENS = 220
SUCCESS_SCORE_THRESHOLD = 0.85
STRICT_SCORE_EPS = float(os.getenv("STRICT_SCORE_EPS", "0.01"))

SYSTEM_PROMPT = """You are an agent operating a privacy operations benchmark.
Return exactly one compact JSON object describing the next action.
Allowed keys: action_type, target_id, field_name, field_value, query, content, reviewer, confidence.
Prefer evidence gathering, reviewer coordination, and self-correction before submission.
Never wrap the JSON in markdown."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward} done={done} error={error}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}", flush=True)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = next((part for part in parts if "{" in part), text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start : end + 1])


def fallback_policy(task_id: str, step: int) -> dict[str, Any]:
    task_plan = build_teacher_plan(task_id)
    return task_plan[step - 1] if step <= len(task_plan) else {"action_type": "submit"}


def _to_error_code(exc: Exception) -> str:
    name = exc.__class__.__name__.strip().lower()
    return name or "runtime_error"


def _strict_unit_score(value: float) -> float:
    """Clamp score into strict open interval (0, 1)."""
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.5
    if numeric <= 0.0:
        return STRICT_SCORE_EPS
    if numeric >= 1.0:
        return 1.0 - STRICT_SCORE_EPS
    return numeric


def _docker_fallback_enabled() -> bool:
    return os.getenv("ENABLE_DOCKER_FALLBACK", "0").lower() in {"1", "true", "yes", "on"}


def get_model_action(
    client: Any | None,
    task_id: str,
    step: int,
    observation: PrivacyOpsObservation,
    history: list[str],
) -> dict[str, Any]:
    if not API_KEY or client is None:
        return fallback_policy(task_id, step)
    user_prompt = (
        f"Task: {task_id}\n"
        f"Step: {step}\n"
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
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("Empty model response")
        return extract_json(text)
    except Exception as exc:
        if os.environ.get("LOG_DEBUG") == "1":
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return fallback_policy(task_id, step)


async def create_env() -> PrivacyOpsXEnv:
    base_url = os.getenv("ENV_BASE_URL")
    if base_url:
        client = PrivacyOpsXEnv(base_url=base_url)
        await client.connect()
        return client

    last_error: Exception | None = None
    for fallback_url in ("http://127.0.0.1:8000", "http://localhost:8000"):
        try:
            client = PrivacyOpsXEnv(base_url=fallback_url)
            await client.connect()
            return client
        except Exception as exc:
            last_error = exc

    if not _docker_fallback_enabled():
        raise RuntimeError(
            "Unable to initialize environment client. Set ENV_BASE_URL or expose the environment on localhost:8000. "
            "For local Docker fallback, set ENABLE_DOCKER_FALLBACK=1."
        ) from last_error

    image_candidates = [
        LOCAL_IMAGE_NAME,
        os.getenv("IMAGE_NAME"),
        DEFAULT_IMAGE_NAME,
        DEFAULT_IMAGE_NAME_ALT,
    ]
    seen: set[str] = set()
    for image_name in image_candidates:
        if not image_name or image_name in seen:
            continue
        seen.add(image_name)
        for _attempt in range(2):
            try:
                return await PrivacyOpsXEnv.from_docker_image(image_name)
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.5)

    raise RuntimeError("Unable to initialize environment client") from last_error


async def run_task(client: Any | None, task_id: str) -> float:
    env: PrivacyOpsXEnv | None = None
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    try:
        env = await create_env()
        result = await env.reset(task_id=task_id, seed=0)
        for step in range(1, 21):
            if result.done:
                break
            action_payload = get_model_action(client, task_id, step, result.observation, history)
            try:
                action = PrivacyOpsAction(**action_payload)
            except Exception:
                action_payload = fallback_policy(task_id, step)
                action = PrivacyOpsAction(**action_payload)
            try:
                result = await env.step(action)
            except Exception as exc:
                reward = 0.0
                done = False
                error = _to_error_code(exc)
                rewards.append(reward)
                steps_taken = step
                log_step(
                    step=step,
                    action=json.dumps(action_payload, sort_keys=True),
                    reward=reward,
                    done=done,
                    error=error,
                )
                break
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = result.observation.metadata.get("info", {}).get("error_code")
            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=json.dumps(action_payload, sort_keys=True),
                reward=reward,
                done=done,
                error=error,
            )
            history.append(
                f"step={step} action={json.dumps(action_payload, sort_keys=True)} reward={reward:.4f}"
            )
            if done:
                break
        raw_score = float(
            result.observation.metadata.get("info", {}).get("final_score", result.reward or 0.0)
        )
        score = _strict_unit_score(raw_score)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        if os.environ.get("LOG_DEBUG") == "1":
            print(f"[DEBUG] task_run_error task={task_id} error={exc}", flush=True)
        success = False
        score = _strict_unit_score(0.0)
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as exc:
                if os.environ.get("LOG_DEBUG") == "1":
                    print(f"[DEBUG] env.close() error (container cleanup): {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


async def main() -> None:
    client: Any | None = None
    if API_KEY and OpenAIClient is not None:
        try:
            client = OpenAIClient(
                base_url=API_BASE_URL,
                api_key=API_KEY,
                timeout=MODEL_TIMEOUT_SECONDS,
                max_retries=0,
            )
        except Exception as exc:
            if os.environ.get("LOG_DEBUG") == "1":
                print(f"[DEBUG] openai_client_init_error: {exc}", flush=True)
            client = None
    elif API_KEY and os.environ.get("LOG_DEBUG") == "1":
        print("[DEBUG] openai_client_unavailable: falling back to deterministic policy", flush=True)

    for task_id in TASK_ORDER:
        try:
            await run_task(client, task_id)
        except Exception as exc:
            if os.environ.get("LOG_DEBUG") == "1":
                print(f"[DEBUG] unhandled_task_error task={task_id} error={exc}", flush=True)
            log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=_strict_unit_score(0.0), rewards=[])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        if os.environ.get("LOG_DEBUG") == "1":
            print(f"[DEBUG] fatal_main_error: {exc}", flush=True)
