import asyncio
import io
from contextlib import redirect_stdout

import inference


def test_log_format_is_exact() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        inference.log_start(task="task", env="env", model="model")
        inference.log_step(step=1, action='{"action_type":"submit"}', reward=1.0, done=True, error=None)
        inference.log_end(success=True, steps=1, score=1.0, rewards=[1.0])
    lines = buf.getvalue().strip().splitlines()
    assert lines[0] == "[START] task=task env=env model=model"
    assert lines[1] == '[STEP] step=1 action={"action_type":"submit"} reward=1.0 done=True error=None'
    assert lines[2] == "[END] success=True steps=1 score=1.0 rewards=[1.0]"


def test_task_order_and_fallback_contract() -> None:
    assert inference.TASK_ORDER == [
        "easy_verified_access_with_injection",
        "medium_unverified_erasure_multi_account",
        "hard_guardian_minor_legal_hold_fraud",
    ]
    assert inference.fallback_policy("easy_verified_access_with_injection", 1)["action_type"] == "inspect_case"
    assert inference.fallback_policy("hard_guardian_minor_legal_hold_fraud", 20)["action_type"] == "submit"


def test_strict_unit_score_stays_inside_open_interval() -> None:
    assert inference._strict_unit_score(0.0) == 0.01
    assert inference._strict_unit_score(1.0) == 0.99
    assert inference._strict_unit_score(0.42) == 0.42


def test_create_env_skips_docker_fallback_by_default(monkeypatch) -> None:
    calls: list[str] = []

    class FakeEnv:
        def __init__(self, base_url: str):
            self.base_url = base_url

        async def connect(self) -> None:
            raise RuntimeError(f"unreachable {self.base_url}")

        @classmethod
        async def from_docker_image(cls, image_name: str):
            calls.append(image_name)
            raise AssertionError("docker fallback should be disabled")

    monkeypatch.delenv("ENV_BASE_URL", raising=False)
    monkeypatch.delenv("ENABLE_DOCKER_FALLBACK", raising=False)
    monkeypatch.setattr(inference, "PrivacyOpsXEnv", FakeEnv)

    try:
        asyncio.run(inference.create_env())
    except RuntimeError as exc:
        assert "ENABLE_DOCKER_FALLBACK=1" in str(exc)
    else:
        raise AssertionError("create_env() should fail cleanly when no environment is reachable")

    assert calls == []
