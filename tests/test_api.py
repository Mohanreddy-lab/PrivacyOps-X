from fastapi.testclient import TestClient

import server.app as app_module
from server.app import app


client = TestClient(app)


def test_root_endpoint_renders_homepage() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "PrivacyOps-X" in response.text
    assert "/docs" in response.text
    assert "Open Playground" in response.text
    assert "Verified access with prompt injection" in response.text


def test_reset_endpoint_responds() -> None:
    response = client.post("/reset", json={"task_id": "easy_verified_access_with_injection"})
    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["task_id"] == "easy_verified_access_with_injection"


def test_state_and_schema_endpoints_respond() -> None:
    state_response = client.get("/state")
    schema_response = client.get("/schema")
    assert state_response.status_code == 200
    assert schema_response.status_code == 200
    schema = schema_response.json()
    assert "action" in schema
    assert "observation" in schema
    assert "state" in schema
    assert "task_id" in schema["state"]["properties"]
    assert "requester_thread" in schema["observation"]["properties"]
    assert "stakeholder_inbox" in schema["observation"]["properties"]
    assert "milestones" in schema["observation"]["properties"]
    assert "sla_deadline" in schema["observation"]["properties"]


def test_step_endpoint_returns_typed_payload() -> None:
    response = client.post("/step", json={"action": {"action_type": "submit"}})
    assert response.status_code == 200
    body = response.json()
    assert "observation" in body
    assert "reward" in body
    assert "done" in body


def test_judge_report_and_curriculum_endpoints_respond() -> None:
    judge_response = client.get("/judge-report")
    curriculum_response = client.get("/curriculum")
    assert judge_response.status_code == 200
    assert curriculum_response.status_code == 200
    judge_body = judge_response.json()
    curriculum_body = curriculum_response.json()
    assert judge_body["env_name"] == "PrivacyOps-X"
    assert "self-improving agent systems" in judge_body["themes"]
    assert len(judge_body["task_cards"]) == 4
    assert "TRAINING.md" in judge_body["training_assets"]
    assert curriculum_body["env_name"] == "PrivacyOps-X"
    assert len(curriculum_body["tracks"]) == 4


def test_malformed_json_payload_is_rejected() -> None:
    response = client.post(
        "/reset",
        data='{"task_id":',
        headers={"content-type": "application/json", "x-forwarded-for": "203.0.113.10"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Malformed JSON payload."


def test_oversized_payload_is_rejected() -> None:
    response = client.post(
        "/step",
        json={"action": {"action_type": "message_requester", "content": "x" * 40_000}},
        headers={"x-forwarded-for": "203.0.113.11"},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "Payload too large."


def test_rate_limit_applies_to_repeated_requests(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "READ_RATE_LIMIT_MAX_REQUESTS", 2)
    app_module.RATE_LIMIT_BUCKETS.clear()
    headers = {"x-forwarded-for": "198.51.100.42"}
    assert client.get("/healthz", headers=headers).status_code == 200
    assert client.get("/healthz", headers=headers).status_code == 200
    limited = client.get("/healthz", headers=headers)
    assert limited.status_code == 429
    assert limited.headers["Retry-After"]
