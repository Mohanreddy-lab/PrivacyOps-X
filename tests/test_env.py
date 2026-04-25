from server.env import PrivacyOpsXEnvironment
from models import PrivacyOpsAction


def test_reset_returns_clean_state() -> None:
    env = PrivacyOpsXEnvironment()
    obs = env.reset(task_id="easy_verified_access_with_injection", seed=0)
    assert obs.task_id == "easy_verified_access_with_injection"
    assert obs.steps_remaining == 15
    assert obs.sla_deadline > 0
    assert obs.workspace.request_type == "unknown"
    assert len(obs.milestones) == 6
    assert obs.theme_alignment.world_modeling >= 0.0
    assert obs.stakeholder_inbox == []
    assert env.state.step_count == 0
    assert env.state.note_history == []
    assert env.state.requester_thread == []


def test_invalid_runtime_action_is_nonfatal() -> None:
    env = PrivacyOpsXEnvironment()
    env.reset(task_id="easy_verified_access_with_injection", seed=0)
    obs = env.step(PrivacyOpsAction(action_type="open_record"))
    assert obs.error == "missing_required_parameter"
    assert obs.done is False
    assert env.state.invalid_action_count == 1


def test_search_policy_reveals_articles() -> None:
    env = PrivacyOpsXEnvironment()
    env.reset(task_id="medium_unverified_erasure_multi_account", seed=0)
    obs = env.step(
        PrivacyOpsAction(
            action_type="search_policy",
            query="billing retention gdpr verification",
        )
    )
    ids = [article.article_id for article in obs.visible_policy_articles]
    assert "policy_billing_retention" in ids
    assert "policy_erasure_identity" in ids


def test_message_requester_reveals_deterministic_facts() -> None:
    env = PrivacyOpsXEnvironment()
    env.reset(task_id="medium_unverified_erasure_multi_account", seed=0)
    env.step(PrivacyOpsAction(action_type="inspect_case"))
    obs = env.step(
        PrivacyOpsAction(
            action_type="message_requester",
            content=(
                "Please verify your identity and confirm which account emails are in scope. "
                "We may need to retain billing or invoice records while deleting eligible data."
            ),
        )
    )
    assert "requester_provided_account_aliases" in obs.revealed_requester_facts
    assert "requester_acknowledges_billing_retention" in obs.revealed_requester_facts
    assert obs.latest_requester_message is not None
    assert any(turn.role == "analyst" for turn in obs.requester_thread)
    assert any(turn.role == "requester" for turn in obs.requester_thread)
    assert any(message.sender == "requester" for message in obs.stakeholder_inbox)


def test_submit_generates_improvement_lessons() -> None:
    env = PrivacyOpsXEnvironment()
    env.reset(task_id="medium_unverified_erasure_multi_account", seed=0)
    obs = env.step(
        PrivacyOpsAction(
            action_type="draft_reply",
            content="We deleted everything immediately.",
        )
    )
    obs = env.step(PrivacyOpsAction(action_type="submit"))
    assert obs.done is True
    assert len(obs.improvement_lessons) >= 1
    assert "improvement_lessons" in obs.metadata["info"]


def test_step_limit_terminates_episode() -> None:
    env = PrivacyOpsXEnvironment()
    env.reset(task_id="easy_verified_access_with_injection", seed=0)
    obs = None
    for _ in range(15):
        obs = env.step(PrivacyOpsAction(action_type="inspect_case"))
    assert obs is not None
    assert obs.done is True
    assert obs.metadata["info"]["final_score"] >= 0.0
