"""Async client for PrivacyOps-X."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import PrivacyOpsAction, PrivacyOpsObservation, PrivacyOpsState
except ImportError:
    from models import PrivacyOpsAction, PrivacyOpsObservation, PrivacyOpsState


class PrivacyOpsXEnv(
    EnvClient[PrivacyOpsAction, PrivacyOpsObservation, PrivacyOpsState]
):
    """Typed client for the PrivacyOps-X environment."""

    def _step_payload(self, action: PrivacyOpsAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[PrivacyOpsObservation]:
        observation = payload.get("observation", {})
        result = PrivacyOpsObservation(
            **observation,
            done=payload.get("done", observation.get("done", False)),
            reward=payload.get("reward", observation.get("reward")),
        )
        return StepResult(
            observation=result,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> PrivacyOpsState:
        return PrivacyOpsState(**payload)
