from __future__ import annotations

import logging
from typing import Dict

from .models import ModelClient
from .types import ModelTarget, RiskLevel, RouteDecision, StepType

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Local-first routing with cloud fallback for complexity/risk.
    """

    def __init__(
        self,
        local_target: ModelTarget | None = None,
        cloud_target: ModelTarget | None = None,
    ) -> None:
        self.local_target = local_target or ModelTarget(
            name="local-small", provider="local", max_tokens=512
        )
        self.cloud_target = cloud_target or ModelTarget(
            name="cloud-reasoner", provider="cloud", max_tokens=1200
        )

    def select(self, step_type: StepType, risk_level: RiskLevel) -> ModelTarget:
        if risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            return self.cloud_target
        if step_type in (StepType.PLAN, StepType.REFLECT, StepType.SKILLIZE):
            return self.cloud_target
        return self.local_target

    def decide(self, step_type: StepType, risk_level: RiskLevel) -> RouteDecision:
        primary = self.select(step_type, risk_level)
        fallback = self.cloud_target if primary.provider == "local" else self.local_target
        reason = f"{step_type.value} with {risk_level.value} risk"
        return RouteDecision(primary=primary, fallback=fallback, reason=reason)

    def invoke(
        self,
        step_type: StepType,
        risk_level: RiskLevel,
        prompt: str,
        clients: Dict[str, ModelClient],
    ) -> str:
        decision = self.decide(step_type, risk_level)
        primary_client = clients.get(decision.primary.provider)
        if primary_client is None:
            raise RuntimeError(f"missing primary model client: {decision.primary.provider}")

        try:
            return primary_client.generate(prompt=prompt, max_tokens=decision.primary.max_tokens)
        except Exception as primary_err:
            if decision.fallback is None:
                raise
            fallback_client = clients.get(decision.fallback.provider)
            if fallback_client is None:
                raise RuntimeError(
                    f"missing fallback model client: {decision.fallback.provider}"
                )
            logger.warning(
                "Primary model failed (%s/%s). Falling back to %s. Error=%s",
                decision.primary.provider,
                decision.primary.name,
                decision.fallback.provider,
                primary_err,
            )
            return fallback_client.generate(
                prompt=prompt, max_tokens=decision.fallback.max_tokens
            )
