from __future__ import annotations

from hermes_agent.models import CloudModelClient, FailingModelClient, LocalModelClient
from hermes_agent.router import ModelRouter
from hermes_agent.types import RiskLevel, StepType


def test_router_selects_local_for_low_risk_observe(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    router = ModelRouter()
    target = router.select(StepType.OBSERVE, RiskLevel.LOW)
    assert target.provider == "local"


def test_router_selects_cloud_for_planning(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    router = ModelRouter()
    target = router.select(StepType.PLAN, RiskLevel.LOW)
    assert target.provider == "cloud"


def test_router_fallback_when_primary_fails(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    router = ModelRouter()
    out = router.invoke(
        StepType.OBSERVE,
        RiskLevel.LOW,
        prompt="hello",
        clients={
            "local": FailingModelClient(),
            "cloud": CloudModelClient(name="cloud-ok"),
        },
    )
    assert "CLOUD:cloud-ok" in out


def test_router_primary_cloud_without_failure(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    router = ModelRouter()
    out = router.invoke(
        StepType.PLAN,
        RiskLevel.HIGH,
        prompt="complex plan",
        clients={
            "local": LocalModelClient(),
            "cloud": CloudModelClient(name="cloud-main"),
        },
    )
    assert "CLOUD:cloud-main" in out
