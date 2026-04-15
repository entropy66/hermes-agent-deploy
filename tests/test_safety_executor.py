from __future__ import annotations

from hermes_agent.executor import AutonomousExecutor
from hermes_agent.safety import KillSwitch, RateLimiter, SafetyPolicy
from hermes_agent.types import Action, SafetyDecision


def test_safety_policy_blocks_critical_command() -> None:
    policy = SafetyPolicy()
    verdict = policy.check(Action(name="danger", command="rm -rf /tmp/work"))
    assert verdict.decision == SafetyDecision.BLOCK


def test_safety_policy_requires_guard_for_delete() -> None:
    policy = SafetyPolicy()
    verdict = policy.check(Action(name="cleanup", command="delete temp files"))
    assert verdict.decision == SafetyDecision.REQUIRE_GUARD


def test_executor_blocks_guarded_action_without_permission() -> None:
    executor = AutonomousExecutor(safety_policy=SafetyPolicy())
    result = executor.execute(
        action=Action(name="cleanup", command="delete temp files"),
        runtime_state={},
        allow_guarded_actions=False,
    )
    assert result.blocked is True
    assert result.requires_guard is True


def test_executor_kill_switch_stops_execution() -> None:
    kill = KillSwitch()
    kill.activate()
    executor = AutonomousExecutor(safety_policy=SafetyPolicy(), kill_switch=kill)
    result = executor.execute(action=Action(name="x", command="echo ok"), runtime_state={})
    assert result.blocked is True
    assert "kill switch" in result.output


def test_executor_rate_limit() -> None:
    limiter = RateLimiter(max_actions_per_minute=1)
    executor = AutonomousExecutor(safety_policy=SafetyPolicy(), rate_limiter=limiter)
    first = executor.execute(action=Action(name="a", command="echo one"), runtime_state={})
    second = executor.execute(action=Action(name="b", command="echo two"), runtime_state={})
    assert first.blocked is False
    assert second.blocked is True


def test_executor_rollback_on_failure() -> None:
    executor = AutonomousExecutor(safety_policy=SafetyPolicy())
    state = {"progress": "init"}
    set_result = executor.execute(
        action=Action(name="set", command="set progress=running"),
        runtime_state=state,
    )
    assert set_result.success is True
    assert state["progress"] == "running"

    fail_result = executor.execute(
        action=Action(name="fail", command="fail-sim"),
        runtime_state=state,
    )
    assert fail_result.success is False
    assert fail_result.rollback_applied is True
    assert state["progress"] == "running"
