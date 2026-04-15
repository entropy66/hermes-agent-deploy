from __future__ import annotations

from hermes_agent.loop import AgentLoop
from hermes_agent.types import TaskContext


def test_agent_loop_run_returns_expected_shape() -> None:
    loop = AgentLoop()
    result = loop.run("summarize latest ci failures", TaskContext(session_id="s1"))
    assert result.task_id.startswith("task-")
    assert isinstance(result.success, bool)
    assert "steps_executed" in result.metrics
    assert result.reflection.summary


def test_agent_loop_response_hides_internal_debug_notes() -> None:
    loop = AgentLoop()
    result = loop.run("帮我总结今天服务状态", TaskContext(session_id="s-debug"))
    assert "act_hint" not in result.response


def test_agent_loop_blocks_guarded_cleanup_without_permission() -> None:
    loop = AgentLoop()
    result = loop.run("please delete temp files safely", TaskContext(session_id="s2"))
    assert result.metrics["blocked_count"] >= 1
    assert "run_cleanup_command" in result.blocked_actions


def test_agent_loop_allows_guarded_cleanup_with_permission() -> None:
    loop = AgentLoop()
    result = loop.run(
        "please delete temp files safely",
        TaskContext(session_id="s3", allow_guarded_actions=True),
    )
    assert result.metrics["blocked_count"] == 0
