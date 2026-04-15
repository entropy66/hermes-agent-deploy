from __future__ import annotations

from hermes_agent.loop import AgentLoop
from hermes_agent.types import TaskContext


def test_replay_improves_latency_and_step_count_with_skillization() -> None:
    loop = AgentLoop()
    task = "summarize build failures and propose next steps"
    session_id = "integration-replay"

    rounds = []
    for _ in range(10):
        rounds.append(loop.run(task, TaskContext(session_id=session_id)))

    first = rounds[0]
    last = rounds[-1]

    assert first.success is True
    assert last.success is True
    assert last.metrics["steps_executed"] <= first.metrics["steps_executed"]
    assert last.metrics["estimated_latency_ms"] <= first.metrics["estimated_latency_ms"]
    assert loop.skill_registry.list_published()
