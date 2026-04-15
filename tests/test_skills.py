from __future__ import annotations

from hermes_agent.skills import SkillFactory, SkillRegistry
from hermes_agent.types import ActionResult, SkillDraft


def test_skill_registry_propose_evaluate_publish_match() -> None:
    registry = SkillRegistry()
    draft = SkillDraft(
        skill_id="skill-build",
        name="Build Recovery",
        trigger="build failure",
        template_steps=["analyze_task", "search_memory"],
        fallback_steps=["replan"],
        source_task_ids=["t1"],
        confidence=0.8,
    )
    skill_id = registry.propose(draft)
    eval_result = registry.evaluate(skill_id)
    assert eval_result.accepted is True
    published = registry.publish(skill_id)
    assert published is not None
    assert published.version == 1
    matched = registry.match("need help with build failure in ci")
    assert matched
    assert matched[0].skill_id == "skill-build"


def test_skill_factory_distills_after_threshold() -> None:
    factory = SkillFactory(success_threshold=2)
    trajectory = [
        ActionResult(action_name="analyze_task", success=True, output="ok"),
        ActionResult(action_name="draft_response", success=True, output="ok"),
    ]
    first = factory.maybe_distill(
        task_text="Summarize build failures",
        task_id="t1",
        trajectory=trajectory,
        reflection_summary="good",
    )
    second = factory.maybe_distill(
        task_text="Summarize build failures",
        task_id="t2",
        trajectory=trajectory,
        reflection_summary="good",
    )
    assert first is None
    assert second is not None
    assert second.skill_id.startswith("skill-summarize-build-failures")
