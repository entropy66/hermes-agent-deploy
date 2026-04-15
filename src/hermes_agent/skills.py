from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .types import ActionResult, SkillDraft, SkillEvaluation, SkillExecutionStat


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, SkillDraft] = {}
        self._stats: Dict[str, SkillExecutionStat] = defaultdict(SkillExecutionStat)

    def propose(self, skill_draft: SkillDraft) -> str:
        self._skills[skill_draft.skill_id] = skill_draft
        return skill_draft.skill_id

    def evaluate(self, skill_id: str) -> SkillEvaluation:
        skill = self._skills.get(skill_id)
        if skill is None:
            return SkillEvaluation(
                skill_id=skill_id,
                score=0.0,
                accepted=False,
                reason="skill not found",
            )

        stat = self._stats[skill_id]
        usage_bonus = min(0.2, (stat.success_count + stat.failure_count) * 0.02)
        latency_penalty = 0.0 if stat.avg_latency_ms < 1500 else 0.1
        score = max(0.0, min(1.0, skill.confidence + usage_bonus - latency_penalty))
        accepted = score >= 0.55
        reason = "accepted" if accepted else "below publish threshold"
        return SkillEvaluation(skill_id=skill_id, score=score, accepted=accepted, reason=reason)

    def publish(self, skill_id: str, version: Optional[int] = None) -> Optional[SkillDraft]:
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        if version is None:
            skill.version += 1
        else:
            skill.version = version
        skill.status = "published"
        return skill

    def match(self, task_text: str) -> List[SkillDraft]:
        text = self._normalize(task_text)
        candidates = [
            skill
            for skill in self._skills.values()
            if skill.status == "published" and self._normalize(skill.trigger) in text
        ]
        candidates.sort(key=lambda s: s.confidence, reverse=True)
        return candidates

    def get(self, skill_id: str) -> Optional[SkillDraft]:
        return self._skills.get(skill_id)

    def list_published(self) -> List[SkillDraft]:
        return [s for s in self._skills.values() if s.status == "published"]

    def record_execution(self, skill_id: str, success: bool, latency_ms: int) -> None:
        stat = self._stats[skill_id]
        if success:
            stat.success_count += 1
        else:
            stat.failure_count += 1
        stat.total_latency_ms += max(0, latency_ms)

    def _normalize(self, value: str) -> str:
        normalized = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in value)
        return " ".join(normalized.split())


class SkillFactory:
    """
    Distills frequently successful trajectories into reusable skills.
    """

    def __init__(self, success_threshold: int = 3) -> None:
        self.success_threshold = success_threshold
        self._success_counter: Dict[str, int] = defaultdict(int)
        self._published_signatures: set[str] = set()

    def maybe_distill(
        self,
        task_text: str,
        task_id: str,
        trajectory: List[ActionResult],
        reflection_summary: str,
    ) -> Optional[SkillDraft]:
        if not trajectory:
            return None
        if not all(step.success and not step.blocked for step in trajectory):
            return None

        signature = self._signature(task_text)
        signature_key = self._slug(signature)
        self._success_counter[signature_key] += 1
        if self._success_counter[signature_key] < self.success_threshold:
            return None
        if signature_key in self._published_signatures:
            return None

        self._published_signatures.add(signature_key)
        steps = [step.action_name for step in trajectory]
        fallback = ["replan_with_cloud_model", "decompose_task", "ask_for_clarification"]
        return SkillDraft(
            skill_id=f"skill-{signature_key}",
            name=f"AutoSkill: {signature}",
            trigger=signature,
            template_steps=steps,
            fallback_steps=fallback,
            source_task_ids=[task_id],
            confidence=0.6 + min(0.3, self._success_counter[signature_key] * 0.05),
        )

    def _signature(self, task_text: str) -> str:
        normalized = "".join(
            ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in task_text
        )
        parts = [p for p in normalized.split() if p]
        return " ".join(parts[:4]) or "general task"

    def _slug(self, value: str) -> str:
        return "-".join(value.split())
