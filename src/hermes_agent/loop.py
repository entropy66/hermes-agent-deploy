from __future__ import annotations

from time import perf_counter, time
from typing import Any, Dict, List

from .executor import AutonomousExecutor
from .memory import InMemoryMemoryStore
from .models import CloudModelClient, LocalModelClient, ModelClient
from .router import ModelRouter
from .safety import SafetyPolicy
from .skills import SkillFactory, SkillRegistry
from .types import (
    Action,
    ActionResult,
    CompactionPolicy,
    MemoryEvent,
    MemoryQuery,
    Reflection,
    RiskLevel,
    StepType,
    TaskContext,
    TaskResult,
    new_task_id,
)


class AgentLoop:
    def __init__(
        self,
        memory_store: InMemoryMemoryStore | None = None,
        skill_registry: SkillRegistry | None = None,
        model_router: ModelRouter | None = None,
        safety_policy: SafetyPolicy | None = None,
        executor: AutonomousExecutor | None = None,
        skill_factory: SkillFactory | None = None,
        model_clients: Dict[str, ModelClient] | None = None,
    ) -> None:
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.skill_registry = skill_registry or SkillRegistry()
        self.model_router = model_router or ModelRouter()
        self.safety_policy = safety_policy or SafetyPolicy()
        self.executor = executor or AutonomousExecutor(self.safety_policy)
        self.skill_factory = skill_factory or SkillFactory(success_threshold=3)
        self.model_clients = model_clients or {
            "local": LocalModelClient(),
            "cloud": CloudModelClient(),
        }
        self._register_default_tools()

    def run(self, task: str, context: TaskContext) -> TaskResult:
        task_id = new_task_id()
        started = perf_counter()
        runtime_state: Dict[str, Any] = {
            "task": task,
            "notes": [],
            "response": "",
            "observed_memories": [],
            "matched_skills": [],
            "metadata": dict(context.metadata),
        }

        blocked_actions: List[str] = []
        skill_updates: List[str] = []
        action_results: List[ActionResult] = []
        total_tokens = 0
        total_latency_ms = 0

        observe_hits = self.memory_store.retrieve(
            MemoryQuery(text=task, limit=6),
            session_id=context.session_id,
        )
        runtime_state["observed_memories"] = [hit.event.result for hit in observe_hits]
        matched_skills = self.skill_registry.match(task)
        runtime_state["matched_skills"] = [skill.skill_id for skill in matched_skills]

        observe_text = self.model_router.invoke(
            StepType.OBSERVE,
            RiskLevel.LOW,
            prompt=(
                f"Task: {task}\n"
                f"Observed memories: {len(observe_hits)}\n"
                f"Matched skills: {[s.name for s in matched_skills]}"
            ),
            clients=self.model_clients,
        )
        self._save_event(
            task_id=task_id,
            session_id=context.session_id,
            action="observe",
            result=observe_text,
            confidence=0.6,
            success=True,
            tags=["observe"],
        )

        plan_actions = self._build_plan(task, matched_skills)
        plan_text = self.model_router.invoke(
            StepType.PLAN,
            self._max_risk(plan_actions),
            prompt=f"Create concise plan for task: {task}. Candidate actions: {[a.name for a in plan_actions]}",
            clients=self.model_clients,
        )
        self._save_event(
            task_id=task_id,
            session_id=context.session_id,
            action="plan",
            result=plan_text,
            confidence=0.7,
            success=True,
            tags=["plan"],
        )

        for action in plan_actions[: context.max_steps]:
            act_hint = self.model_router.invoke(
                StepType.ACT,
                action.risk_level,
                prompt=f"Prepare action '{action.name}' for task '{task}'",
                clients=self.model_clients,
            )
            runtime_state["notes"].append(f"act_hint:{act_hint[:48]}")

            result = self.executor.execute(
                action=action,
                runtime_state=runtime_state,
                allow_guarded_actions=context.allow_guarded_actions,
            )
            action_results.append(result)
            total_tokens += result.tokens_used
            total_latency_ms += result.latency_ms
            if result.blocked:
                blocked_actions.append(action.name)

            self._save_event(
                task_id=task_id,
                session_id=context.session_id,
                action=action.name,
                result=result.output,
                confidence=0.55 if result.success else 0.35,
                success=result.success,
                tags=["act"] + (["blocked"] if result.blocked else []),
                latency_ms=result.latency_ms,
                token_usage=result.tokens_used,
            )

        success = self._is_success(action_results)
        reflection = self._reflect(
            task=task,
            success=success,
            action_results=action_results,
            blocked_actions=blocked_actions,
            matched_skills=[s.name for s in matched_skills],
        )
        self._save_event(
            task_id=task_id,
            session_id=context.session_id,
            action="reflect",
            result=reflection.summary,
            confidence=reflection.confidence,
            success=success,
            tags=["reflect"],
        )

        draft = self.skill_factory.maybe_distill(
            task_text=task,
            task_id=task_id,
            trajectory=[res for res in action_results if not res.blocked],
            reflection_summary=reflection.summary,
        )
        if draft is not None:
            sid = self.skill_registry.propose(draft)
            eval_result = self.skill_registry.evaluate(sid)
            if eval_result.accepted:
                published = self.skill_registry.publish(sid)
                if published is not None:
                    skill_updates.append(f"published {published.skill_id}@v{published.version}")
            else:
                skill_updates.append(f"kept draft {sid}: {eval_result.reason}")

        self._save_event(
            task_id=task_id,
            session_id=context.session_id,
            action="skillize",
            result=", ".join(skill_updates) if skill_updates else "no skill update",
            confidence=0.7 if skill_updates else 0.4,
            success=True,
            tags=["skillize", "skill"] if skill_updates else ["skillize"],
        )
        self.memory_store.compact(CompactionPolicy())

        response = self._compose_response(
            task=task,
            reflection=reflection,
            action_results=action_results,
            runtime_state=runtime_state,
            blocked_actions=blocked_actions,
        )
        total_run_ms = int((perf_counter() - started) * 1000)

        metrics = {
            "steps_executed": len(action_results),
            "blocked_count": len(blocked_actions),
            "estimated_latency_ms": total_latency_ms,
            "estimated_tokens": total_tokens,
            "wall_clock_ms": total_run_ms,
            "matched_skills": len(matched_skills),
            "skill_updates": len(skill_updates),
        }
        return TaskResult(
            task_id=task_id,
            success=success,
            response=response,
            steps=action_results,
            reflection=reflection,
            skill_updates=skill_updates,
            blocked_actions=blocked_actions,
            metrics=metrics,
        )

    def _build_plan(self, task: str, matched_skills: List[Any]) -> List[Action]:
        if matched_skills:
            top = matched_skills[0]
            condensed_steps = top.template_steps[:2] if len(top.template_steps) > 2 else top.template_steps
            actions = [Action(name=step, payload={"skill_id": top.skill_id}) for step in condensed_steps]
            actions.append(Action(name="draft_response", payload={"source": "skill"}))
            self.skill_registry.record_execution(top.skill_id, success=True, latency_ms=120)
            return actions

        actions = [
            Action(name="analyze_task", payload={"task": task}),
            Action(name="search_memory", payload={"query": task}),
        ]
        lowered = task.lower()
        if any(keyword in lowered for keyword in ("delete", "cleanup", "remove")):
            actions.append(
                Action(
                    name="run_cleanup_command",
                    command="delete temp_files",
                    risk_level=RiskLevel.HIGH,
                )
            )
        else:
            actions.append(Action(name="set_state", command="set progress=planning"))
        actions.append(Action(name="draft_response", payload={"source": "default"}))
        return actions

    def _reflect(
        self,
        task: str,
        success: bool,
        action_results: List[ActionResult],
        blocked_actions: List[str],
        matched_skills: List[str],
    ) -> Reflection:
        success_count = sum(1 for r in action_results if r.success and not r.blocked)
        total = max(1, len(action_results))
        ratio = success_count / total
        improvements: List[str] = []
        if blocked_actions:
            improvements.append("replace blocked commands with policy-safe alternatives")
        if not matched_skills:
            improvements.append("accumulate repeated successful trajectories for skill distillation")
        else:
            improvements.append("prioritize published skill templates for faster planning")
        if ratio < 1:
            improvements.append("decompose failing steps into smaller actions")

        reflect_text = self.model_router.invoke(
            StepType.REFLECT,
            RiskLevel.MEDIUM,
            prompt=f"Task: {task}\nSuccess ratio: {ratio:.2f}\nBlocked: {blocked_actions}\nImprovements: {improvements}",
            clients=self.model_clients,
        )

        confidence = min(0.95, 0.45 + ratio * 0.4 + (0.1 if success else 0.0))
        return Reflection(
            summary=reflect_text,
            confidence=confidence,
            improvements=improvements,
            can_skillize=success and ratio >= 0.75,
        )

    def _compose_response(
        self,
        task: str,
        reflection: Reflection,
        action_results: List[ActionResult],
        runtime_state: Dict[str, Any],
        blocked_actions: List[str],
    ) -> str:
        successful_actions = [r.action_name for r in action_results if r.success and not r.blocked]
        notes = runtime_state.get("notes", [])
        clean_notes = [note for note in notes if isinstance(note, str) and not note.startswith("act_hint:")]
        compact_notes = "; ".join(clean_notes[-6:]) if clean_notes else "none"
        prompt = (
            "You are Hermes, a concise and practical assistant.\n"
            f"User task: {task}\n"
            f"Successful actions: {successful_actions}\n"
            f"Blocked actions: {blocked_actions}\n"
            f"Execution notes: {compact_notes}\n"
            f"Reflection improvements: {reflection.improvements}\n"
            "Write a direct user-facing answer in the same language as the task. "
            "Do not reveal internal tool traces, model routing, or debug tags."
        )
        try:
            text = self.model_router.invoke(
                StepType.REFLECT,
                RiskLevel.MEDIUM,
                prompt=prompt,
                clients=self.model_clients,
            )
            return text.strip()
        except Exception:
            return (
                f"任务已执行完成。成功步骤: {successful_actions}; "
                f"阻塞步骤: {blocked_actions or '无'}; "
                f"置信度: {reflection.confidence:.2f}。"
            )

    def _is_success(self, action_results: List[ActionResult]) -> bool:
        non_blocked = [r for r in action_results if not r.blocked]
        if not non_blocked:
            return False
        return all(r.success for r in non_blocked)

    def _max_risk(self, actions: List[Action]) -> RiskLevel:
        ranking = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3,
        }
        if not actions:
            return RiskLevel.LOW
        return max(actions, key=lambda a: ranking[a.risk_level]).risk_level

    def _save_event(
        self,
        task_id: str,
        session_id: str,
        action: str,
        result: str,
        confidence: float,
        success: bool,
        tags: List[str],
        latency_ms: int = 0,
        token_usage: int = 0,
    ) -> None:
        event = MemoryEvent(
            task_id=task_id,
            session_id=session_id,
            timestamp=time(),
            action=action,
            result=result,
            confidence=confidence,
            success=success,
            tags=tags,
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
        self.memory_store.save(event)

    def _register_default_tools(self) -> None:
        def analyze_task(payload: Dict[str, Any], state: Dict[str, Any]) -> str:
            task = payload.get("task", state.get("task", ""))
            state["notes"].append(f"analyzed:{task}")
            return f"task analyzed: {task[:60]}"

        def search_memory(payload: Dict[str, Any], state: Dict[str, Any]) -> str:
            query = payload.get("query", "")
            memories = state.get("observed_memories", [])
            match_count = sum(1 for m in memories if query.lower() in m.lower())
            state["notes"].append(f"memory_matches:{match_count}")
            return f"memory matches for '{query}': {match_count}"

        def draft_response(payload: Dict[str, Any], state: Dict[str, Any]) -> str:
            source = payload.get("source", "default")
            notes = "; ".join(state.get("notes", []))
            state["response"] = f"[{source}] actionable answer built from notes: {notes or 'none'}"
            return state["response"]

        self.executor.register_tool("analyze_task", analyze_task)
        self.executor.register_tool("search_memory", search_memory)
        self.executor.register_tool("draft_response", draft_response)
