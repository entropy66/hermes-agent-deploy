from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, List, Optional


class StepType(str, Enum):
    OBSERVE = "observe"
    PLAN = "plan"
    ACT = "act"
    REFLECT = "reflect"
    SKILLIZE = "skillize"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_GUARD = "require_guard"


@dataclass
class Action:
    name: str
    command: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.LOW


@dataclass
class ActionResult:
    action_name: str
    success: bool
    output: str
    blocked: bool = False
    requires_guard: bool = False
    rollback_applied: bool = False
    latency_ms: int = 0
    tokens_used: int = 0


@dataclass
class MemoryEvent:
    task_id: str
    session_id: str
    timestamp: float
    action: str
    result: str
    confidence: float
    success: bool
    tags: List[str] = field(default_factory=list)
    latency_ms: int = 0
    token_usage: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryQuery:
    text: str = ""
    task_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    limit: int = 5
    include_short_term: bool = True
    include_long_term: bool = True
    include_skill_memory: bool = True


@dataclass
class CompactionPolicy:
    max_long_term_events: int = 2000
    expire_seconds: Optional[int] = 30 * 24 * 3600
    min_confidence: float = 0.25


@dataclass
class MemoryHit:
    event: MemoryEvent
    score: float


@dataclass
class SkillDraft:
    skill_id: str
    name: str
    trigger: str
    template_steps: List[str]
    fallback_steps: List[str]
    source_task_ids: List[str]
    confidence: float
    status: str = "draft"
    version: int = 0


@dataclass
class SkillEvaluation:
    skill_id: str
    score: float
    accepted: bool
    reason: str


@dataclass
class SkillExecutionStat:
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    @property
    def avg_latency_ms(self) -> int:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0
        return int(self.total_latency_ms / total)


@dataclass
class ModelTarget:
    name: str
    provider: str
    max_tokens: int


@dataclass
class RouteDecision:
    primary: ModelTarget
    fallback: Optional[ModelTarget]
    reason: str


@dataclass
class SafetyVerdict:
    decision: SafetyDecision
    reason: str
    risk_level: RiskLevel


@dataclass
class TaskContext:
    session_id: str
    user_id: str = "single-user"
    max_steps: int = 8
    allow_guarded_actions: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Reflection:
    summary: str
    confidence: float
    improvements: List[str]
    can_skillize: bool


@dataclass
class TaskResult:
    task_id: str
    success: bool
    response: str
    steps: List[ActionResult]
    reflection: Reflection
    skill_updates: List[str]
    blocked_actions: List[str]
    metrics: Dict[str, Any]


def new_task_id() -> str:
    return f"task-{int(time() * 1000)}"
