from __future__ import annotations

import copy
import re
from collections import deque
from time import time
from typing import Any, Deque, Dict, Optional

from .types import Action, RiskLevel, SafetyDecision, SafetyVerdict


class SafetyPolicy:
    def __init__(self) -> None:
        self.hard_block_patterns = [
            r"\brm\s+-rf\b",
            r"\bdel\s+/f\s+/s\s+/q\b",
            r"\bformat\b",
            r"\bmkfs\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bpoweroff\b",
            r"\buserdel\b",
            r"\bnet\s+user\b",
            r"\bDROP\s+DATABASE\b",
        ]
        self.guard_patterns = [
            r"\bdelete\b",
            r"\btruncate\b",
            r"\bdrop\s+table\b",
            r"\btransfer\b",
            r"\bsend\s+money\b",
            r"\binvoke-webrequest\b",
            r"\bcurl\b",
            r"\bwget\b",
            r"\bscp\b",
            r"\bgit\s+push\b",
        ]

    def check(self, action: Action) -> SafetyVerdict:
        command = (action.command or "").strip()
        if not command:
            return SafetyVerdict(
                decision=SafetyDecision.ALLOW,
                reason="non-shell action",
                risk_level=RiskLevel.LOW,
            )

        lowered = command.lower()
        for pattern in self.hard_block_patterns:
            if re.search(pattern, lowered, re.IGNORECASE):
                return SafetyVerdict(
                    decision=SafetyDecision.BLOCK,
                    reason=f"hard-block policy matched: {pattern}",
                    risk_level=RiskLevel.CRITICAL,
                )
        for pattern in self.guard_patterns:
            if re.search(pattern, lowered, re.IGNORECASE):
                return SafetyVerdict(
                    decision=SafetyDecision.REQUIRE_GUARD,
                    reason=f"guard policy matched: {pattern}",
                    risk_level=RiskLevel.HIGH,
                )

        return SafetyVerdict(
            decision=SafetyDecision.ALLOW,
            reason="passes policy",
            risk_level=self.classify_risk(action),
        )

    def classify_risk(self, action: Action) -> RiskLevel:
        command = (action.command or "").lower()
        if any(keyword in command for keyword in ("http", "https", "invoke-webrequest", "curl")):
            return RiskLevel.MEDIUM
        if any(keyword in command for keyword in ("write", "update", "insert")):
            return RiskLevel.MEDIUM
        return action.risk_level


class RateLimiter:
    def __init__(self, max_actions_per_minute: int = 120) -> None:
        self.max_actions_per_minute = max_actions_per_minute
        self._timestamps: Deque[float] = deque()

    def allow(self) -> bool:
        now = time()
        window_start = now - 60
        while self._timestamps and self._timestamps[0] < window_start:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_actions_per_minute:
            return False
        self._timestamps.append(now)
        return True


class KillSwitch:
    def __init__(self) -> None:
        self._enabled = False

    def activate(self) -> None:
        self._enabled = True

    def deactivate(self) -> None:
        self._enabled = False

    @property
    def is_active(self) -> bool:
        return self._enabled


class ContextRollbackManager:
    def __init__(self) -> None:
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._counter = 0

    def snapshot(self, state: Dict[str, Any]) -> str:
        self._counter += 1
        snapshot_id = f"snap-{self._counter}"
        self._snapshots[snapshot_id] = copy.deepcopy(state)
        return snapshot_id

    def rollback(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        snap = self._snapshots.get(snapshot_id)
        if snap is None:
            return None
        return copy.deepcopy(snap)
