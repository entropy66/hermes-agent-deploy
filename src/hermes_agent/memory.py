from __future__ import annotations

from collections import defaultdict, deque
from time import time
from typing import Deque, Dict, List

from .types import CompactionPolicy, MemoryEvent, MemoryHit, MemoryQuery


class InMemoryMemoryStore:
    """
    Three-tier memory:
    - short-term: recent session events
    - long-term: task history
    - skill memory: events tagged with "skill"
    """

    def __init__(self, short_term_window: int = 50) -> None:
        self.short_term_window = short_term_window
        self._short_term: Dict[str, Deque[MemoryEvent]] = defaultdict(
            lambda: deque(maxlen=self.short_term_window)
        )
        self._long_term: List[MemoryEvent] = []
        self._skill_memory: List[MemoryEvent] = []

    def save(self, event: MemoryEvent) -> None:
        self._short_term[event.session_id].append(event)
        self._long_term.append(event)
        if "skill" in event.tags:
            self._skill_memory.append(event)

    def retrieve(self, query: MemoryQuery, session_id: str | None = None) -> List[MemoryHit]:
        candidates: List[MemoryEvent] = []

        if query.include_short_term and session_id:
            candidates.extend(list(self._short_term.get(session_id, [])))
        if query.include_long_term:
            candidates.extend(self._long_term)
        if query.include_skill_memory:
            candidates.extend(self._skill_memory)

        seen = set()
        deduped: List[MemoryEvent] = []
        for event in candidates:
            # `(task_id, action, timestamp)` is stable enough for this MVP.
            key = (event.task_id, event.action, event.timestamp)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(event)

        ranked = sorted(
            (MemoryHit(event=e, score=self._score(e, query)) for e in deduped),
            key=lambda h: h.score,
            reverse=True,
        )
        return ranked[: query.limit]

    def compact(self, policy: CompactionPolicy) -> int:
        now = time()
        before = len(self._long_term)

        kept: List[MemoryEvent] = []
        for event in self._long_term:
            too_old = (
                policy.expire_seconds is not None
                and now - event.timestamp > policy.expire_seconds
            )
            low_conf = event.confidence < policy.min_confidence
            if too_old and low_conf:
                continue
            kept.append(event)

        if len(kept) > policy.max_long_term_events:
            kept = kept[-policy.max_long_term_events :]

        self._long_term = kept
        self._skill_memory = [e for e in self._long_term if "skill" in e.tags]
        self._rebuild_short_term()

        return before - len(self._long_term)

    def get_session_short_term(self, session_id: str) -> List[MemoryEvent]:
        return list(self._short_term.get(session_id, []))

    def long_term_size(self) -> int:
        return len(self._long_term)

    def _score(self, event: MemoryEvent, query: MemoryQuery) -> float:
        score = 0.0
        text_blob = f"{event.action} {event.result} {' '.join(event.tags)}".lower()
        q = query.text.lower().strip()
        if q and q in text_blob:
            score += 1.0
        if query.task_id and query.task_id == event.task_id:
            score += 0.6
        if query.tags:
            overlap = len(set(query.tags).intersection(set(event.tags)))
            if overlap:
                score += 0.4 + (0.1 * overlap)
        if event.success:
            score += 0.2
        score += event.confidence * 0.2
        return score

    def _rebuild_short_term(self) -> None:
        rebuilt: Dict[str, Deque[MemoryEvent]] = defaultdict(
            lambda: deque(maxlen=self.short_term_window)
        )
        for event in self._long_term:
            rebuilt[event.session_id].append(event)
        self._short_term = rebuilt
