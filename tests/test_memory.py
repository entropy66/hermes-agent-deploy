from __future__ import annotations

from time import time

from hermes_agent.memory import InMemoryMemoryStore
from hermes_agent.types import CompactionPolicy, MemoryEvent, MemoryQuery


def test_memory_retrieve_by_text_and_tags() -> None:
    store = InMemoryMemoryStore(short_term_window=10)
    now = time()
    store.save(
        MemoryEvent(
            task_id="t1",
            session_id="s1",
            timestamp=now,
            action="analyze",
            result="build pipeline failed on unit tests",
            confidence=0.9,
            success=True,
            tags=["build", "ci"],
        )
    )
    store.save(
        MemoryEvent(
            task_id="t2",
            session_id="s1",
            timestamp=now,
            action="reflect",
            result="use retry for network tasks",
            confidence=0.7,
            success=True,
            tags=["network"],
        )
    )

    hits = store.retrieve(MemoryQuery(text="build", tags=["ci"], limit=3), session_id="s1")
    assert hits
    assert hits[0].event.task_id == "t1"


def test_memory_compaction_removes_old_low_confidence() -> None:
    store = InMemoryMemoryStore(short_term_window=10)
    old = time() - (40 * 24 * 3600)
    store.save(
        MemoryEvent(
            task_id="old",
            session_id="s1",
            timestamp=old,
            action="act",
            result="stale event",
            confidence=0.1,
            success=False,
            tags=[],
        )
    )
    store.save(
        MemoryEvent(
            task_id="new",
            session_id="s1",
            timestamp=time(),
            action="act",
            result="fresh event",
            confidence=0.9,
            success=True,
            tags=[],
        )
    )

    removed = store.compact(CompactionPolicy(max_long_term_events=100, expire_seconds=30 * 24 * 3600))
    assert removed == 1
    assert store.long_term_size() == 1
