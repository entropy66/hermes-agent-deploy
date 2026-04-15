from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hermes_agent.telegram_bridge import (
    TelegramBridge,
    TelegramBridgeConfig,
    chunk_text,
    parse_allowed_chat_ids,
)
from hermes_agent.types import Reflection, TaskResult


@dataclass
class FakeAgent:
    response_text: str = "ok"
    called: int = 0

    def run(self, task: str, context: Any) -> TaskResult:
        self.called += 1
        return TaskResult(
            task_id="task-test",
            success=True,
            response=self.response_text,
            steps=[],
            reflection=Reflection(
                summary="done",
                confidence=0.9,
                improvements=[],
                can_skillize=False,
            ),
            skill_updates=[],
            blocked_actions=[],
            metrics={"steps_executed": 1, "blocked_count": 0, "estimated_latency_ms": 120},
        )


@dataclass
class FakeTelegramClient:
    sent: List[Dict[str, Any]]

    def __init__(self) -> None:
        self.sent = []

    def get_updates(self, offset: Optional[int], timeout: int) -> List[Dict[str, Any]]:
        return []

    def send_message(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.sent.append(
            {"chat_id": chat_id, "text": text, "reply_to_message_id": reply_to_message_id}
        )
        return {}


def test_parse_allowed_chat_ids() -> None:
    assert parse_allowed_chat_ids("") is None
    assert parse_allowed_chat_ids(None) is None
    assert parse_allowed_chat_ids("1001, 1002") == {1001, 1002}


def test_chunk_text_respects_limit() -> None:
    text = "a" * 10 + "\n" + "b" * 10 + "\n" + "c" * 10
    chunks = chunk_text(text, max_len=12)
    assert len(chunks) >= 3
    assert all(len(c) <= 12 for c in chunks)


def test_bridge_denies_not_allowed_chat() -> None:
    agent = FakeAgent()
    client = FakeTelegramClient()
    bridge = TelegramBridge(
        agent_loop=agent,
        client=client,
        config=TelegramBridgeConfig(
            allowed_chat_ids={12345},
            max_steps=8,
            allow_guarded_actions=False,
        ),
    )

    bridge.process_update(
        {
            "update_id": 1,
            "message": {
                "message_id": 7,
                "chat": {"id": 999},
                "from": {"id": 111},
                "text": "hello",
            },
        }
    )
    assert agent.called == 0
    assert client.sent
    assert "Access denied" in client.sent[0]["text"]


def test_bridge_processes_message_and_replies() -> None:
    agent = FakeAgent(response_text="Hermes result")
    client = FakeTelegramClient()
    bridge = TelegramBridge(
        agent_loop=agent,
        client=client,
        config=TelegramBridgeConfig(
            allowed_chat_ids=None,
            max_steps=8,
            allow_guarded_actions=False,
        ),
    )
    bridge.process_update(
        {
            "update_id": 2,
            "message": {
                "message_id": 8,
                "chat": {"id": 100},
                "from": {"id": 200},
                "text": "do task",
            },
        }
    )

    assert agent.called == 1
    assert client.sent
    assert "Hermes result" in client.sent[0]["text"]
    assert "task_id=task-test" in client.sent[0]["text"]


def test_bridge_handles_ping_command() -> None:
    agent = FakeAgent()
    client = FakeTelegramClient()
    bridge = TelegramBridge(
        agent_loop=agent,
        client=client,
        config=TelegramBridgeConfig(
            allowed_chat_ids=None,
            max_steps=8,
            allow_guarded_actions=False,
        ),
    )
    bridge.process_update(
        {
            "update_id": 3,
            "message": {
                "message_id": 9,
                "chat": {"id": 100},
                "from": {"id": 200},
                "text": "/ping",
            },
        }
    )
    assert agent.called == 0
    assert client.sent[0]["text"] == "pong"
