from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Set
from urllib import error, request

from .loop import AgentLoop
from .types import TaskContext


class TelegramClientProtocol(Protocol):
    def get_updates(self, offset: Optional[int], timeout: int) -> List[Dict[str, Any]]:
        ...

    def send_message(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        ...


@dataclass
class TelegramBotClient:
    token: str
    api_base: str = "https://api.telegram.org"
    timeout_seconds: int = 70

    def get_updates(self, offset: Optional[int], timeout: int) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        data = self._post("getUpdates", payload)
        return data.get("result", [])

    def send_message(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        data = self._post("sendMessage", payload)
        return data.get("result", {})

    def _post(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_base.rstrip('/')}/bot{self.token}/{method}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as http_err:
            details = self._http_error_body(http_err)
            raise RuntimeError(f"telegram {method} http {http_err.code}: {details}") from http_err
        except error.URLError as url_err:
            raise RuntimeError(f"telegram {method} connection error: {url_err.reason}") from url_err

        parsed = json.loads(raw)
        if not parsed.get("ok"):
            raise RuntimeError(f"telegram {method} api error: {parsed}")
        return parsed

    def _http_error_body(self, err: error.HTTPError) -> str:
        try:
            if err.fp is None:
                return err.reason or "http error"
            return err.fp.read().decode("utf-8", errors="replace")
        except Exception:
            return err.reason or "http error"


@dataclass
class TelegramBridgeConfig:
    allowed_chat_ids: Optional[Set[int]]
    max_steps: int
    allow_guarded_actions: bool
    session_prefix: str = "tg"
    include_metrics: bool = True
    chunk_size: int = 3500


class TelegramBridge:
    def __init__(
        self,
        agent_loop: AgentLoop,
        client: TelegramClientProtocol,
        config: TelegramBridgeConfig,
    ) -> None:
        self.agent_loop = agent_loop
        self.client = client
        self.config = config

    def process_update(self, update: Dict[str, Any]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            return

        chat = message.get("chat", {})
        user = message.get("from", {})
        chat_id = int(chat.get("id", 0))
        message_id = message.get("message_id")
        user_id = str(user.get("id", "telegram-user"))

        if chat_id == 0:
            return

        if not self._is_allowed_chat(chat_id):
            self.client.send_message(chat_id, "Access denied for this chat.", message_id)
            return

        text = text.strip()
        if text in ("/start", "/help"):
            self.client.send_message(
                chat_id,
                "Hermes Telegram bridge is online.\nSend any text task and I will run Hermes Agent.",
                message_id,
            )
            return
        if text == "/ping":
            self.client.send_message(chat_id, "pong", message_id)
            return

        context = TaskContext(
            session_id=f"{self.config.session_prefix}-{chat_id}",
            user_id=user_id,
            max_steps=self.config.max_steps,
            allow_guarded_actions=self.config.allow_guarded_actions,
            metadata={"channel": "telegram", "chat_id": chat_id, "message_id": message_id},
        )

        try:
            result = self.agent_loop.run(text, context)
        except Exception as exc:
            self.client.send_message(chat_id, f"Hermes execution failed: {exc}", message_id)
            return

        reply = result.response
        if self.config.include_metrics:
            reply += (
                "\n\n"
                f"[task_id={result.task_id}] "
                f"success={result.success} "
                f"steps={result.metrics.get('steps_executed')} "
                f"blocked={result.metrics.get('blocked_count')} "
                f"latency_ms={result.metrics.get('estimated_latency_ms')}"
            )
        for part in chunk_text(reply, max_len=self.config.chunk_size):
            self.client.send_message(chat_id, part, message_id)

    def _is_allowed_chat(self, chat_id: int) -> bool:
        allowed = self.config.allowed_chat_ids
        if not allowed:
            return True
        return chat_id in allowed


def chunk_text(text: str, max_len: int = 3500) -> List[str]:
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    normalized = text or ""
    if len(normalized) <= max_len:
        return [normalized]

    chunks: List[str] = []
    current = normalized
    while len(current) > max_len:
        split = current.rfind("\n", 0, max_len)
        if split <= 0:
            split = max_len
        chunks.append(current[:split].strip())
        current = current[split:].lstrip()
    if current:
        chunks.append(current)
    return chunks


def parse_allowed_chat_ids(raw: Optional[str]) -> Optional[Set[int]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    values: Set[int] = set()
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.add(int(stripped))
    return values or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-agent-telegram",
        description="Telegram bridge for Hermes self-evolving agent",
    )
    parser.add_argument("--bot-token", default=os.getenv("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--allowed-chat-ids", default=os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", ""))
    parser.add_argument("--poll-timeout", type=int, default=int(os.getenv("TELEGRAM_POLL_TIMEOUT", "30")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("HERMES_MAX_STEPS", "8")))
    parser.add_argument("--allow-guarded", action="store_true")
    parser.add_argument(
        "--allow-guarded-from-env",
        action="store_true",
        help="Use HERMES_ALLOW_GUARDED=true|1 to allow guarded actions",
    )
    return parser


def run_polling(
    client: TelegramClientProtocol,
    bridge: TelegramBridge,
    poll_timeout: int,
) -> None:
    offset: Optional[int] = None
    while True:
        updates = client.get_updates(offset=offset, timeout=poll_timeout)
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                offset = update_id + 1
            bridge.process_update(update)
        # Prevent aggressive tight loops on empty returns.
        if not updates:
            time.sleep(0.2)


def main() -> None:
    args = build_arg_parser().parse_args()
    token = (args.bot_token or "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required")

    allow_guarded = bool(args.allow_guarded)
    if args.allow_guarded_from_env and os.getenv("HERMES_ALLOW_GUARDED", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        allow_guarded = True

    allowed_chat_ids = parse_allowed_chat_ids(args.allowed_chat_ids)
    config = TelegramBridgeConfig(
        allowed_chat_ids=allowed_chat_ids,
        max_steps=args.max_steps,
        allow_guarded_actions=allow_guarded,
    )
    bridge = TelegramBridge(
        agent_loop=AgentLoop(),
        client=TelegramBotClient(token=token),
        config=config,
    )
    run_polling(
        client=bridge.client,
        bridge=bridge,
        poll_timeout=max(1, args.poll_timeout),
    )
