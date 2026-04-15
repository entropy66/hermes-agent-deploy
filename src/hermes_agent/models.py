from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol
from urllib import error, request


class ModelClient(Protocol):
    name: str

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        ...


@dataclass
class LocalModelClient:
    name: str = "local-small"

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Fast, lower-depth response profile.
        summary = prompt.strip().replace("\n", " ")[:180]
        return f"[LOCAL:{self.name}] {summary}"


@dataclass
class CloudModelClient:
    name: str = "cloud-reasoner"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    responses_url: str | None = None
    reasoning_effort: str | None = None
    timeout_seconds: int = 45
    store: bool | None = None
    chat_completions_url: str | None = None

    def __post_init__(self) -> None:
        self.api_key = (self.api_key or os.getenv("OPENAI_API_KEY", "")).strip() or None
        self.base_url = (self.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip()
        self.responses_url = self._build_responses_url(
            self.base_url,
            (self.responses_url or os.getenv("OPENAI_RESPONSES_URL", "")).strip() or None,
        )
        self.response_urls = self._build_responses_url_candidates(
            self.base_url,
            self.responses_url,
        )
        explicit_chat_url = (
            self.chat_completions_url or os.getenv("OPENAI_CHAT_COMPLETIONS_URL", "")
        ).strip() or None
        self.chat_completions_url = self._build_chat_completions_url(self.base_url, explicit_chat_url)
        self.chat_completion_urls = self._build_chat_completions_url_candidates(
            self.base_url,
            self.chat_completions_url,
        )
        self.model = (self.model or os.getenv("OPENAI_MODEL", "gpt-5.4")).strip()
        self.reasoning_effort = (
            self.reasoning_effort or os.getenv("OPENAI_REASONING_EFFORT", "xhigh")
        ).strip()
        self.store = self._env_bool("OPENAI_STORE", default=False) if self.store is None else self.store

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        if not self.api_key:
            return self._simulate_response(prompt)

        try:
            return self._call_responses_api(prompt, max_tokens=max_tokens, with_reasoning=True)
        except RuntimeError as exc:
            if self.reasoning_effort and self._looks_like_reasoning_arg_error(exc):
                # Compatibility retry for gateways/models that reject reasoning config.
                return self._call_responses_api(prompt, max_tokens=max_tokens, with_reasoning=False)
            raise

    def _call_responses_api(
        self,
        prompt: str,
        max_tokens: int,
        with_reasoning: bool,
    ) -> str:
        payloads = self._payload_variants(prompt=prompt, max_tokens=max_tokens, with_reasoning=with_reasoning)
        errors: List[str] = []
        for url in self.response_urls:
            for payload in payloads:
                token_key = "max_output_tokens" if "max_output_tokens" in payload else "max_tokens"
                try:
                    parsed = self._post_json(url=url, payload=payload)
                    output_text = self._extract_output_text(parsed)
                    if output_text:
                        return output_text
                    errors.append(f"{url} ({token_key}): empty output_text")
                except RuntimeError as exc:
                    errors.append(f"{url} ({token_key}): {exc}")
                    continue

        chat_text, chat_errors = self._try_chat_completions(prompt=prompt, max_tokens=max_tokens)
        if chat_text:
            return chat_text
        errors.extend(chat_errors)
        compact = " | ".join(errors[:4])
        raise RuntimeError(f"model api failed across endpoints: {compact}")

    def _payload_variants(
        self,
        prompt: str,
        max_tokens: int,
        with_reasoning: bool,
    ) -> List[Dict[str, Any]]:
        base: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "store": bool(self.store),
        }
        if with_reasoning and self.reasoning_effort:
            base["reasoning"] = {"effort": self.reasoning_effort}
        return [
            {**base, "max_output_tokens": max_tokens},
            {**base, "max_tokens": max_tokens},
        ]

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as http_err:
            details = self._read_http_error_body(http_err)
            raise RuntimeError(f"http {http_err.code}: {details}") from http_err
        except error.URLError as url_err:
            raise RuntimeError(f"connection error: {url_err.reason}") from url_err
        return json.loads(raw)

    def _try_chat_completions(self, prompt: str, max_tokens: int) -> tuple[str, List[str]]:
        errors: List[str] = []
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        for url in self.chat_completion_urls:
            try:
                parsed = self._post_json(url=url, payload=payload)
                text = self._extract_chat_completion_text(parsed)
                if text:
                    return text, []
                errors.append(f"{url} (chat): empty choices")
            except RuntimeError as exc:
                errors.append(f"{url} (chat): {exc}")
                continue
        return "", errors

    def _extract_chat_completion_text(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text")
                        if isinstance(text, str) and text.strip():
                            parts.append(text.strip())
                return "\n".join(parts).strip()
        text = first.get("text")
        if isinstance(text, str):
            return text.strip()
        return ""

    def _extract_output_text(self, payload: Dict[str, Any]) -> str:
        direct = payload.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        collected: list[str] = []
        output = payload.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if isinstance(text, str) and text.strip():
                                collected.append(text.strip())
                text_field = item.get("text")
                if isinstance(text_field, str) and text_field.strip():
                    collected.append(text_field.strip())
        return "\n".join(collected).strip()

    def _simulate_response(self, prompt: str) -> str:
        summary = prompt.strip().replace("\n", " ")[:260]
        return f"[CLOUD:{self.name}] {summary}"

    def _looks_like_reasoning_arg_error(self, exc: RuntimeError) -> bool:
        lowered = str(exc).lower()
        return "reasoning" in lowered or "effort" in lowered

    def _build_responses_url(self, base_url: str, explicit_url: str | None) -> str:
        if explicit_url:
            return explicit_url.rstrip("/")
        normalized = base_url.rstrip("/")
        if normalized.endswith("/responses"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/responses"
        return f"{normalized}/v1/responses"

    def _build_chat_completions_url(self, base_url: str, explicit_url: str | None) -> str:
        if explicit_url:
            return explicit_url.rstrip("/")
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"

    def _build_responses_url_candidates(self, base_url: str, primary: str) -> List[str]:
        candidates: List[str] = []

        def add(url: str) -> None:
            url = url.rstrip("/")
            if url and url not in candidates:
                candidates.append(url)

        add(primary)
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            add(f"{normalized}/responses")
            add(f"{normalized[:-3]}/responses")
        elif normalized.endswith("/responses"):
            add(normalized)
            base = normalized[: -len("/responses")]
            add(f"{base}/v1/responses")
        else:
            add(f"{normalized}/responses")
            add(f"{normalized}/v1/responses")
        return candidates

    def _build_chat_completions_url_candidates(self, base_url: str, primary: str) -> List[str]:
        candidates: List[str] = []

        def add(url: str) -> None:
            url = url.rstrip("/")
            if url and url not in candidates:
                candidates.append(url)

        add(primary)
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            add(f"{normalized}/chat/completions")
            add(f"{normalized[:-3]}/chat/completions")
        elif normalized.endswith("/chat/completions"):
            add(normalized)
            base = normalized[: -len("/chat/completions")]
            add(f"{base}/v1/chat/completions")
        else:
            add(f"{normalized}/chat/completions")
            add(f"{normalized}/v1/chat/completions")
        return candidates

    def _env_bool(self, key: str, default: bool) -> bool:
        raw = os.getenv(key)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _read_http_error_body(self, err: error.HTTPError) -> str:
        try:
            if err.fp is None:
                return err.reason or "http error"
            body = err.fp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                msg = parsed.get("error")
                if isinstance(msg, dict):
                    text = msg.get("message")
                    if isinstance(text, str):
                        return text
                if isinstance(msg, str):
                    return msg
            return body
        except Exception:
            return err.reason or "http error"


@dataclass
class FailingModelClient:
    name: str = "failing-model"
    error: str = "forced model failure"

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        raise RuntimeError(self.error)
