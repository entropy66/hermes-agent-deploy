from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Protocol
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

    def __post_init__(self) -> None:
        self.api_key = (self.api_key or os.getenv("OPENAI_API_KEY", "")).strip() or None
        self.base_url = (self.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip()
        self.responses_url = self._build_responses_url(
            self.base_url,
            (self.responses_url or os.getenv("OPENAI_RESPONSES_URL", "")).strip() or None,
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
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "store": bool(self.store),
        }
        if with_reasoning and self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.responses_url,
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
            raise RuntimeError(f"responses api http {http_err.code}: {details}") from http_err
        except error.URLError as url_err:
            raise RuntimeError(f"responses api connection error: {url_err.reason}") from url_err

        parsed = json.loads(raw)
        output_text = self._extract_output_text(parsed)
        if not output_text:
            raise RuntimeError("responses api returned empty output_text")
        return output_text

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
