from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        # Slower, higher-depth response profile.
        summary = prompt.strip().replace("\n", " ")[:260]
        return f"[CLOUD:{self.name}] {summary}"


@dataclass
class FailingModelClient:
    name: str = "failing-model"
    error: str = "forced model failure"

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        raise RuntimeError(self.error)
