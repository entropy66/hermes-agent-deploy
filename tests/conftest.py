from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clean_openai_env(monkeypatch):
    keys = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_RESPONSES_URL",
        "OPENAI_MODEL",
        "OPENAI_REASONING_EFFORT",
        "OPENAI_STORE",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
