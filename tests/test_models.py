from __future__ import annotations

from hermes_agent.models import CloudModelClient


def test_cloud_client_simulates_without_api_key() -> None:
    client = CloudModelClient(api_key=None, model="gpt-5.4")
    out = client.generate("hello world", max_tokens=64)
    assert out.startswith("[CLOUD:")


def test_cloud_client_builds_multiple_endpoint_candidates() -> None:
    client = CloudModelClient(
        api_key="test",
        base_url="https://api.negentropypath.xyz",
        responses_url="https://api.negentropypath.xyz/responses",
    )
    assert client.response_urls[0] == "https://api.negentropypath.xyz/responses"
    assert "https://api.negentropypath.xyz/v1/responses" in client.response_urls
    assert "https://api.negentropypath.xyz/v1/chat/completions" in client.chat_completion_urls


def test_cloud_client_retries_payload_field_variant() -> None:
    client = CloudModelClient(api_key="test", base_url="https://example.com")
    calls = []

    def fake_post(url, payload):
        calls.append((url, payload))
        if "max_output_tokens" in payload:
            raise RuntimeError("http 400: unknown field max_output_tokens")
        return {"output_text": "ok from max_tokens"}

    client.response_urls = ["https://example.com/v1/responses"]
    client._post_json = fake_post  # type: ignore[method-assign]
    out = client.generate("test", max_tokens=77)
    assert out == "ok from max_tokens"
    assert len(calls) >= 2


def test_cloud_client_falls_back_to_chat_completions_when_responses_fails() -> None:
    client = CloudModelClient(api_key="test", base_url="https://example.com")

    def fake_post(url, payload):
        if url.endswith("/responses"):
            raise RuntimeError("http 404: not found")
        if url.endswith("/chat/completions"):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "chat completion ok",
                        }
                    }
                ]
            }
        raise RuntimeError("unexpected endpoint")

    client.response_urls = ["https://example.com/responses"]
    client.chat_completion_urls = ["https://example.com/chat/completions"]
    client._post_json = fake_post  # type: ignore[method-assign]
    out = client.generate("hello", max_tokens=40)
    assert out == "chat completion ok"
