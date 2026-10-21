"""OpenAI adapter request shape and failure handling, against a fake client (no network, no key)."""
from types import SimpleNamespace

import pytest

from src.generation import llm_client
from src.generation.llm_client import OpenAILLM, get_llm, is_reasoning_model


def reply(text, finish="stop", tin=10, tout=5):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout))


class FakeClient:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class AuthenticationError(Exception):
    pass


class RateLimitError(Exception):
    pass


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)


def test_reasoning_model_detection():
    assert is_reasoning_model("gpt-5.4-nano") and is_reasoning_model("o3-mini")
    assert not is_reasoning_model("gpt-4.1-mini")


def test_reasoning_model_request_has_no_temperature_or_max_tokens():
    c = FakeClient(reply('{"answer": "x"}'))
    r = OpenAILLM("gpt-5.4-nano", client=c, reasoning_effort="low", reasoning_headroom=1000).complete("s", "u", 300)
    kw = c.calls[0]
    assert r.text == '{"answer": "x"}' and r.error is None
    assert "temperature" not in kw and "max_tokens" not in kw
    assert kw["max_completion_tokens"] == 1300 and kw["reasoning_effort"] == "low"


def test_non_reasoning_model_keeps_temperature():
    c = FakeClient(reply("ok"))
    OpenAILLM("gpt-4.1-mini", temperature=0.0, client=c).complete("s", "u", 300)
    kw = c.calls[0]
    assert kw["temperature"] == 0.0 and kw["max_completion_tokens"] == 300 and "reasoning_effort" not in kw


def test_empty_output_from_exhausted_budget_is_an_error():
    r = OpenAILLM("gpt-5.4-nano", client=FakeClient(reply("", finish="length", tout=2300))).complete("s", "u")
    assert r.text == "" and "budget" in r.error and r.tokens_out == 2300


def test_auth_error_is_not_retried():
    c = FakeClient(AuthenticationError("bad key"), reply("never reached"))
    r = OpenAILLM("gpt-5.4-nano", client=c).complete("s", "u")
    assert len(c.calls) == 1 and r.error.startswith("AuthenticationError")


def test_rate_limit_is_retried_then_succeeds():
    c = FakeClient(RateLimitError("slow down"), reply("ok"))
    r = OpenAILLM("gpt-5.4-nano", client=c).complete("s", "u")
    assert len(c.calls) == 2 and r.text == "ok" and r.error is None


def test_get_llm_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_llm({"provider": "openai", "model": "gpt-5.4-nano"})
