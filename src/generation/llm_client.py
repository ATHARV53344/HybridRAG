"""Provider-agnostic LLM adapter. Retrieval code never imports this module.

Providers: `mock` (offline heuristic reader - PIPELINE TESTING ONLY, not a language model),
`anthropic`, `openai`. API keys come from environment variables (.env), never from code.
Failures are returned in `LLMResponse.error`; they are never converted into answers.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str = ""
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None


class LLMClient:
    provider = "base"
    model = "base"

    def complete(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------------------
class MockLLM(LLMClient):
    """Deterministic extractive heuristic. NOT a language model; exists so the whole pipeline,
    metrics and reports can be exercised without an API key. Never report its numbers as LLM results."""

    provider, model = "mock", "mock-extractive-v1"

    def complete(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:
        t0 = time.perf_counter()
        if "Passage [" in user and "Claim:" in user:  # verification request
            text = self._verify(user)
        elif "Passages:" in user:
            text = self._answer(user)
        else:  # no-retrieval: the mock has no parametric knowledge
            text = json.dumps({"answer": "insufficient evidence", "citations": [], "sufficient": False, "facts": []})
        return LLMResponse(text, time.perf_counter() - t0, len(user.split()), len(text.split()))

    @staticmethod
    def _answer(user: str) -> str:
        from src.indexing.entities import extract_entities, split_sentences

        q = re.search(r"Question: (.*)", user).group(1)
        ctx = user.split("Passages:")[1].split("Instructions:")[0]
        qtoks = set(re.findall(r"\w+", q.lower()))
        best, best_pid, best_ov = None, None, 0
        for line in ctx.splitlines():
            m = re.match(r"\[(p_\w+)\] [^:]*: (.*)", line)
            if not m:
                continue
            for sent in split_sentences(m.group(2)):
                ov = len(qtoks & set(re.findall(r"\w+", sent.lower())))
                if ov > best_ov:
                    best, best_pid, best_ov = sent, m.group(1), ov
        if not best or best_ov < 2:
            return json.dumps({"answer": "insufficient evidence", "citations": [], "sufficient": False, "facts": []})
        cands = [e for e in extract_entities(best) if e not in q.lower()]
        nums = [n for n in re.findall(r"\b\d{4}\b", best) if n not in q]
        ans = (nums[0] if nums and re.match(r"(when|what year)", q.lower()) else (cands[-1] if cands else (nums[0] if nums else "")))
        if not ans:
            return json.dumps({"answer": "insufficient evidence", "citations": [], "sufficient": False, "facts": []})
        facts, cites, seen = [best], [best_pid], {ans}
        # crude multi-hop chaining (max 2 extra hops) for questions with nested-reference cues
        from src.routing.features import estimate_hops

        sents = []
        for line in ctx.splitlines():
            m = re.match(r"\[(p_\w+)\] [^:]*: (.*)", line)
            if m:
                sents += [(m.group(1), s) for s in split_sentences(m.group(2))]
        for _ in range(min(2, estimate_hops(q) - 1)):
            nxt = [(pid, s) for pid, s in sents if ans in s.lower() and s != facts[-1] and s not in facts
                   and len(qtoks & set(re.findall(r"\w+", s.lower()))) >= 1]
            nxt.sort(key=lambda ps: -len(qtoks & set(re.findall(r"\w+", ps[1].lower()))))
            moved = False
            for pid, s in nxt:
                new = [e for e in extract_entities(s) if e not in q.lower() and e not in seen and e != ans]
                if new:
                    ans, moved = new[-1], True
                    seen.add(ans)
                    facts.append(s)
                    cites.append(pid)
                    break
            if not moved:
                break
        return json.dumps({"answer": ans, "citations": cites, "sufficient": True, "facts": facts[:3]})

    @staticmethod
    def _verify(user: str) -> str:
        text = re.search(r"Passage \[[^\]]*\]: (.*)", user).group(1).lower()
        claim = re.search(r"Claim: (.*)", user).group(1).lower()
        ct = [t for t in re.findall(r"\w+", claim) if len(t) > 2]
        score = sum(t in text for t in ct) / max(len(ct), 1)
        label = "supported" if score >= 0.8 else "not_enough_info"
        return json.dumps({"label": label, "score": round(score, 3)})


# ---------------------------------------------------------------------------------------
class AnthropicLLM(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, temperature: float = 0.0, timeout_s: float = 60, retries: int = 3):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env or use generation.provider: mock.")
        import anthropic

        self.client = anthropic.Anthropic(api_key=key, timeout=timeout_s)
        self.model, self.temperature, self.retries = model, temperature, retries

    def complete(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:
        t0, err = time.perf_counter(), None
        for attempt in range(self.retries):
            try:
                r = self.client.messages.create(model=self.model, max_tokens=max_tokens, temperature=self.temperature,
                                                system=system, messages=[{"role": "user", "content": user}])
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                return LLMResponse(text, time.perf_counter() - t0, r.usage.input_tokens, r.usage.output_tokens)
            except Exception as e:  # rate limit / timeout / network - retry then report
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
        return LLMResponse("", time.perf_counter() - t0, error=err)


_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
# 4xx errors that a retry cannot fix (bad key, bad request, unknown model) - fail fast, don't burn retries
_NON_RETRYABLE = ("AuthenticationError", "PermissionDeniedError", "BadRequestError", "NotFoundError",
                  "UnprocessableEntityError")


def is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(_REASONING_PREFIXES)


class OpenAILLM(LLMClient):
    """Chat Completions adapter.

    Current models take `max_completion_tokens` (not `max_tokens`). Reasoning models (gpt-5*, o-series)
    reject a custom temperature and spend hidden reasoning tokens out of the same completion budget, so
    for them temperature is omitted, `reasoning_effort` is sent and `reasoning_headroom` extra tokens are
    added to the cap - otherwise the visible JSON answer can come back empty. Reasoning tokens are billed
    as output and are included in `tokens_out`.
    """

    provider = "openai"

    def __init__(self, model: str, temperature: float = 0.0, timeout_s: float = 60, retries: int = 3,
                 reasoning: bool | None = None, reasoning_effort: str | None = "low", reasoning_headroom: int = 2000,
                 client=None):
        if client is None:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or use generation.provider: mock.")
            from openai import OpenAI

            client = OpenAI(api_key=key, timeout=timeout_s)
        self.client = client
        self.model, self.temperature, self.retries = model, temperature, retries
        self.reasoning = is_reasoning_model(model) if reasoning is None else reasoning
        self.reasoning_effort, self.reasoning_headroom = reasoning_effort, reasoning_headroom

    def request_kwargs(self, system: str, user: str, max_tokens: int) -> dict:
        kw = {"model": self.model,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.reasoning:
            kw["max_completion_tokens"] = max_tokens + self.reasoning_headroom
            if self.reasoning_effort:
                kw["reasoning_effort"] = self.reasoning_effort
        else:
            kw["max_completion_tokens"] = max_tokens
            kw["temperature"] = self.temperature
        return kw

    def complete(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:
        t0, err = time.perf_counter(), None
        kw = self.request_kwargs(system, user, max_tokens)
        for attempt in range(self.retries):
            try:
                r = self.client.chat.completions.create(**kw)
                text = r.choices[0].message.content or ""
                if not text and r.choices[0].finish_reason == "length":
                    return LLMResponse("", time.perf_counter() - t0, r.usage.prompt_tokens, r.usage.completion_tokens,
                                       error="empty output: completion budget used up (raise reasoning_headroom "
                                             "or lower reasoning_effort)")
                return LLMResponse(text, time.perf_counter() - t0, r.usage.prompt_tokens, r.usage.completion_tokens)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if type(e).__name__ in _NON_RETRYABLE:
                    break
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
        return LLMResponse("", time.perf_counter() - t0, error=err)


def get_llm(cfg: dict) -> LLMClient:
    p = cfg.get("provider", "mock")
    if p == "mock":
        return MockLLM()
    if p == "anthropic":
        return AnthropicLLM(cfg["model"], cfg.get("temperature", 0.0), cfg.get("timeout_s", 60))
    if p == "openai":
        return OpenAILLM(cfg["model"], cfg.get("temperature", 0.0), cfg.get("timeout_s", 60),
                         reasoning=cfg.get("reasoning"), reasoning_effort=cfg.get("reasoning_effort", "low"),
                         reasoning_headroom=cfg.get("reasoning_headroom", 2000))
    raise ValueError(f"Unknown LLM provider '{p}'")
