import json

from src.generation.answer_generator import AnswerGenerator, GenResult, parse_json_answer
from src.generation.llm_client import LLMClient, LLMResponse, MockLLM
from src.retrieval.types import Evidence
from src.verification.evidence_verifier import EvidenceVerifier

VCFG = {"tau_u": 0.5, "tau_a": 0.8, "tau_b": 0.4}
EV = [Evidence("p_1", "T", "Mira Valdoren was born in Eldoria.", 1.0, "vector")]


class Scripted(LLMClient):
    provider, model = "scripted", "s"

    def __init__(self, text="", error=None):
        self.text, self.error = text, error

    def complete(self, system, user, max_tokens=300):
        return LLMResponse(self.text, 0.01, 5, 5, self.error)


def gen(answer="Eldoria", facts=("Mira Valdoren was born in Eldoria.",), sufficient=True):
    return GenResult(answer=answer, facts=list(facts), sufficient=sufficient, parse_ok=True)


def test_supported_answer_accepted():
    v = EvidenceVerifier(VCFG, backend="lexical").verify("q", gen(), EV)
    assert v.decision == "accept" and v.kappa == 1.0 and v.claims[0].evidence_ids == ["p_1"]


def test_unsupported_answer_abstains():
    g = gen("Atlantis", facts=("Atlantis is the capital of the moon.",))
    v = EvidenceVerifier(VCFG, backend="lexical").verify("q", g, EV)
    assert v.decision == "abstain" and v.kappa == 0.0


def test_partial_support_is_qualified():
    g = gen("Eldoria", facts=("Zorblax invented teleportation in Narnia.",))
    v = EvidenceVerifier(VCFG, backend="lexical").verify("q", g, EV)
    assert v.decision == "qualify" and 0.4 <= v.kappa < 0.8


def test_insufficient_evidence_abstains_without_calls():
    v = EvidenceVerifier(VCFG, backend="lexical").verify("q", gen(), [])
    assert v.decision == "abstain"
    v2 = EvidenceVerifier(VCFG, backend="lexical").verify("q", gen(sufficient=False), EV)
    assert v2.decision == "abstain"


def test_malformed_verifier_output_is_not_supported():
    v = EvidenceVerifier(VCFG, Scripted("not json at all"), "llm").verify("q", gen(), EV)
    assert v.decision == "abstain" and v.error and v.error.startswith("parse_error")


def test_verifier_api_failure_recorded_not_supported():
    v = EvidenceVerifier(VCFG, Scripted("", "timeout"), "llm").verify("q", gen(), EV)
    assert v.decision == "abstain" and "llm_error" in v.error


def test_llm_verifier_parses_structured_output():
    out = json.dumps({"label": "supported", "score": 0.95})
    v = EvidenceVerifier(VCFG, Scripted(out), "llm").verify("q", gen(), EV)
    assert v.decision == "accept" and v.n_llm_calls >= 1


def test_invalid_thresholds_rejected():
    import pytest

    with pytest.raises(ValueError):
        EvidenceVerifier({"tau_u": 0.5, "tau_a": 0.3, "tau_b": 0.6})


# ---- generation failure handling ---------------------------------------------------------------------
def test_malformed_generation_recorded_as_error():
    g = AnswerGenerator(Scripted("blah blah")).generate("q", EV)
    assert g.error and g.error.startswith("parse_error") and g.answer == ""


def test_api_failure_never_becomes_an_answer():
    g = AnswerGenerator(Scripted("", "rate limit")).generate("q", EV)
    assert g.error and "llm_error" in g.error and g.answer == "" and not g.sufficient


def test_parse_json_with_surrounding_text():
    d = parse_json_answer('Sure! {"answer": "x", "citations": []} done')
    assert d["answer"] == "x"
    assert parse_json_answer("no json") is None


def test_mock_marks_itself():
    assert MockLLM().provider == "mock"
