import pytest

from pulse.redact import name_terms, redact
from pulse.safety import is_confidential, keyword_flags, normalise, route
from pulse.schema import TaggedComment
from pulse.synthetic import CONCEPTS, ROMAN_URDU


@pytest.mark.parametrize("text,kind", [
    ("Call me on +971 50 123 4567 please", "phone"),
    ("رقمي 0501234567", "phone"),
    ("mujhse 0300-1234567 par rabta karein", "phone"),
    ("write to ali.khan@example.com", "email"),
    ("my badge ID 48213 was blocked", "id"),
    ("see https://intranet.example.com/x", "url"),
])
def test_redacts_contact_details(text, kind):
    out, kinds = redact(text)
    assert kind in kinds
    assert "[" in out


@pytest.mark.parametrize("text", [
    "The scaffolding at Unit 3 has no guardrails.",
    "The site bus leaves at 4:30 in the morning.",
    "We work twelve-hour shifts six days a week.",
    "Overtime delayed for 3 months since 2024.",
])
def test_does_not_redact_ordinary_numbers(text):
    out, kinds = redact(text)
    assert out == text and kinds == []


def test_redacts_names_and_first_names():
    terms = name_terms(["Tariq Mehmood", "Ayesha Siddiqui"])
    out, kinds = redact("Tariq listens to us, unlike Ayesha Siddiqui.", terms)
    assert out == "[NAME] listens to us, unlike [NAME]."
    assert kinds == ["name"]


def test_normalise_folds_script_variants():
    assert normalise("إهانة") == normalise("اهانه")
    assert normalise("ख़राब") == normalise("खराब")
    assert normalise("I CAN’T  sleep") == "i can't sleep"


@pytest.mark.parametrize("concept", [c for c in CONCEPTS if c.sensitive != "none"], ids=lambda c: c.id)
def test_backstop_catches_sensitive_sentences_in_every_language(concept):
    for lang, text in [*concept.text.items(), ("ur_rom", ROMAN_URDU[concept.id])]:
        assert keyword_flags(text), f"{concept.id} not caught in {lang}: {text}"


@pytest.mark.parametrize("text", [
    "Everyone here is treated with respect, whatever their nationality.",
    "الجميع هنا يُعامَلون باحترام بغض النظر عن جنسياتهم.",
])
def test_nationality_backstop_ignores_respectful_mentions(text):
    assert "discrimination" not in keyword_flags(text)


def test_backstop_is_quiet_on_most_ordinary_comments():
    ordinary = [c for c in CONCEPTS if c.sensitive == "none"]
    texts = [t for c in ordinary for t in c.text.values()]
    flagged = [t for t in texts if keyword_flags(t)]
    assert len(flagged) / len(texts) < 0.05, flagged


def test_route_marks_confidential_and_quality_reasons():
    t = TaggedComment(comment_id="1", text="x", text_redacted="x", sensitive_category="harassment", urgency="critical",
                      confidence=0.9, evidence="x", evidence_grounded=True)
    route(t)
    assert t.needs_review and is_confidential(t)
    assert t.review_reasons[:2] == ["model:harassment", "critical_urgency"]

    q = TaggedComment(comment_id="2", text="y", text_redacted="y", confidence=0.3, evidence="zzz", evidence_grounded=False)
    route(q)
    assert q.needs_review and not is_confidential(q)
    assert set(q.review_reasons) == {"low_confidence", "unverified_evidence"}

    k = TaggedComment(comment_id="3", text="y", text_redacted="y", confidence=0.9, keyword_flags=["safety_hazard"])
    route(k)
    assert is_confidential(k) and "keyword:safety_hazard" in k.review_reasons
