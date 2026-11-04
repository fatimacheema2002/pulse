import io
import math

import pytest

from pulse.config import from_env
from pulse.pipeline import run_survey
from pulse.validation import cohen_kappa, load_gold, multilabel_scores, validate


def test_kappa_reference_values():
    labels = ["a", "b"]
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"], labels) == pytest.approx(1.0)
    # Classic textbook example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b, ["y", "n"]) == pytest.approx(0.4)


def test_weighted_kappa_penalises_far_misses_more():
    levels = ["low", "medium", "high", "critical"]
    truth = ["low", "medium", "high", "critical"] * 5
    near = ["medium", "medium", "high", "high"] * 5
    far = ["critical", "medium", "high", "low"] * 5
    assert cohen_kappa(truth, near, levels, "quadratic") > cohen_kappa(truth, far, levels, "quadratic")


def test_multilabel_scores():
    pred = [{"a"}, {"a", "b"}, {"c"}]
    gold = [{"a"}, {"b"}, {"c", "a"}]
    per, s = multilabel_scores(pred, gold, ["a", "b", "c"])
    # tp=3 (a, b, c), fp=1 (a in row 2), fn=1 (a in row 3)
    assert s["micro_f1"] == pytest.approx(6 / 8)
    assert per.set_index("theme").loc["b", "f1"] == pytest.approx(1.0)


def test_perfect_predictions_pass_the_gate(gold_sample):
    pred = gold_sample.rename(columns={}).copy()
    pred["themes"] = pred["themes"].apply(sorted)
    pred["confidential"] = pred["sensitive_category"] != "none"
    rep = validate(pred, gold_sample)
    assert rep.passed
    assert rep.themes["micro_f1"] == pytest.approx(1.0)
    assert rep.sensitive_recall_routed == pytest.approx(1.0)
    lo, hi = rep.themes_ci
    assert lo <= 1.0 <= hi + 1e-9


def test_keyword_baseline_fails_the_gate(survey, gold_sample, tmp_path):
    res, _ = run_survey(survey, from_env("offline"), save=False)
    rep = validate(res, gold_sample)
    assert not rep.passed
    failed = {c.name for c in rep.checks if not c.passed}
    assert "Sentiment agreement (Cohen's kappa)" in failed
    assert not math.isnan(rep.sentiment_kappa)


def test_missing_ids_are_reported(gold_sample):
    pred = gold_sample.head(50).copy()
    pred["confidential"] = False
    rep = validate(pred, gold_sample)
    assert rep.n == 50 and rep.n_missing == len(gold_sample) - 50
    assert not rep.checks[0].passed  # sample too small


def test_load_gold_accepts_yes_no_sensitive_column():
    csv = "ID,Themes,Sentiment,Urgency,Sensitive\n1,pay_benefits|workload,Negative,High,yes\n2,training,positive,low,no\n"
    g = load_gold(io.StringIO(csv))
    assert list(g["comment_id"]) == ["1", "2"]
    assert g.loc[0, "themes"] == {"pay_benefits", "workload"}
    assert list(g["sensitive_category"]) == ["unspecified", "none"]
    assert list(g["sentiment"]) == ["negative", "positive"]
