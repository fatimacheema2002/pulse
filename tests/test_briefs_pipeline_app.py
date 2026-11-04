from pathlib import Path

import pandas as pd
import pytest

from fake_model import FakeModel
from pulse.briefs import build_briefs, check_brief, template_brief, team_stats
from pulse.config import from_env
from pulse.pipeline import ColumnMap, list_runs, load_run, run_survey
from pulse.providers import ChatProvider
from pulse.synthetic import CONCEPTS, ROMAN_URDU, generate, human_coded_sample
from pulse.taxonomy import DEFAULT_THEMES

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def offline_results(survey):
    res, man = run_survey(survey, from_env("offline"), save=False)
    return res


# -- synthetic data ------------------------------------------------------------------
def test_generator_is_deterministic():
    a, ga = generate(seed=3, scale=0.1)
    b, gb = generate(seed=3, scale=0.1)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ga, gb)


def test_every_concept_has_all_languages():
    for c in CONCEPTS:
        assert set(c.text) == {"en", "ar", "ur", "hi"} and all(c.text.values())
        assert ROMAN_URDU[c.id]


def test_gold_labels_follow_the_rules(gold_full):
    by_concepts = gold_full[gold_full["concepts"] != "NONE"]
    from pulse.synthetic import BY_ID

    for _, r in by_concepts.sample(200, random_state=0).iterrows():
        cs = [BY_ID[c] for c in r["concepts"].split(";")]
        pols = {c.sentiment for c in cs} - {"neutral"}
        expected = "mixed" if pols == {"positive", "negative"} else (pols.pop() if pols else "neutral")
        assert r["sentiment"] == expected


def test_human_sample_is_enriched_with_sensitive_comments(gold_full):
    s = human_coded_sample(gold_full)
    assert (s["sample_stratum"] == "enriched_sensitive").sum() > 0
    assert s["comment_id"].is_unique


# -- briefs -----------------------------------------------------------------------------
def test_small_groups_are_suppressed(offline_results):
    briefs = build_briefs(offline_results, "team", None, min_group_size=5)
    small = [b for b in briefs if b.status == "suppressed"]
    assert small and all(b.n_comments < 5 for b in small)
    assert all(not b.content for b in small)


def test_confidential_comments_never_reach_a_brief(offline_results):
    conf = offline_results[offline_results["confidential"]]
    team = conf["team"].iloc[0]
    briefs = build_briefs(offline_results, "team", None, groups=[team], min_group_size=1)
    usable = offline_results[(offline_results["team"] == team) & ~offline_results["confidential"]
                             & (offline_results["skipped"] == "")]
    assert briefs[0].n_comments == len(usable)


def test_template_brief_is_grounded(offline_results):
    usable = offline_results[~offline_results["confidential"] & (offline_results["skipped"] == "")]
    stats = team_stats(usable.head(40), usable, DEFAULT_THEMES)
    b = template_brief(stats)
    assert 1 <= len(b["actions"]) <= 3
    warnings = check_brief(b, stats, {t.id for t in DEFAULT_THEMES})
    assert all("actions" in w for w in warnings)  # numbers and themes always match the statistics


def test_check_brief_flags_invented_numbers_and_themes():
    stats = {"comments_analysed": 20, "high_urgency_count": 2,
             "sentiment_counts": {"positive": 5, "neutral": 3, "negative": 10, "mixed": 2},
             "themes": [{"theme": "workload", "label": "Workload", "mentions": 8, "negative": 6, "positive": 1,
                         "share_of_team_comments_pct": 40, "company_share_pct": 12}]}
    content = {"headline": "47 people are unhappy", "concerns": [{"theme": "salary", "point": "x"}],
               "actions": [{"theme": "workload", "action": "a", "first_step": "b"}] * 3}
    w = check_brief(content, stats, {"workload"})
    assert any("47" in x for x in w) and any("salary" in x for x in w)


def test_model_briefs_with_fake_model(offline_results, meta_settings):
    fake = FakeModel()
    provider = ChatProvider(meta_settings, transport=fake, backoff_s=0)
    briefs = build_briefs(offline_results, "site", provider, language="Arabic", min_group_size=5)
    ok = [b for b in briefs if b.status == "ok"]
    assert ok and all(b.source == "model" for b in ok)
    system = fake.requests[0]["messages"][0]["content"]
    user = fake.requests[0]["messages"][1]["content"]
    assert "Write all text values in Arabic" in system
    assert all(str(t) not in user for t in offline_results["text"].head(50) if len(str(t)) > 30)  # no original text sent
    md = ok[0].to_markdown({t.id: t.label for t in DEFAULT_THEMES}, "Site")
    assert "Actions for the next 30 days" in md


# -- pipeline -------------------------------------------------------------------------------
def test_column_guess(survey):
    cols = ColumnMap.guess(survey)
    assert cols.text == "comment" and cols.id == "respondent_id" and cols.names == ["manager_name"]


def test_run_save_and_load_roundtrip(survey, tmp_path):
    res, man = run_survey(survey.head(300), from_env("offline"), runs_dir=tmp_path)
    assert man["n_comments"] == len(res) and man["data_leaves_network"] is False
    runs = list_runs(tmp_path)
    assert runs[0]["run_id"] == man["run_id"]
    loaded, man2 = load_run(man["run_id"], tmp_path)
    assert len(loaded) == len(res) and isinstance(loaded["themes"].iloc[0], list)
    assert man2 == man


def test_manager_first_names_are_redacted(survey):
    res, _ = run_survey(survey, from_env("offline"), save=False)
    named = res[res["redactions"].str.contains("name")]
    assert len(named) > 0
    for _, r in named.iterrows():
        assert r["manager_name"].split()[0] not in r["text_redacted"]


# -- app smoke test ------------------------------------------------------------------------
def test_app_renders_every_tab(offline_results):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Pulse" in t.value for t in at.title)
    at.session_state["results"] = offline_results.copy()
    at.session_state["manifest"] = {"run_id": "test", "backend": "offline", "model": "keyword-baseline",
                                    "source": "synthetic_survey.csv", "stats": {}, "usage": {}}
    at.run()
    assert not at.exception, at.exception
    labels = [t.label for t in at.tabs]
    assert labels[0] == "Overview" and labels[3] == "Release gate"
    assert any("Release gate failed" in e.value for e in at.error)  # baseline does not pass
