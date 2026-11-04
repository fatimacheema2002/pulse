from dataclasses import replace

import pandas as pd

from fake_model import FakeModel
from pulse.cache import TagCache
from pulse.pipeline import ColumnMap, prepare
from pulse.providers import ChatProvider
from pulse.tagger import CommentIn, Tagger, evidence_is_grounded, is_no_content


def _items(survey: pd.DataFrame, n: int = 40) -> list[CommentIn]:
    rows = survey[survey["comment"] != ""].head(n)
    _, items = prepare(rows, ColumnMap(text="comment", id="respondent_id", names=["manager_name"]))
    return items


def _tagger(settings, fake, cache=None):
    return Tagger(settings, provider=ChatProvider(settings, transport=fake, backoff_s=0),
                  cache=cache if cache is not None else TagCache(None))


def test_tags_every_comment_and_matches_labels(meta_settings, survey, gold_full):
    fake = FakeModel()
    items = _items(survey)
    out = _tagger(meta_settings, fake).tag(items)
    assert [o.comment_id for o in out] == [i.comment_id for i in items]
    gold = gold_full.set_index("comment_id")
    for o in out:
        if o.skipped:
            continue
        assert not o.error
        assert set(o.themes) == set(gold.loc[o.comment_id, "themes"].split(";"))
        assert o.evidence_grounded


def test_cache_means_no_second_bill(meta_settings, survey):
    cache = TagCache(None)
    fake = FakeModel()
    items = _items(survey)
    _tagger(meta_settings, fake, cache).tag(items)
    n = len(fake.requests)
    t2 = _tagger(meta_settings, fake, cache)
    t2.tag(items)
    assert len(fake.requests) == n
    assert t2.stats["cached"] + t2.stats["skipped"] == len(items)


def test_identical_comments_are_sent_once(meta_settings):
    fake = FakeModel(gold=pd.DataFrame(columns=["comment_id", "themes", "sentiment", "urgency", "sensitive_category", "language"]))
    items = [CommentIn(f"c{i}", "The food in the mess is poor.", "The food in the mess is poor.") for i in range(6)]
    t = _tagger(replace(meta_settings, recheck=False), fake)
    out = t.tag(items)
    assert len(fake.requests) == 1
    assert '"c0"' in fake.requests[0]["messages"][-1]["content"]
    assert '"c1"' not in fake.requests[0]["messages"][-1]["content"]
    assert len(out) == 6 and all(o.comment_id == f"c{i}" for i, o in enumerate(out))
    assert t.stats["deduplicated"] == 5


def test_no_content_comments_skip_the_model(meta_settings):
    assert is_no_content("N/A") and is_no_content("لا يوجد.") and is_no_content("Koi comment nahi.")
    fake = FakeModel()
    out = _tagger(meta_settings, fake).tag([CommentIn("a", "کچھ نہیں۔", "کچھ نہیں۔")])
    assert out[0].skipped == "no_content" and not fake.requests


def test_repairs_unparseable_output(meta_settings, survey):
    fake = FakeModel(garble_first=1)
    items = _items(survey, 5)
    out = _tagger(replace(meta_settings, batch_size=10, recheck=False), fake).tag(items)
    assert not any(o.error for o in out)
    assert "could not be used" in fake.requests[1]["messages"][-1]["content"]


def test_missing_ids_are_retried_one_by_one(meta_settings, survey):
    items = [i for i in _items(survey, 12) if not is_no_content(i.text)][:5]
    fake = FakeModel(drop_ids={items[2].comment_id})
    out = _tagger(replace(meta_settings, recheck=False), fake).tag(items)
    assert out[2].error  # dropped every time -> recorded as a failure, not silently lost
    assert "missing" in out[2].error
    assert all(not o.error for i, o in enumerate(out) if i != 2)


def test_api_failure_keeps_keyword_backstop(meta_settings):
    fake = FakeModel(status_sequence=[500] * 50)
    item = CommentIn("s1", "Our safety harnesses are old and damaged.", "Our safety harnesses are old and damaged.")
    out = _tagger(replace(meta_settings, recheck=False), fake).tag([item])
    assert out[0].error and out[0].needs_review
    assert "keyword:safety_hazard" in out[0].review_reasons


def test_second_look_uses_higher_reasoning_effort(meta_settings, survey, gold_full):
    sens_ids = set(gold_full.loc[gold_full["sensitive_category"] != "none", "comment_id"])
    rows = survey[survey["respondent_id"].isin(sens_ids)].head(4)
    _, items = prepare(rows, ColumnMap(text="comment", id="respondent_id", names=["manager_name"]))
    fake = FakeModel()
    t = _tagger(meta_settings, fake)
    out = t.tag(items)
    efforts = [r.get("reasoning_effort") for r in fake.requests]
    assert "high" in efforts
    assert t.stats["rechecked"] == len(items)
    assert all(o.rechecked and o.needs_review for o in out)


def test_evidence_grounding():
    assert evidence_is_grounded("الأولويات في فريقنا", "الأولويات في فريقنا تتغير كل يوم")
    assert evidence_is_grounded("“nobody explains why…”", "Overtime is late and nobody explains why.")
    assert not evidence_is_grounded("priorities change daily", "الأولويات في فريقنا تتغير كل يوم")
