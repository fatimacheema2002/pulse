import pytest

from fake_model import FakeModel
from pulse.providers import AuthError, ChatProvider, ModelOutputError, extract_json


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('<think>let me see {not json}</think>{"a": 1}', {"a": 1}),
    ('Here you go: {"a": {"b": "}"}} hope it helps', {"a": {"b": "}"}}),
    ('[{"id": "x"}]', {"results": [{"id": "x"}]}),
])
def test_extract_json(raw, expected):
    assert extract_json(raw) == expected


def test_extract_json_rejects_prose():
    with pytest.raises(ModelOutputError):
        extract_json("I could not do that.")


def _msgs():
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "<comments>\n[]\n</comments>"}]


def test_first_request_uses_schema_and_reasoning(meta_settings):
    fake = FakeModel()
    p = ChatProvider(meta_settings, transport=fake)
    p.complete_json(_msgs(), schema={"type": "object"}, reasoning_effort="low")
    body = fake.requests[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["reasoning_effort"] == "low"
    assert body["temperature"] == 0
    assert body["model"] == "muse-spark-1.3"


def test_steps_down_when_server_rejects_parameters(meta_settings):
    fake = FakeModel(reject={"response_format", "reasoning_effort"})
    p = ChatProvider(meta_settings, transport=fake)
    data, _ = p.complete_json(_msgs(), schema={"type": "object"}, reasoning_effort="low")
    assert data == {"results": []}
    last = fake.requests[-1]
    assert "response_format" not in last and "reasoning_effort" not in last
    assert p.format_mode == "none"
    assert any("reasoning_effort" in n for n in p.notes)
    # remembered for the next call: no more rejected requests
    n = len(fake.requests)
    p.complete_json(_msgs(), schema={"type": "object"}, reasoning_effort="low")
    assert len(fake.requests) == n + 1


def test_retries_rate_limits(meta_settings):
    fake = FakeModel(status_sequence=[429, 503, 200])
    p = ChatProvider(meta_settings, transport=fake, backoff_s=0)
    data, _ = p.complete_json(_msgs())
    assert data == {"results": []}
    assert p.usage.errors == 2 and p.usage.calls == 1


def test_auth_error_is_not_retried(meta_settings):
    fake = FakeModel(status_sequence=[401])
    p = ChatProvider(meta_settings, transport=fake, backoff_s=0)
    with pytest.raises(AuthError):
        p.complete_json(_msgs())
    assert len(fake.requests) == 1


def test_requires_configuration(meta_settings):
    from dataclasses import replace

    with pytest.raises(ValueError):
        ChatProvider(replace(meta_settings, api_key=""))
