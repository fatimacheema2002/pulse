"""End-to-end over real HTTP: a local OpenAI-compatible server stands in for vLLM / Meta Model API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from fake_model import FakeModel
from pulse.cache import TagCache
from pulse.config import Settings
from pulse.pipeline import ColumnMap, run_survey
from pulse.tagger import Tagger


def _serve(fake: FakeModel):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/v1/chat/completions"
            assert self.headers["Authorization"] == "Bearer local"
            status, data, _ = fake(self.path, dict(self.headers), body, 10)
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_open_weights_backend_over_http(survey):
    fake = FakeModel(reject={"reasoning_effort"})  # many self-hosted servers reject it
    server = _serve(fake)
    try:
        s = Settings(backend="open_weights", base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="local",
                     model="meta-models/Muse-Glimmer-30B", reasoning_effort="low", recheck_reasoning_effort="high",
                     batch_size=6, max_workers=2, cache_path=":memory:")
        assert s.leaves_network is False
        rows = survey[survey["comment"] != ""].head(25)
        tagger = Tagger(s, cache=TagCache(None))
        res, man = run_survey(rows, s, ColumnMap(text="comment", id="respondent_id", names=["manager_name"]),
                              tagger=tagger, save=False)
        assert len(res) == 25 and (res["error"] == "").all()
        assert man["data_leaves_network"] is False
        assert man["usage"]["calls"] >= 4
        assert any("reasoning_effort" in n for n in man["provider_notes"])
    finally:
        server.shutdown()


# -- local Ollama ------------------------------------------------------------------------------
def _serve_ollama(fake: FakeModel, installed=("llama3.1:8b", "llama3.2:3b")):
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, data):
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            assert self.path == "/api/tags"
            self._send(200, {"models": [{"name": n, "size": 4_900_000_000, "details": {"parameter_size": "8.0B"}}
                                        for n in installed]})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path == "/api/chat"
            seen.append({**body, "_auth_header": self.headers.get("Authorization")})
            if body["model"] not in installed:
                return self._send(404, {"error": f'model "{body["model"]}" not found, try pulling it first'})
            status, data, _ = fake(self.path, dict(self.headers), body, 10)
            content = data["choices"][0]["message"]["content"]
            self._send(status, {"model": body["model"], "message": {"role": "assistant", "content": content},
                                "done": True, "prompt_eval_count": 900, "eval_count": 150})

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_local_ollama_backend_without_api_key(survey, monkeypatch):
    from pulse.config import from_env

    server, seen = _serve_ollama(FakeModel())
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        for var in ("MODEL_API_KEY", "META_API_KEY", "PULSE_BACKEND", "PULSE_LOCAL_MODEL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("PULSE_LOCAL_BASE_URL", url)
        s = from_env()  # no key, Ollama running -> local backend chosen automatically
        assert s.backend == "local" and s.model == "llama3.1:8b" and s.api_key == ""
        assert s.leaves_network is False and s.batch_size == 4 and s.max_workers == 1
        from dataclasses import replace

        s = replace(s, cache_path=":memory:")
        rows = survey[survey["comment"] != ""].head(12)
        res, man = run_survey(rows, s, ColumnMap(text="comment", id="respondent_id", names=["manager_name"]),
                              tagger=Tagger(s, cache=TagCache(None)), save=False)
        assert (res["error"] == "").all()
        assert man["estimated_cost_usd"] == 0.0 and man["usage"]["prompt_tokens"] > 0
        first = seen[0]
        assert first["options"]["num_ctx"] >= 8192 and first["stream"] is False
        assert isinstance(first["format"], dict) and first["format"]["required"] == ["results"]
        assert first["_auth_header"] is None  # no key needed, none sent
    finally:
        server.shutdown()


def test_local_model_not_installed_says_how_to_fix(monkeypatch):
    import pytest

    from pulse.config import Settings
    from pulse.providers import APIError, ChatProvider

    server, _ = _serve_ollama(FakeModel())
    try:
        s = Settings(backend="local", base_url=f"http://127.0.0.1:{server.server_port}", model="muse-glimmer:30b")
        with pytest.raises(APIError) as err:
            ChatProvider(s, backoff_s=0).check()
        assert "ollama pull muse-glimmer:30b" in str(err.value)
    finally:
        server.shutdown()


def test_best_local_model_prefers_meta_and_newest():
    from pulse.local import LocalModel, best_model

    models = [LocalModel("qwen3:8b", 5, "8B", False), LocalModel("llama3.2:3b", 2, "3B", True),
              LocalModel("muse-glimmer:30b", 18, "30B", True)]
    assert best_model(models) == "muse-glimmer:30b"
    assert best_model(models[:2]) == "llama3.2:3b"
    assert best_model([]) is None
