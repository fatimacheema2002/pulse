"""One client for Meta Model API, self-hosted servers and local Ollama.

- Meta Model API:   https://api.meta.ai/v1, model ``muse-spark-1.3`` (OpenAI-compatible)
- Open weights:     any vLLM / SGLang / Baseten / Fireworks endpoint serving
                    ``meta-models/Muse-Spark-Open-816B-A42B`` or ``meta-models/Muse-Glimmer-30B``
- Local (Ollama):   http://localhost:11434, e.g. ``muse-glimmer:30b`` or ``llama3.1:8b``. No API key.
                    Uses Ollama's native /api/chat so the context window can be raised
                    (Ollama's 4,096-token default silently truncates long prompts) and the
                    JSON schema is enforced by constrained decoding.

Servers differ in which optional parameters they accept (structured output,
reasoning effort, temperature). The client starts with the strictest request,
steps down on a 400 error, and remembers what worked.

It talks HTTP directly (``requests``) so it does not depend on any SDK version.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from .config import Settings

FORMAT_MODES = ("json_schema", "json_object", "none")
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class ModelOutputError(ValueError):
    """The model answered, but not with usable JSON."""


class APIError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class BadRequestError(APIError):
    pass


class AuthError(APIError):
    pass


# (url, headers, json_body, timeout) -> (status_code, parsed_json_or_text, headers)
Transport = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, Any, dict[str, str]]]


def requests_transport(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> tuple[int, Any, dict[str, str]]:
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    try:
        data: Any = r.json()
    except ValueError:
        data = r.text
    return r.status_code, data, dict(r.headers)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, prompt: int, completion: int, seconds: float) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.seconds += seconds

    def error(self) -> None:
        with self._lock:
            self.errors += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "errors": self.errors,
            "seconds": round(self.seconds, 1),
        }


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerates think tags and code fences)."""
    if not text or not text.strip():
        raise ModelOutputError("empty reply")
    t = _THINK.sub("", text).strip()
    t = _FENCE.sub("", t).strip()
    try:
        value = json.loads(t)
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"results": value}
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(t[start : i + 1])
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        pass
                    break
        start = t.find("{", start + 1)
    raise ModelOutputError("no JSON object found in reply")


def _error_message(data: Any) -> str:
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message") or err)
        return str(err)
    return str(data)[:500]


class ChatProvider:
    """Thread-safe JSON-returning chat client with capability fallback and retries."""

    def __init__(self, settings: Settings, transport: Transport | None = None, max_retries: int = 4,
                 backoff_s: float = 1.5):
        if not settings.uses_model:
            raise ValueError("ChatProvider needs a model backend (meta or open_weights).")
        if not settings.is_configured:
            raise ValueError(
                "Model backend is not configured. Set MODEL_API_KEY for Meta Model API, "
                "or PULSE_OPEN_BASE_URL and PULSE_OPEN_MODEL for a self-hosted model."
            )
        self.settings = settings
        self.model = settings.model
        self.ollama = settings.backend == "local"
        self.url = settings.base_url.rstrip("/") + ("/api/chat" if self.ollama else "/chat/completions")
        self.transport = transport or requests_transport
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.usage = Usage()
        self._lock = threading.Lock()
        self._format_mode = 0  # index into FORMAT_MODES
        self._send_reasoning = True
        self._send_temperature = True
        self.notes: list[str] = []

    # -- capability fallback -------------------------------------------------
    def _step_down(self, message: str) -> bool:
        """Drop the optional parameter the server objected to. Returns False if nothing is left to drop."""
        msg = message.lower()
        with self._lock:
            if "reasoning" in msg and self._send_reasoning:
                self._send_reasoning = False
                self.notes.append("Server rejected reasoning_effort; continuing without it.")
                return True
            if "temperature" in msg and self._send_temperature:
                self._send_temperature = False
                self.notes.append("Server rejected temperature; continuing without it.")
                return True
            if self._format_mode < len(FORMAT_MODES) - 1:
                old = FORMAT_MODES[self._format_mode]
                self._format_mode += 1
                self.notes.append(f"Server rejected response_format={old}; using {FORMAT_MODES[self._format_mode]}.")
                return True
            if self._send_reasoning:
                self._send_reasoning = False
                return True
            if self._send_temperature:
                self._send_temperature = False
                return True
        return False

    @property
    def format_mode(self) -> str:
        return FORMAT_MODES[self._format_mode]

    def _body(self, messages: list[dict[str, str]], schema: dict[str, Any] | None, reasoning_effort: str) -> dict[str, Any]:
        mode = FORMAT_MODES[self._format_mode]
        if self.ollama:
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_ctx": self.settings.num_ctx},
            }
            if mode == "json_schema" and schema is not None:
                body["format"] = schema
            elif mode in ("json_schema", "json_object"):
                body["format"] = "json"
            return body
        body = {"model": self.model, "messages": messages}
        if mode == "json_schema" and schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "pulse_output", "schema": schema, "strict": True},
            }
        elif mode in ("json_schema", "json_object"):
            body["response_format"] = {"type": "json_object"}
        if self._send_temperature:
            body["temperature"] = 0
        if reasoning_effort and self._send_reasoning:
            body["reasoning_effort"] = reasoning_effort
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key or not self.ollama:
            headers["Authorization"] = f"Bearer {self.settings.api_key or 'none'}"
        attempt = 0
        while True:
            attempt += 1
            try:
                status, data, resp_headers = self.transport(self.url, headers, body, self.settings.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as err:
                self.usage.error()
                if attempt > (1 if self.ollama else self.max_retries):  # a local server that is down won't come back
                    hint = " Is the Ollama app running?" if self.ollama else ""
                    raise APIError(0, f"could not reach {self.url}.{hint} ({err})") from err
                time.sleep(self.backoff_s * attempt)
                continue
            if 200 <= status < 300 and isinstance(data, dict):
                return data
            self.usage.error()
            message = _error_message(data)
            if status in (401, 403):
                raise AuthError(status, message)
            if status in (400, 422) or (self.ollama and any(w in message.lower() for w in ("format", "schema", "grammar"))):
                raise BadRequestError(status, message)
            if self.ollama and status == 404:
                raise APIError(404, f"{message}. Install it with:  ollama pull {self.model}")
            if status in RETRY_STATUS and attempt <= self.max_retries:
                retry_after = (resp_headers or {}).get("retry-after") or (resp_headers or {}).get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else self.backoff_s * (2 ** (attempt - 1))
                except ValueError:
                    wait = self.backoff_s * (2 ** (attempt - 1))
                time.sleep(min(wait, 30) + random.random() * 0.25)
                continue
            raise APIError(status, message)

    # -- main call -----------------------------------------------------------
    def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        reasoning_effort: str = "",
    ) -> tuple[dict[str, Any], str]:
        """Send messages and return (parsed_json, raw_text)."""
        for _ in range(8):
            t0 = time.time()
            try:
                data = self._post(self._body(messages, schema, reasoning_effort))
            except BadRequestError as err:
                if self._step_down(err.message):
                    continue
                raise
            if self.ollama:
                self.usage.add(int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0), time.time() - t0)
                text = (data.get("message") or {}).get("content") or ""
                return extract_json(text), text
            u = data.get("usage") or {}
            self.usage.add(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0), time.time() - t0)
            choices = data.get("choices") or []
            msg = (choices[0].get("message") or {}) if choices else {}
            text = msg.get("content") or ""
            if isinstance(text, list):  # some servers return content parts
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            return extract_json(text), text
        raise BadRequestError(400, "request rejected after removing all optional parameters")

    def check(self) -> str:
        """Tiny request to confirm the key, URL and model work."""
        data, _ = self.complete_json(
            [
                {"role": "system", "content": 'Reply only with JSON like {"ok": true, "model_says": "<three words>"}.'},
                {"role": "user", "content": "مرحبا! Say hello in three words."},
            ],
        )
        return json.dumps(data, ensure_ascii=False)
