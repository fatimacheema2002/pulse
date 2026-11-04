"""Runtime settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

try:  # optional dependency; the app still works with plain environment variables
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover
    pass

BACKENDS = ("local", "meta", "open_weights", "offline")

BACKEND_LABELS = {
    "local": "On this computer, no API key (Ollama: Muse Glimmer / Llama)",
    "meta": "Meta Model API (Muse Spark)",
    "open_weights": "Open weights on a server (vLLM, SGLang or a hosted provider)",
    "offline": "Keyword baseline (no model)",
}

META_BASE_URL = "https://api.meta.ai/v1"
META_DEFAULT_MODEL = "muse-spark-1.3"
OPEN_DEFAULT_BASE_URL = "http://localhost:8000/v1"
OPEN_DEFAULT_MODEL = "meta-models/Muse-Glimmer-30B"
LOCAL_DEFAULT_BASE_URL = "http://localhost:11434"
LOCAL_FALLBACK_MODEL = "llama3.1:8b"

# USD per 1M tokens (input, output). Used only for the cost estimate shown in the run log.
PRICES_PER_M: dict[str, tuple[float, float]] = {
    "muse-spark-1.3": (1.25, 4.25),
    "muse-spark-1.2": (1.25, 4.25),
    "muse-spark-1.1": (1.25, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.20),
}


def is_private_host(url: str) -> bool:
    """True if the URL points at this machine or a private network (data stays in-house)."""
    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    if host in ("localhost",) or host.endswith((".local", ".internal", ".lan", ".corp")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    backend: str = "offline"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    reasoning_effort: str = "low"          # first pass
    recheck_reasoning_effort: str = "high"  # second pass on doubtful items
    recheck: bool = True
    batch_size: int = 8
    max_workers: int = 4
    timeout_s: float = 120.0
    num_ctx: int = 8192  # context window requested from Ollama (its default of 4096 silently truncates)
    min_group_size: int = 5
    cache_path: str = "outputs/cache.sqlite"
    redact_terms: tuple[str, ...] = field(default_factory=tuple)

    @property
    def uses_model(self) -> bool:
        return self.backend in ("meta", "open_weights", "local")

    @property
    def leaves_network(self) -> bool:
        """True when comment text is sent to a third-party cloud API."""
        if self.backend == "meta":
            return True
        if self.backend in ("open_weights", "local"):
            return not is_private_host(self.base_url)
        return False

    @property
    def is_configured(self) -> bool:
        if self.backend == "offline":
            return True
        if self.backend == "meta":
            return bool(self.api_key)
        return bool(self.base_url and self.model)

    def with_backend(self, backend: str) -> "Settings":
        """Return settings for another backend, filling that backend's defaults from the environment."""
        return from_env(backend=backend, base=self)


def from_env(backend: str | None = None, base: Settings | None = None) -> Settings:
    backend = (backend or _env("PULSE_BACKEND", "")).lower()
    if not backend:
        if _env("MODEL_API_KEY") or _env("META_API_KEY"):
            backend = "meta"
        else:
            from .local import is_running

            backend = "local" if is_running(_env("PULSE_LOCAL_BASE_URL", LOCAL_DEFAULT_BASE_URL)) else "offline"
    if backend not in BACKENDS:
        raise ValueError(f"PULSE_BACKEND must be one of {BACKENDS}, got {backend!r}")
    batch_default, workers_default, timeout_default = 8, 4, 120.0

    if backend == "meta":
        base_url = _env("META_BASE_URL", META_BASE_URL)
        api_key = _env("MODEL_API_KEY") or _env("META_API_KEY")
        model = _env("PULSE_META_MODEL", META_DEFAULT_MODEL)
        effort = _env("PULSE_REASONING_EFFORT", "low")
    elif backend == "open_weights":
        base_url = _env("PULSE_OPEN_BASE_URL", OPEN_DEFAULT_BASE_URL)
        api_key = _env("PULSE_OPEN_API_KEY", "not-needed")
        model = _env("PULSE_OPEN_MODEL", OPEN_DEFAULT_MODEL)
        effort = _env("PULSE_REASONING_EFFORT", "")  # many self-hosted servers ignore or reject it
    elif backend == "local":
        from .local import best_model, installed_models

        base_url = _env("PULSE_LOCAL_BASE_URL", LOCAL_DEFAULT_BASE_URL)
        api_key = ""
        model = _env("PULSE_LOCAL_MODEL") or best_model(installed_models(base_url)) or LOCAL_FALLBACK_MODEL
        effort = ""
        # One request at a time with small batches: what a laptop model handles best.
        batch_default, workers_default, timeout_default = 4, 1, 900.0
    else:
        base_url, api_key, model, effort = "", "", "keyword-baseline", ""

    redact = tuple(t.strip() for t in _env("PULSE_REDACT_TERMS", "").split(",") if t.strip())
    settings = Settings(
        backend=backend,
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort=effort,
        recheck_reasoning_effort=_env("PULSE_RECHECK_REASONING_EFFORT", "high" if backend == "meta" else ""),
        recheck=_env("PULSE_RECHECK", "1") not in ("0", "false", "no"),
        batch_size=max(1, _env_int("PULSE_BATCH_SIZE", batch_default)),
        max_workers=max(1, _env_int("PULSE_MAX_WORKERS", workers_default)),
        timeout_s=_env_float("PULSE_TIMEOUT_S", timeout_default),
        num_ctx=max(2048, _env_int("PULSE_NUM_CTX", 8192)),
        min_group_size=max(1, _env_int("PULSE_MIN_GROUP_SIZE", 5)),
        cache_path=_env("PULSE_CACHE_PATH", "outputs/cache.sqlite"),
        redact_terms=redact,
    )
    if base is not None:
        settings = replace(
            settings,
            min_group_size=base.min_group_size,
            cache_path=base.cache_path,
            redact_terms=base.redact_terms,
        )
    return settings
