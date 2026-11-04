"""Helpers for running Meta open-weights models on this computer with Ollama (no API key)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

DEFAULT_URL = "http://localhost:11434"

# Meta open-weights models that Pulse knows how to use, best first.
# (tag, plain description, rough download size, what hardware it needs)
RECOMMENDED: tuple[tuple[str, str, str, str], ...] = (
    ("muse-glimmer:30b", "Muse Glimmer 30B: Meta's newest open model", "about 18 GB",
     "NVIDIA GPU with 24 GB+ memory, or 32 GB+ RAM (slow on CPU)"),
    ("llama3.1:8b", "Llama 3.1 8B", "about 5 GB", "any laptop with 16 GB RAM; much faster with a GPU"),
    ("llama3.2:3b", "Llama 3.2 3B", "about 2 GB", "any laptop with 8 GB RAM; fastest, least accurate"),
)

_PREFERENCE = ("muse-glimmer", "muse", "llama4", "llama3.3", "llama3.1", "llama3.2", "llama3", "llama")


@dataclass(frozen=True)
class LocalModel:
    name: str
    size_gb: float
    parameters: str
    is_meta: bool


_cache: dict[str, tuple[float, list[LocalModel] | None]] = {}


def installed_models(base_url: str = DEFAULT_URL, timeout: float = 0.8, max_age_s: float = 5.0) -> list[LocalModel] | None:
    """Models installed in Ollama, or None if Ollama is not running. Cached for a few seconds."""
    hit = _cache.get(base_url)
    if hit and time.time() - hit[0] < max_age_s:
        return hit[1]
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=timeout)
        r.raise_for_status()
        models = []
        for m in r.json().get("models", []):
            name = str(m.get("name") or m.get("model") or "")
            details = m.get("details") or {}
            models.append(LocalModel(
                name=name,
                size_gb=round(float(m.get("size") or 0) / 1e9, 1),
                parameters=str(details.get("parameter_size") or ""),
                is_meta=any(p in name.lower() for p in ("muse", "llama")),
            ))
        result: list[LocalModel] | None = models
    except (requests.RequestException, ValueError):
        result = None
    _cache[base_url] = (time.time(), result)
    return result


def best_model(models: list[LocalModel] | None) -> str | None:
    """Pick the most capable installed Meta model, else any installed model."""
    if not models:
        return None
    for p in _PREFERENCE:
        for m in models:
            if m.name.lower().startswith(p):
                return m.name
    return models[0].name


def is_running(base_url: str = DEFAULT_URL) -> bool:
    return installed_models(base_url) is not None
