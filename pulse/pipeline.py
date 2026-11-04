"""End-to-end run: survey CSV in, tagged comments + run manifest out."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from .config import PRICES_PER_M, Settings
from .redact import name_terms, redact
from .safety import is_confidential
from .tagger import CommentIn, ProgressFn, Tagger
from .taxonomy import DEFAULT_THEMES, Theme

RUNS_DIR = Path("outputs/runs")

ID_CANDIDATES = ("comment_id", "respondent_id", "response_id", "id")
TEXT_CANDIDATES = ("comment", "comments", "text", "open_text", "response", "verbatim")
NAME_CANDIDATES = ("manager_name", "supervisor_name", "manager", "supervisor")


def is_text(series: pd.Series) -> bool:
    """True for text columns under both pandas 2 (object dtype) and pandas 3 (string dtype)."""
    return series.dtype == object or pd.api.types.is_string_dtype(series)


@dataclass
class ColumnMap:
    text: str
    id: str | None = None
    names: list[str] = field(default_factory=list)  # columns holding person names to redact

    @classmethod
    def guess(cls, df: pd.DataFrame) -> "ColumnMap":
        cols = {c.lower(): c for c in df.columns}
        text = next((cols[c] for c in TEXT_CANDIDATES if c in cols), None)
        if text is None:  # longest average string column
            obj = [c for c in df.columns if is_text(df[c])]
            if not obj:
                raise ValueError("No text column found in the file.")
            text = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
        cid = next((cols[c] for c in ID_CANDIDATES if c in cols), None)
        names = [cols[c] for c in NAME_CANDIDATES if c in cols and is_text(df[cols[c]])]
        return cls(text=text, id=cid, names=names)


def load_survey(path_or_buffer: Any) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df


def prepare(df: pd.DataFrame, cols: ColumnMap, extra_terms: tuple[str, ...] = ()) -> tuple[pd.DataFrame, list[CommentIn]]:
    """Drop blank comments, assign ids, redact personal details."""
    work = df.copy()
    work[cols.text] = work[cols.text].fillna("").astype(str)
    work = work[work[cols.text].str.strip() != ""].copy()
    if cols.id and cols.id in work.columns:
        work["comment_id"] = work[cols.id].astype(str)
    else:
        work["comment_id"] = [f"C{i:05d}" for i in range(1, len(work) + 1)]
    if work["comment_id"].duplicated().any():
        raise ValueError("Comment ids must be unique.")
    terms = name_terms(*(work[c].astype(str) for c in cols.names if c in work.columns)) + list(extra_terms)
    items: list[CommentIn] = []
    redacted_col, kinds_col = [], []
    for cid, text in zip(work["comment_id"], work[cols.text]):
        red, kinds = redact(text, terms)
        items.append(CommentIn(cid, text, red))
        redacted_col.append(red)
        kinds_col.append(",".join(kinds))
    work["redactions"] = kinds_col
    return work, items


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = PRICES_PER_M.get(model)
    if not price:
        return None
    return round(prompt_tokens / 1e6 * price[0] + completion_tokens / 1e6 * price[1], 4)


def run_survey(
    df: pd.DataFrame,
    settings: Settings,
    cols: ColumnMap | None = None,
    themes: tuple[Theme, ...] = DEFAULT_THEMES,
    tagger: Tagger | None = None,
    progress: ProgressFn | None = None,
    source_name: str = "upload.csv",
    save: bool = True,
    runs_dir: Path = RUNS_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tag every comment. Returns (results DataFrame, manifest)."""
    cols = cols or ColumnMap.guess(df)
    started = time.time()
    meta, items = prepare(df, cols, settings.redact_terms)
    tagger = tagger or Tagger(settings, themes)
    tags = tagger.tag(items, progress=progress)

    rows = []
    for t in tags:
        d = t.model_dump()
        d["primary_theme"] = t.themes[0] if t.themes else "other"
        d["confidential"] = is_confidential(t)
        rows.append(d)
    tagged = pd.DataFrame(rows)
    keep_meta = [c for c in meta.columns if c not in tagged.columns and c != cols.text]
    results = meta[["comment_id", *keep_meta]].merge(tagged, on="comment_id", how="right")

    usage = tagger.provider.usage.as_dict() if tagger.provider else {}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_name,
        "source_sha256": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16],
        "backend": settings.backend,
        "model": settings.model,
        "endpoint_host": urlparse(settings.base_url).hostname if settings.base_url else None,
        "data_leaves_network": settings.leaves_network,
        "prompt_version": tags[0].prompt_version if tags else "",
        "reasoning_effort": settings.reasoning_effort,
        "recheck": settings.recheck and settings.uses_model,
        "recheck_reasoning_effort": settings.recheck_reasoning_effort,
        "batch_size": settings.batch_size,
        "columns": {"text": cols.text, "id": cols.id, "names_redacted_from": cols.names},
        "n_comments": len(results),
        "n_redacted": int((meta["redactions"] != "").sum()),
        "stats": dict(tagger.stats),
        "usage": usage,
        "estimated_cost_usd": 0.0 if settings.backend in ("local", "offline") or not usage
        else estimate_cost(settings.model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
        "provider_notes": list(tagger.provider.notes) if tagger.provider else [],
        "structured_output_mode": tagger.provider.format_mode if tagger.provider else None,
        "duration_s": round(time.time() - started, 1),
        "n_needs_review": int(results["needs_review"].sum()) if len(results) else 0,
        "n_confidential": int(results["confidential"].sum()) if len(results) else 0,
    }
    if save:
        save_run(results, manifest, runs_dir)
    return results, manifest


def save_run(results: pd.DataFrame, manifest: dict[str, Any], runs_dir: Path = RUNS_DIR) -> Path:
    out = Path(runs_dir) / manifest["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    results.to_json(out / "tags.jsonl", orient="records", lines=True, force_ascii=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def list_runs(runs_dir: Path = RUNS_DIR) -> list[dict[str, Any]]:
    runs = []
    for m in sorted(Path(runs_dir).glob("*/manifest.json"), reverse=True):
        try:
            runs.append(json.loads(m.read_text(encoding="utf-8")))
        except Exception:
            continue
    return runs


def load_run(run_id: str, runs_dir: Path = RUNS_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    folder = Path(runs_dir) / run_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    results = pd.read_json(folder / "tags.jsonl", orient="records", lines=True, dtype={"comment_id": str})
    for col in ("themes", "keyword_flags", "review_reasons"):
        if col in results.columns:
            results[col] = results[col].apply(lambda v: list(v) if isinstance(v, (list, tuple)) else [])
    return results, manifest
