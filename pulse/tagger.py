"""Comment tagging with Muse Spark: batching, validation, caching and a second-look pass."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable

from pydantic import ValidationError

from .baseline import tag_with_keywords
from .cache import TagCache
from .config import Settings
from .lang import guess_language
from .prompts import (
    RECHECK_NOTE,
    TAG_PROMPT_VERSION,
    repair_prompt,
    tagging_system_prompt,
    tagging_user_prompt,
)
from .providers import ChatProvider, ModelOutputError
from .safety import keyword_flags, normalise, route
from .schema import ModelTag, TaggedComment, tag_json_schema
from .taxonomy import DEFAULT_THEMES, Theme

ProgressFn = Callable[[int, int, str], None]

_NO_CONTENT = {
    normalise(s)
    for s in [
        "no comments", "no comment", "none", "nil", "n/a", "na", "nothing", "nothing to add", "no", "-", ".",
        "لا يوجد", "لا شيء", "لا تعليق", "لايوجد",
        "کوئی تبصرہ نہیں", "کچھ نہیں", "کوئی نہیں",
        "कोई टिप्पणी नहीं", "कुछ नहीं", "कोई नहीं",
        "koi comment nahi", "koi comment nahin", "kuch nahi", "kuch nahin", "koi nahi",
    ]
}


def _eta(started: float, done: int, total: int) -> str:
    if done <= 0 or done >= total:
        return ""
    left = (time.time() - started) / done * (total - done)
    if left < 90:
        return f" · about {max(5, int(round(left, -1)))} s left"
    return f" · about {int(round(left / 60))} min left"


def is_no_content(text: str) -> bool:
    t = normalise(text).strip(" .!-—–۔।")
    return len(t) < 2 or t in _NO_CONTENT


@dataclass(frozen=True)
class CommentIn:
    comment_id: str
    text: str
    redacted: str


def _clean_span(s: str) -> str:
    s = normalise(s).strip(" \"'“”‘’«»….،,؛;:")
    return re.sub(r"\.{3}|…", " ", s).strip()


def evidence_is_grounded(evidence: str, *texts: str) -> bool:
    ev = _clean_span(evidence)
    if len(ev) < 3:
        return False
    for t in texts:
        if ev in normalise(t):
            return True
    return False


class Tagger:
    def __init__(
        self,
        settings: Settings,
        themes: tuple[Theme, ...] = DEFAULT_THEMES,
        provider: ChatProvider | None = None,
        cache: TagCache | None = None,
    ):
        self.settings = settings
        self.themes = themes
        self.provider = provider
        if self.provider is None and settings.uses_model:
            self.provider = ChatProvider(settings)
        self.cache = cache if cache is not None else TagCache(settings.cache_path)
        self.system = tagging_system_prompt(themes)
        self.schema = tag_json_schema(themes, batch=True)
        tax = "|".join(f"{t.id}:{t.description}" for t in themes)
        self.taxonomy_hash = hashlib.sha256(tax.encode()).hexdigest()[:12]
        self._lock = threading.Lock()
        self.stats = {"total": 0, "cached": 0, "model": 0, "deduplicated": 0, "skipped": 0, "baseline": 0,
                      "rechecked": 0, "recheck_changed": 0, "failed": 0}

    # -- helpers ---------------------------------------------------------------
    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.stats[key] += n

    def _key(self, redacted: str, stage: str) -> str:
        effort = self.settings.reasoning_effort if stage == "first" else self.settings.recheck_reasoning_effort
        return TagCache.key(self.settings.backend, self.settings.model, TAG_PROMPT_VERSION,
                            self.taxonomy_hash, stage, effort, redacted)

    def _record(self, c: CommentIn, tag: ModelTag, flags: list[str]) -> TaggedComment:
        return TaggedComment(
            comment_id=c.comment_id,
            text=c.text,
            text_redacted=c.redacted,
            language=tag.language or guess_language(c.text),
            themes=tag.normalise_themes(self.themes),
            sentiment=tag.sentiment,
            urgency=tag.urgency,
            sensitive_category=tag.sensitive_category,
            summary_en=tag.summary_en,
            evidence=tag.evidence,
            evidence_grounded=evidence_is_grounded(tag.evidence, c.redacted, c.text),
            confidence=tag.confidence,
            keyword_flags=flags,
            backend=self.settings.backend,
            model=self.settings.model,
            prompt_version=TAG_PROMPT_VERSION,
        )

    def _error_record(self, c: CommentIn, flags: list[str], err: str) -> TaggedComment:
        self._bump("failed")
        return TaggedComment(
            comment_id=c.comment_id, text=c.text, text_redacted=c.redacted, language=guess_language(c.text),
            keyword_flags=flags, error=err[:300], backend=self.settings.backend, model=self.settings.model,
            prompt_version=TAG_PROMPT_VERSION, sensitive_category=flags[0] if flags else "none",
            urgency="high" if flags else "low",
        )

    # -- model calls -------------------------------------------------------------
    def _call(self, items: list[CommentIn], system: str, effort: str) -> dict[str, ModelTag | str]:
        """One request for a batch. Returns id -> ModelTag, or id -> error message."""
        assert self.provider is not None
        payload = [{"id": c.comment_id, "text": c.redacted} for c in items]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": tagging_user_prompt(payload)}]
        try:
            data, raw = self.provider.complete_json(messages, schema=self.schema, reasoning_effort=effort)
        except ModelOutputError as err:
            messages += [{"role": "assistant", "content": "(unusable answer)"},
                         {"role": "user", "content": repair_prompt(str(err))}]
            try:
                data, raw = self.provider.complete_json(messages, schema=self.schema, reasoning_effort=effort)
            except ModelOutputError as err2:
                return {c.comment_id: f"unparseable model output: {err2}" for c in items}
        except Exception as err:  # network, auth, rate limit after retries
            return {c.comment_id: f"{type(err).__name__}: {err}" for c in items}

        results = data.get("results", data if isinstance(data, list) else [])
        if isinstance(results, dict):
            results = [results]
        wanted = {c.comment_id for c in items}
        out: dict[str, ModelTag | str] = {}
        for i, obj in enumerate(results if isinstance(results, list) else []):
            if not isinstance(obj, dict):
                continue
            if len(items) == 1 and str(obj.get("id", "")).strip() not in wanted:
                obj = {**obj, "id": items[0].comment_id}  # single item: trust position
            try:
                tag = ModelTag.model_validate(obj)
            except ValidationError as err:
                cid = str(obj.get("id", "")).strip()
                if cid in wanted:
                    out[cid] = f"invalid fields: {err.errors()[0].get('msg', 'validation error')}"
                continue
            if tag.id in wanted:
                out[tag.id] = tag
        for c in items:
            out.setdefault(c.comment_id, "missing from model output")
        return out

    def _tag_batch(self, items: list[CommentIn]) -> dict[str, ModelTag | str]:
        out = self._call(items, self.system, self.settings.reasoning_effort)
        retry = [c for c in items if not isinstance(out[c.comment_id], ModelTag)]
        if retry and len(items) > 1:  # fall back to one comment per request
            for c in retry:
                out.update(self._call([c], self.system, self.settings.reasoning_effort))
        return out

    # -- public API --------------------------------------------------------------
    def tag(self, comments: Iterable[CommentIn], progress: ProgressFn | None = None) -> list[TaggedComment]:
        comments = list(comments)
        total = len(comments)
        self.stats["total"] = total
        records: dict[str, TaggedComment] = {}
        flags_by_id: dict[str, list[str]] = {}
        todo: list[CommentIn] = []
        done = 0

        def tick(n: int, msg: str) -> None:
            nonlocal done
            done += n
            if progress:
                progress(min(done, total), total, msg)

        for c in comments:
            flags = keyword_flags(c.text)
            flags_by_id[c.comment_id] = flags
            if is_no_content(c.text):
                records[c.comment_id] = TaggedComment(
                    comment_id=c.comment_id, text=c.text, text_redacted=c.redacted, language=guess_language(c.text),
                    themes=["other"], sentiment="neutral", urgency="low", confidence=1.0, skipped="no_content",
                    keyword_flags=flags, backend=self.settings.backend, model=self.settings.model,
                    prompt_version=TAG_PROMPT_VERSION,
                )
                self._bump("skipped")
                continue
            if not self.settings.uses_model:
                rec = tag_with_keywords(c.comment_id, c.text, c.redacted, self.themes)
                rec.backend, rec.model, rec.prompt_version = "offline", "keyword-baseline", "keywords-v1"
                records[c.comment_id] = rec
                self._bump("baseline")
                continue
            hit = self.cache.get(self._key(c.redacted, "first"))
            if hit:
                records[c.comment_id] = self._record(c, ModelTag.model_validate({**hit, "id": c.comment_id}), flags)
                self._bump("cached")
                continue
            todo.append(c)
        tick(total - len(todo), "Loaded cached and empty comments")

        if todo:
            # Identical (redacted) comments are sent once and the answer is shared.
            groups: dict[str, list[CommentIn]] = {}
            for c in todo:
                groups.setdefault(c.redacted, []).append(c)
            unique = [members[0] for members in groups.values()]
            self.stats["deduplicated"] = len(todo) - len(unique)
            size = self.settings.batch_size
            batches = [unique[i : i + size] for i in range(0, len(unique), size)]
            started, unique_done = time.time(), 0
            with ThreadPoolExecutor(max_workers=self.settings.max_workers) as pool:
                futures = {pool.submit(self._tag_batch, b): b for b in batches}
                for fut in as_completed(futures):
                    batch = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as err:  # pragma: no cover - defensive
                        result = {c.comment_id: f"{type(err).__name__}: {err}" for c in batch}
                    n_done = 0
                    for rep in batch:
                        value = result.get(rep.comment_id, "missing")
                        if isinstance(value, ModelTag):
                            self.cache.put(self._key(rep.redacted, "first"), value.model_dump())
                        for c in groups[rep.redacted]:
                            n_done += 1
                            if isinstance(value, ModelTag):
                                tag = value.model_copy(update={"id": c.comment_id})
                                records[c.comment_id] = self._record(c, tag, flags_by_id[c.comment_id])
                                self._bump("model")
                            else:
                                records[c.comment_id] = self._error_record(c, flags_by_id[c.comment_id], str(value))
                    unique_done += len(batch)
                    tick(n_done, f"Tagged {min(done + n_done, total)} of {total}"
                         + _eta(started, unique_done, len(unique)))

        if self.settings.uses_model and self.settings.recheck:
            self._recheck(comments, records, progress)

        ordered = [records[c.comment_id] for c in comments]
        for rec in ordered:
            route(rec)
        return ordered

    # -- second look ---------------------------------------------------------------
    @staticmethod
    def needs_second_look(rec: TaggedComment) -> bool:
        if rec.skipped or rec.error or rec.rechecked:
            return False
        return (
            rec.sensitive_category != "none"
            or bool(rec.keyword_flags)
            or rec.urgency == "critical"
            or rec.confidence < 0.6
            or (bool(rec.evidence) and not rec.evidence_grounded)
        )

    def _recheck(self, comments: list[CommentIn], records: dict[str, TaggedComment], progress: ProgressFn | None) -> None:
        by_id = {c.comment_id: c for c in comments}
        flagged = [by_id[cid] for cid, rec in records.items() if self.needs_second_look(rec)]
        if not flagged:
            return
        same_text: dict[str, list[CommentIn]] = {}
        for c in flagged:
            same_text.setdefault(c.redacted, []).append(c)
        targets = [members[0] for members in same_text.values()]
        system = self.system + RECHECK_NOTE
        effort = self.settings.recheck_reasoning_effort

        def one(c: CommentIn) -> tuple[CommentIn, ModelTag | str]:
            key = self._key(c.redacted, "recheck")
            hit = self.cache.get(key)
            if hit:
                return c, ModelTag.model_validate({**hit, "id": c.comment_id})
            value = self._call([c], system, effort)[c.comment_id]
            if isinstance(value, ModelTag):
                self.cache.put(key, value.model_dump())
            return c, value

        finished = 0
        with ThreadPoolExecutor(max_workers=self.settings.max_workers) as pool:
            for fut in as_completed([pool.submit(one, c) for c in targets]):
                c, value = fut.result()
                finished += 1
                if progress:
                    progress(finished, len(targets), f"Second look at flagged comments: {finished} of {len(targets)}")
                if not isinstance(value, ModelTag):
                    continue  # keep the first-pass answer
                for member in same_text[c.redacted]:
                    old = records[member.comment_id]
                    new = self._record(member, value.model_copy(update={"id": member.comment_id}), old.keyword_flags)
                    new.rechecked = True
                    if (new.themes, new.sentiment, new.urgency, new.sensitive_category) != (
                        old.themes, old.sentiment, old.urgency, old.sensitive_category
                    ):
                        self._bump("recheck_changed")
                    self._bump("rechecked")
                    records[member.comment_id] = new
