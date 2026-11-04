"""A stand-in for an OpenAI-compatible server, for tests only.

It answers tagging requests from the synthetic data's known labels (with optional
noise, so it behaves like an imperfect model) and brief requests with a plausible
brief. Nothing here is used by the app.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd

from pulse.synthetic import BY_ID

ROOT = Path(__file__).resolve().parent.parent


class FakeModel:
    def __init__(self, gold: pd.DataFrame | None = None, noise: float = 0.0, seed: int = 0,
                 reject: set[str] | None = None, garble_first: int = 0, drop_ids: set[str] | None = None,
                 status_sequence: list[int] | None = None):
        if gold is None:
            gold = pd.read_csv(ROOT / "data" / "synthetic_labels_full.csv", dtype=str, keep_default_na=False)
        self.gold = gold.set_index("comment_id")
        self.noise = noise
        self.rng = random.Random(seed)
        self.reject = set(reject or ())  # request fields to reject with HTTP 400, e.g. {"response_format"}
        self.garble_first = garble_first  # answer the first N requests with non-JSON text
        self.drop_ids = set(drop_ids or ())  # ids left out of batch answers
        self.status_sequence = list(status_sequence or [])  # e.g. [429, 200]
        self.requests: list[dict[str, Any]] = []

    # -- transport signature: (url, headers, body, timeout) -> (status, data, headers)
    def __call__(self, url: str, headers: dict[str, str], body: dict[str, Any], timeout: float):
        self.requests.append(body)
        if self.status_sequence:
            code = self.status_sequence.pop(0)
            if code != 200:
                return code, {"error": {"message": f"simulated {code}"}}, {"retry-after": "0"}
        for field in self.reject:
            if field in body:
                return 400, {"error": {"message": f"Unsupported parameter: {field}"}}, {}
        if self.garble_first > 0:
            self.garble_first -= 1
            return 200, self._wrap("Sure! Here are the tags you asked for."), {}
        system = body["messages"][0]["content"]
        user = body["messages"][-1]["content"]
        if "action briefs" in system:
            return 200, self._wrap(json.dumps(self._brief(user))), {}
        m = re.search(r"<comments>\n(.*)\n</comments>", user, re.DOTALL)
        items = json.loads(m.group(1)) if m else []
        results = [self._tag(it) for it in items if it["id"] not in self.drop_ids]
        return 200, self._wrap(json.dumps({"results": results}, ensure_ascii=False)), {}

    @staticmethod
    def _wrap(content: str) -> dict[str, Any]:
        return {"choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}

    def _tag(self, item: dict[str, str]) -> dict[str, Any]:
        cid, text = item["id"], item["text"]
        if cid in self.gold.index:
            g = self.gold.loc[cid]
            themes = [t for t in g["themes"].split(";") if t]
            sentiment, urgency, sensitive, lang = g["sentiment"], g["urgency"], g["sensitive_category"], g["language"]
            concepts = [c for c in g.get("concepts", "").split(";") if c in BY_ID]
            summary = " ".join(BY_ID[c].text["en"] for c in concepts)[:160] or "No substantive content."
        else:
            themes, sentiment, urgency, sensitive, lang, summary = ["other"], "neutral", "low", "none", "en", "Unknown."
        if self.noise and self.rng.random() < self.noise:
            sentiment = {"negative": "mixed", "mixed": "negative", "neutral": "positive", "positive": "neutral"}[sentiment]
        if self.noise and self.rng.random() < self.noise and len(themes) > 1:
            themes = themes[:-1]
        words = text.split()
        evidence = " ".join(words[: min(8, len(words))])
        return {"id": cid, "language": lang, "themes": themes, "sentiment": sentiment, "urgency": urgency,
                "sensitive_category": sensitive, "summary_en": summary, "evidence": evidence, "confidence": 0.9}

    @staticmethod
    def _brief(user: str) -> dict[str, Any]:
        payload = json.loads(user.split("\n", 1)[1])
        themes = payload["themes"]
        n = payload["comments_analysed"]
        top = themes[:3] or [{"theme": "other", "negative": 0, "label": "Other"}]
        return {
            "headline": f"{n} comments; the main concern is {top[0]['label'].lower()}.",
            "strengths": ["People value their colleagues."],
            "concerns": [{"theme": t["theme"], "point": f"{t['negative']} comments raise this."} for t in top[:2]],
            "actions": [{"theme": t["theme"], "action": f"Address {t['label'].lower()}.", "first_step": "Talk to the team."}
                        for t in (top * 3)[:3]],
            "conversation_starter": "What should we change first?",
        }
