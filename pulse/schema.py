"""Data shapes for model output and for tagged comments."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .taxonomy import SENSITIVE_CATEGORIES, SENTIMENTS, URGENCY_LEVELS, Theme

_SENTIMENT_ALIASES = {
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
    "very negative": "negative",
    "very positive": "positive",
    "ambivalent": "mixed",
}
_URGENCY_ALIASES = {"urgent": "high", "severe": "critical", "moderate": "medium", "none": "low"}


class ModelTag(BaseModel):
    """What the model must return for one comment."""

    id: str
    language: str = Field(description="ISO 639-1 code of the comment's language, e.g. en, ar, ur, hi")
    themes: list[str] = Field(min_length=1)
    sentiment: str
    urgency: str
    sensitive_category: str = "none"
    summary_en: str = Field(description="Neutral English paraphrase, max 25 words, no names")
    evidence: str = Field(description="Short verbatim span copied from the original comment")
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("id", mode="before")
    @classmethod
    def _id_str(cls, v: Any) -> str:
        return str(v).strip()

    @field_validator("themes", mode="before")
    @classmethod
    def _themes(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = [p for p in v.replace(";", ",").split(",")]
        items = [str(x).strip() for x in (v or []) if str(x).strip()]
        return items[:3] or ["other"]  # keep the three most important; never drop the whole answer

    @field_validator("language", mode="before")
    @classmethod
    def _lang(cls, v: Any) -> str:
        v = str(v or "und").strip().lower().replace("_", "-")
        return v.split("-")[0][:3] or "und"

    @field_validator("sentiment", mode="before")
    @classmethod
    def _sentiment(cls, v: Any) -> str:
        v = str(v).strip().lower()
        v = _SENTIMENT_ALIASES.get(v, v)
        if v not in SENTIMENTS:
            raise ValueError(f"sentiment must be one of {SENTIMENTS}")
        return v

    @field_validator("urgency", mode="before")
    @classmethod
    def _urgency(cls, v: Any) -> str:
        v = str(v).strip().lower()
        v = _URGENCY_ALIASES.get(v, v)
        if v not in URGENCY_LEVELS:
            raise ValueError(f"urgency must be one of {URGENCY_LEVELS}")
        return v

    @field_validator("sensitive_category", mode="before")
    @classmethod
    def _sensitive(cls, v: Any) -> str:
        v = str(v or "none").strip().lower().replace(" ", "_").replace("-", "_")
        if v in ("", "null", "no", "false", "n/a"):
            v = "none"
        if v not in SENSITIVE_CATEGORIES:
            raise ValueError(f"sensitive_category must be one of {SENSITIVE_CATEGORIES}")
        return v

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, v: Any) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        if v > 1.0 and v <= 100.0:  # some models answer in percent
            v = v / 100.0
        return max(0.0, min(1.0, v))

    @field_validator("summary_en", "evidence", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return str(v or "").strip()

    def normalise_themes(self, themes: tuple[Theme, ...]) -> list[str]:
        """Map labels or ids (any case) onto taxonomy ids; drop unknowns; keep order."""
        by_key: dict[str, str] = {}
        for t in themes:
            by_key[t.id.lower()] = t.id
            by_key[t.label.lower()] = t.id
        out: list[str] = []
        for raw in self.themes:
            key = str(raw).strip().lower()
            tid = by_key.get(key) or by_key.get(key.replace(" ", "_"))
            if tid and tid not in out:
                out.append(tid)
        return out[:3] or ["other"]


class TaggedComment(BaseModel):
    """One analysed comment, as stored and shown in the dashboard."""

    comment_id: str
    text: str
    text_redacted: str
    language: str = "und"
    themes: list[str] = Field(default_factory=lambda: ["other"])
    sentiment: str = "neutral"
    urgency: str = "low"
    sensitive_category: str = "none"
    summary_en: str = ""
    evidence: str = ""
    evidence_grounded: bool = False
    confidence: float = 0.0
    keyword_flags: list[str] = Field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    rechecked: bool = False
    skipped: str = ""  # e.g. "no_content" when no model call was needed
    error: str = ""
    backend: str = ""
    model: str = ""
    prompt_version: str = ""

    @property
    def sensitive(self) -> bool:
        return self.sensitive_category != "none"


def tag_json_schema(themes: tuple[Theme, ...], batch: bool = True) -> dict[str, Any]:
    """JSON schema handed to the API as a structured-output constraint."""
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "language",
            "themes",
            "sentiment",
            "urgency",
            "sensitive_category",
            "summary_en",
            "evidence",
            "confidence",
        ],
        "properties": {
            "id": {"type": "string"},
            "language": {"type": "string"},
            "themes": {
                "type": "array",
                "items": {"type": "string", "enum": [t.id for t in themes]},
                "minItems": 1,
                "maxItems": 3,
            },
            "sentiment": {"type": "string", "enum": list(SENTIMENTS)},
            "urgency": {"type": "string", "enum": list(URGENCY_LEVELS)},
            "sensitive_category": {"type": "string", "enum": list(SENSITIVE_CATEGORIES)},
            "summary_en": {"type": "string"},
            "evidence": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }
    if not batch:
        return item
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {"results": {"type": "array", "items": item}},
    }
