"""Label sets used everywhere in Pulse.

The theme taxonomy can be replaced with a JSON file (see ``load_taxonomy``) so an
HR team can use its own engagement model without touching code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Theme:
    id: str
    label: str
    description: str


DEFAULT_THEMES: tuple[Theme, ...] = (
    Theme("workload", "Workload & staffing", "Volume of work, staffing levels, pace, covering for vacancies."),
    Theme("pay_benefits", "Pay & benefits", "Salary, increments, overtime pay, allowances, insurance, payment delays."),
    Theme("manager_support", "Direct manager", "Behaviour, support, fairness and clarity of the person's own supervisor or manager."),
    Theme("leadership", "Leadership & direction", "Senior management, strategy, trust in company direction, visibility of leaders."),
    Theme("career_growth", "Career growth", "Promotion, grades, career paths, stretch opportunities."),
    Theme("training", "Training & development", "Technical or soft-skill training, courses, certification, language classes."),
    Theme("communication", "Communication", "How information is shared, rumours, town halls, language of notices."),
    Theme("recognition", "Recognition", "Appreciation, awards, acknowledgement of good work."),
    Theme("work_life", "Hours & work-life balance", "Shift length, rest days, rotation, leave, time with family."),
    Theme("health_safety", "Health & safety", "Physical safety, PPE, permits, hazards, safety culture."),
    Theme("respect_inclusion", "Respect & fairness", "Respect, favouritism, discrimination, harassment, bullying, retaliation."),
    Theme("facilities", "Accommodation & facilities", "Camp/accommodation, transport, food, mess, offices, air conditioning."),
    Theme("tools_equipment", "Tools & systems", "Tools, spare parts, equipment, IT and ERP systems, digital tools."),
    Theme("job_security", "Job security", "Contracts, renewals, layoffs, project end, permanence."),
    Theme("teamwork", "Teamwork", "Collaboration within and between teams, colleagues."),
    Theme("wellbeing", "Wellbeing", "Stress, mental health, morale, personal pressure, support services."),
    Theme("ethics_integrity", "Ethics & integrity", "Fraud, falsified records, conflicts of interest, compliance breaches."),
    Theme("other", "Other / no content", "Nothing above fits, or the comment has no substantive content."),
)

SENTIMENTS: tuple[str, ...] = ("positive", "neutral", "negative", "mixed")

URGENCY_LEVELS: tuple[str, ...] = ("low", "medium", "high", "critical")
URGENCY_RANK: dict[str, int] = {u: i for i, u in enumerate(URGENCY_LEVELS)}

URGENCY_DEFINITIONS: dict[str, str] = {
    "low": "Praise, mild suggestion, or no action needed.",
    "medium": "A real problem that should be addressed in normal planning.",
    "high": "A serious problem hurting retention, pay, rest or operations; act this quarter.",
    "critical": "Possible immediate risk to someone's safety or wellbeing, or an allegation of serious misconduct such as harassment.",
}

SENSITIVE_CATEGORIES: tuple[str, ...] = (
    "none",
    "safety_hazard",
    "harassment",
    "discrimination",
    "wellbeing_risk",
    "misconduct",
    "retaliation",
)

SENSITIVE_DEFINITIONS: dict[str, str] = {
    "safety_hazard": "An unsafe condition or practice that could injure someone (missing guardrails, no gas test, damaged PPE, being told to work unsafely).",
    "harassment": "Bullying, abuse, threats, insults, sexual or unwanted personal comments.",
    "discrimination": "Unequal treatment based on nationality, ethnicity, religion, gender, age or similar.",
    "wellbeing_risk": "Signs of hopelessness, despair, severe distress or possible self-harm.",
    "misconduct": "Fraud, falsified records, theft, bribery or other ethics breaches.",
    "retaliation": "Punishment or worse treatment after someone raised a concern.",
}

# Plain-language labels for the dashboard.
SENSITIVE_LABELS: dict[str, str] = {
    "none": "None",
    "safety_hazard": "Safety hazard",
    "harassment": "Harassment",
    "discrimination": "Discrimination",
    "wellbeing_risk": "Wellbeing risk",
    "misconduct": "Misconduct",
    "retaliation": "Retaliation",
}

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "ar": "Arabic",
    "ur": "Urdu",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "tl": "Tagalog",
    "ms": "Malay",
    "und": "Undetermined",
}


def load_taxonomy(path: str | Path | None = None) -> tuple[Theme, ...]:
    """Return the default themes, or themes from a JSON list of {id,label,description}."""
    if not path:
        return DEFAULT_THEMES
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    themes = tuple(Theme(t["id"], t["label"], t.get("description", "")) for t in raw)
    ids = [t.id for t in themes]
    if len(set(ids)) != len(ids):
        raise ValueError("Theme ids in the taxonomy file must be unique.")
    if "other" not in ids:
        themes = themes + (Theme("other", "Other / no content", "Nothing else fits."),)
    return themes


def theme_ids(themes: tuple[Theme, ...] = DEFAULT_THEMES) -> list[str]:
    return [t.id for t in themes]


def theme_labels(themes: tuple[Theme, ...] = DEFAULT_THEMES) -> dict[str, str]:
    return {t.id: t.label for t in themes}
