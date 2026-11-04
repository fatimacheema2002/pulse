"""Manager action briefs.

Privacy rules, enforced in code rather than left to the prompt:
- confidential comments (safety, harassment, ...) never feed a manager's brief;
- groups with fewer than ``min_group_size`` usable comments get no brief;
- the model only sees counts and neutral English paraphrases, never original text,
  names or the language a comment was written in.
"""

from __future__ import annotations

import random
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .prompts import BRIEF_PROMPT_VERSION, brief_system_prompt, brief_user_prompt
from .providers import ChatProvider
from .taxonomy import DEFAULT_THEMES, Theme

THEME_ACTIONS: dict[str, tuple[str, str]] = {
    "workload": ("Agree with the team which work can be paused, reassigned or resourced differently.",
                 "List the three heaviest tasks this week and ask who is covering for vacancies."),
    "pay_benefits": ("Get clear answers from HR or payroll on the pay issues raised and share them with the team.",
                     "Send HR the questions raised and set a date to report back."),
    "manager_support": ("Set up regular one-to-ones and explain the reasons behind changes in priorities.",
                        "Book 15-minute check-ins with each team member over the next two weeks."),
    "leadership": ("Invite a senior leader to visit the team and hear concerns directly.",
                   "Request a site visit date from your director."),
    "career_growth": ("Discuss growth plans individually and make promotion criteria visible.",
                      "Share the current grade and promotion criteria with the team."),
    "training": ("Agree a short training plan for the most-requested skills.",
                 "Ask the team which two skills would help them most this quarter."),
    "communication": ("Share changes with the team before they spread informally, in the languages people use.",
                      "Start a weekly five-minute update at the shift briefing."),
    "recognition": ("Recognise good work specifically and publicly.",
                    "Name one piece of good work at the next team meeting."),
    "work_life": ("Review shift patterns, rest days and pending leave with the team.",
                  "Check the leave schedule for anyone whose leave has been postponed."),
    "health_safety": ("Walk the work area with the team and close the safety gaps they point out.",
                      "Hold a toolbox talk asking what feels unsafe right now."),
    "respect_inclusion": ("Reinforce standards of respect and fair allocation of work and rooms.",
                          "Review how shifts and assignments are allocated and share the rule."),
    "facilities": ("Escalate accommodation, transport or food issues with facilities and agree dates.",
                   "Raise the top facilities issue with the camp or facilities lead this week."),
    "tools_equipment": ("Fix the tool, parts or system delays that waste the team's time.",
                        "Track tool and parts waiting time for one week to size the problem."),
    "job_security": ("Share what is known about contracts and project timelines, and when more will be known.",
                     "Ask HR what can be said about contract renewals and when."),
    "teamwork": ("Create regular touchpoints with the teams you depend on.",
                 "Set up a joint coordination meeting with the other team."),
    "wellbeing": ("Make support services visible and check workload pressure.",
                  "Share the counselling and support contacts at the next meeting."),
    "ethics_integrity": ("Remind the team how to raise integrity concerns safely.",
                         "Share the speak-up channel details."),
    "other": ("Ask the team what one change would make the biggest difference.",
              "Put the question on the agenda for the next team meeting."),
}


@dataclass
class Brief:
    group: str
    n_comments: int
    status: str  # "ok" | "suppressed" | "error"
    stats: dict[str, Any] = field(default_factory=dict)
    content: dict[str, Any] = field(default_factory=dict)
    source: str = ""  # "model" | "template"
    warnings: list[str] = field(default_factory=list)
    note: str = ""

    def to_markdown(self, theme_labels: dict[str, str], group_label: str = "Team") -> str:
        lines = [f"## {group_label}: {self.group}", ""]
        if self.status != "ok":
            return "\n".join(lines + [f"_{self.note}_", ""])
        c = self.content
        lines += [f"**{c.get('headline', '')}**", "", f"Based on {self.n_comments} comments.", ""]
        if c.get("strengths"):
            lines += ["**What is working**"] + [f"- {s}" for s in c["strengths"]] + [""]
        if c.get("concerns"):
            lines += ["**Top concerns**"] + [
                f"- {theme_labels.get(x.get('theme', ''), x.get('theme', ''))}: {x.get('point', '')}" for x in c["concerns"]
            ] + [""]
        if c.get("actions"):
            lines += ["**Actions for the next 30 days**"] + [
                f"{i}. {a.get('action', '')} First step: {a.get('first_step', '')}" for i, a in enumerate(c["actions"], 1)
            ] + [""]
        if c.get("conversation_starter"):
            lines += [f"**Ask your team:** {c['conversation_starter']}", ""]
        return "\n".join(lines)


def _usable(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    mask = ~df["confidential"].fillna(False).astype(bool)
    if "skipped" in df:
        mask &= df["skipped"].fillna("") == ""
    if "error" in df:
        mask &= df["error"].fillna("") == ""
    return df[mask]


def theme_membership(df: pd.DataFrame, themes: tuple[Theme, ...]) -> pd.DataFrame:
    """Boolean frame: one column per theme, True where the comment mentions it."""
    sets = df["themes"].map(lambda ts: set(ts) if isinstance(ts, (list, tuple, set)) else set())
    return pd.DataFrame({t.id: sets.map(lambda s, tid=t.id: tid in s) for t in themes}, index=df.index)


def team_stats(team: pd.DataFrame, company: pd.DataFrame, themes: tuple[Theme, ...],
               membership: pd.DataFrame | None = None, company_share: dict[str, float] | None = None) -> dict[str, Any]:
    n = len(team)
    labels = {t.id: t.label for t in themes}
    if membership is None:
        membership = theme_membership(pd.concat([team, company]).loc[lambda d: ~d.index.duplicated()], themes)
    if company_share is None:
        company_share = membership.loc[company.index].mean().to_dict() if len(company) else {}
    member = membership.loc[team.index]
    neg = team["sentiment"].isin(["negative", "mixed"])
    pos = team["sentiment"].isin(["positive", "mixed"])
    sent = team["sentiment"].value_counts().to_dict()
    rows = []
    for tid in labels:
        if tid not in member:
            continue
        has = member[tid]
        k = int(has.sum())
        if not k:
            continue
        rows.append({
            "theme": tid,
            "label": labels[tid],
            "mentions": k,
            "negative": int((has & neg).sum()),
            "positive": int((has & pos).sum()),
            "share_of_team_comments_pct": round(100 * k / n),
            "company_share_pct": round(100 * float(company_share.get(tid, 0.0))),
        })
    rows.sort(key=lambda r: (-r["negative"], -r["mentions"]))
    return {
        "comments_analysed": n,
        "sentiment_counts": {s_: int(sent.get(s_, 0)) for s_ in ("positive", "neutral", "negative", "mixed")},
        "high_urgency_count": int(team["urgency"].isin(["high", "critical"]).sum()),
        "themes": rows[:6],
    }


def _summaries(team: pd.DataFrame, k: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    s = team[team["summary_en"].fillna("").str.len() > 0]
    priority = s[s["sentiment"].isin(["negative", "mixed"]) | s["urgency"].isin(["high", "critical"])]
    rest = s.drop(priority.index)
    chosen = list(priority["summary_en"].sample(min(len(priority), k * 2 // 3), random_state=seed)) if len(priority) else []
    chosen += list(rest["summary_en"].sample(min(len(rest), k - len(chosen)), random_state=seed)) if len(rest) else []
    rng.shuffle(chosen)
    return chosen


def template_brief(stats: dict[str, Any]) -> dict[str, Any]:
    themes = stats["themes"]
    n = stats["comments_analysed"]
    neg = sorted([t for t in themes if t["negative"]], key=lambda t: -t["negative"])[:3]
    pos = sorted([t for t in themes if t["positive"]], key=lambda t: -t["positive"])[:2]
    sc = stats["sentiment_counts"]
    tone = "mostly positive" if sc["positive"] > sc["negative"] + sc["mixed"] else "mixed" if sc["positive"] else "mostly critical"
    return {
        "headline": f"Feedback from {n} comments is {tone}" + (f"; the biggest concern is {neg[0]['label'].lower()}." if neg else "."),
        "strengths": [f"{t['label']}: {t['positive']} positive mentions." for t in pos] or ["No clear strengths stood out."],
        "concerns": [{"theme": t["theme"], "point": f"{t['negative']} of {n} comments raise concerns about {t['label'].lower()}."} for t in neg],
        "actions": [
            {"theme": t["theme"], "action": THEME_ACTIONS.get(t["theme"], THEME_ACTIONS["other"])[0],
             "first_step": THEME_ACTIONS.get(t["theme"], THEME_ACTIONS["other"])[1]}
            for t in (neg or [{"theme": "other"}])
        ][:3],
        "conversation_starter": "What is one thing that would make the biggest difference to your work next month?",
    }


def _allowed_numbers(stats: dict[str, Any]) -> set[int]:
    nums: set[int] = set(range(0, 11))  # small numbers (e.g. "3 actions", "two weeks") are fine
    n = stats["comments_analysed"]
    vals = [n, stats["high_urgency_count"], *stats["sentiment_counts"].values()]
    for t in stats["themes"]:
        vals += [t["mentions"], t["negative"], t["positive"], t["share_of_team_comments_pct"], t["company_share_pct"]]
    for v in vals:
        nums.add(int(v))
        if n:
            nums.add(round(100 * v / n))
    nums |= {30, 15, 100}
    return nums


def check_brief(content: dict[str, Any], stats: dict[str, Any], theme_ids: set[str]) -> list[str]:
    """Light grounding checks on a model-written brief."""
    warnings = []
    for key in ("concerns", "actions"):
        for item in content.get(key, []) or []:
            tid = item.get("theme", "")
            if tid not in theme_ids:
                warnings.append(f"{key[:-1]} refers to unknown theme '{tid}'")
    allowed = _allowed_numbers(stats)
    text = " ".join(
        [str(content.get("headline", ""))]
        + [str(c.get("point", "")) for c in content.get("concerns", []) or []]
        + [str(s) for s in content.get("strengths", []) or []]
    )
    for tok in re.findall(r"\d+", text):
        try:
            if int(tok) not in allowed:
                warnings.append(f"number {tok} does not match the team's statistics")
        except ValueError:
            continue
    if len(content.get("actions", []) or []) != 3:
        warnings.append("expected exactly 3 actions")
    return warnings


def build_briefs(
    results: pd.DataFrame,
    group_col: str,
    provider: ChatProvider | None = None,
    language: str = "English",
    min_group_size: int = 5,
    themes: tuple[Theme, ...] = DEFAULT_THEMES,
    max_summaries: int = 20,
    max_workers: int = 4,
    groups: list[str] | None = None,
) -> list[Brief]:
    usable = _usable(results)
    membership = theme_membership(usable, themes)
    company_share = membership.mean().to_dict() if len(usable) else {}
    theme_ids = {t.id for t in themes}
    all_groups = sorted(results[group_col].dropna().astype(str).unique())
    wanted = [g for g in all_groups if groups is None or g in groups]

    def one(group: str) -> Brief:
        team = usable[usable[group_col].astype(str) == group]
        n = len(team)
        if n < min_group_size:
            return Brief(group, n, "suppressed", note=(
                f"Not enough comments to report without risking anonymity ({n} usable, minimum {min_group_size}). "
                "These comments still count in the site and country views."))
        stats = team_stats(team, usable, themes, membership, company_share)
        if provider is None:
            return Brief(group, n, "ok", stats, template_brief(stats), "template")
        payload = {**stats, "comment_summaries": _summaries(team, max_summaries, seed=zlib.crc32(group.encode()) % 10_000)}
        messages = [
            {"role": "system", "content": brief_system_prompt(language)},
            {"role": "user", "content": brief_user_prompt(payload)},
        ]
        try:
            content, _ = provider.complete_json(messages, schema=None, reasoning_effort=provider.settings.reasoning_effort)
        except Exception as err:
            b = Brief(group, n, "ok", stats, template_brief(stats), "template")
            b.warnings.append(f"Model brief failed ({type(err).__name__}); showing the template brief.")
            return b
        warnings = check_brief(content, stats, theme_ids)
        return Brief(group, n, "ok", stats, content, "model", warnings)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(one, wanted))


__all__ = ["Brief", "build_briefs", "team_stats", "theme_membership", "template_brief", "check_brief", "BRIEF_PROMPT_VERSION"]
