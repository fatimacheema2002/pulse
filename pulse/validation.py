"""Release gate: compare the model's tags with a human-coded sample before results go out."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .taxonomy import LANGUAGE_NAMES, SENSITIVE_CATEGORIES, SENTIMENTS, URGENCY_LEVELS


# -- metrics -------------------------------------------------------------------
def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Sequence[str], weights: str | None = None) -> float:
    """Cohen's kappa. ``weights='quadratic'`` gives the ordinal (weighted) version."""
    idx = {lab: i for i, lab in enumerate(labels)}
    pairs = [(idx[x], idx[y]) for x, y in zip(a, b) if x in idx and y in idx]
    n, k = len(pairs), len(labels)
    if n == 0:
        return float("nan")
    obs = np.zeros((k, k))
    for i, j in pairs:
        obs[i, j] += 1
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    if weights == "quadratic":
        w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    else:
        w = 1.0 - np.eye(k)
    num, den = float((w * obs).sum()), float((w * exp).sum())
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return 1.0 - num / den


def multilabel_scores(pred: Sequence[set[str]], gold: Sequence[set[str]], labels: Sequence[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    tp_all = fp_all = fn_all = 0
    for lab in labels:
        tp = sum(1 for p, g in zip(pred, gold) if lab in p and lab in g)
        fp = sum(1 for p, g in zip(pred, gold) if lab in p and lab not in g)
        fn = sum(1 for p, g in zip(pred, gold) if lab not in p and lab in g)
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        rows.append({"theme": lab, "support": tp + fn, "predicted": tp + fp, "precision": prec, "recall": rec, "f1": f1})
    per = pd.DataFrame(rows)
    micro_p = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0.0
    micro_r = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0.0
    micro_f1 = 2 * tp_all / (2 * tp_all + fp_all + fn_all) if (2 * tp_all + fp_all + fn_all) else 0.0
    supported = per[per["support"] > 0]
    macro = float(supported["f1"].fillna(0).mean()) if len(supported) else 0.0
    exact = float(np.mean([p == g for p, g in zip(pred, gold)])) if len(pred) else 0.0
    return per, {"micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1, "macro_f1": macro, "exact_match": exact}


def bootstrap_ci(n: int, stat: Callable[[np.ndarray], float], reps: int = 400, seed: int = 7) -> tuple[float, float]:
    """95% percentile bootstrap interval over comment indices."""
    if n < 10:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        v = stat(rng.integers(0, n, n))
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# -- inputs ----------------------------------------------------------------------
def parse_themes(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip() for v in value if str(v).strip()}
    s = str(value or "").strip()
    if not s:
        return set()
    if s.startswith("["):
        try:
            return {str(v).strip() for v in json.loads(s)}
        except json.JSONDecodeError:
            pass
    for sep in (";", "|", ","):
        if sep in s:
            return {p.strip() for p in s.split(sep) if p.strip()}
    return {s}


def load_gold(path_or_buffer: Any) -> pd.DataFrame:
    """Human-coded sample: comment_id, themes, sentiment, urgency, sensitive_category (or sensitive yes/no)."""
    gold = pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False)
    gold.columns = [c.strip().lower() for c in gold.columns]
    if "comment_id" not in gold.columns:
        for alt in ("respondent_id", "id", "response_id"):
            if alt in gold.columns:
                gold = gold.rename(columns={alt: "comment_id"})
                break
    missing = {"comment_id", "themes", "sentiment", "urgency"} - set(gold.columns)
    if missing:
        raise ValueError(f"Human-coded file is missing columns: {sorted(missing)}")
    if "sensitive_category" not in gold.columns:
        flag = gold.get("sensitive", pd.Series(["no"] * len(gold)))
        gold["sensitive_category"] = np.where(flag.str.lower().isin(["1", "yes", "true", "y"]), "unspecified", "none")
    for c in ("sentiment", "urgency", "sensitive_category"):
        gold[c] = gold[c].str.strip().str.lower().replace({"": "none"} if c == "sensitive_category" else {})
    gold["themes"] = gold["themes"].apply(parse_themes)
    return gold


# -- report ------------------------------------------------------------------------
@dataclass
class Thresholds:
    theme_micro_f1: float = 0.70
    sentiment_kappa: float = 0.60
    urgency_weighted_kappa: float = 0.60
    sensitive_recall: float = 0.95
    language_theme_f1: float = 0.60
    min_sample: int = 100
    min_language_n: int = 15


@dataclass
class GateCheck:
    name: str
    value: float
    threshold: float
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    n: int
    n_missing: int
    themes: dict[str, float]
    themes_ci: tuple[float, float]
    per_theme: pd.DataFrame
    sentiment_accuracy: float
    sentiment_kappa: float
    sentiment_kappa_ci: tuple[float, float]
    sentiment_confusion: pd.DataFrame
    urgency_accuracy: float
    urgency_within_one: float
    urgency_weighted_kappa: float
    urgency_weighted_kappa_ci: tuple[float, float]
    urgency_confusion: pd.DataFrame
    n_sensitive_gold: int
    sensitive_recall_model: float
    sensitive_recall_routed: float
    sensitive_precision_routed: float
    missed_sensitive: pd.DataFrame
    by_language: pd.DataFrame
    disagreements: pd.DataFrame
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "theme_micro_f1": round(self.themes["micro_f1"], 3),
            "theme_macro_f1": round(self.themes["macro_f1"], 3),
            "sentiment_kappa": round(self.sentiment_kappa, 3),
            "urgency_weighted_kappa": round(self.urgency_weighted_kappa, 3),
            "sensitive_recall_routed": round(self.sensitive_recall_routed, 3),
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
        }


def _safe(x: float) -> float:
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def validate(pred: pd.DataFrame, gold: pd.DataFrame, thresholds: Thresholds | None = None,
             theme_labels: Sequence[str] | None = None) -> ValidationReport:
    th = thresholds or Thresholds()
    p = pred.copy()
    p["comment_id"] = p["comment_id"].astype(str)
    g = gold[[c for c in ("comment_id", "themes", "sentiment", "urgency", "sensitive_category", "language") if c in gold]].copy()
    g["comment_id"] = g["comment_id"].astype(str)
    if "language" in g and "language" in p:
        g = g.rename(columns={"language": "language_gold"})
        p = p.rename(columns={"language": "language_pred"})
    elif "language" in p:
        p = p.rename(columns={"language": "language_pred"})
    m = g.merge(p, on="comment_id", how="left", suffixes=("_gold", "_pred"), indicator=True)
    n_missing = int((m["_merge"] == "left_only").sum())
    m = m[m["_merge"] == "both"].reset_index(drop=True)
    n = len(m)

    pred_sets = [set(v) if isinstance(v, (list, tuple, set)) else parse_themes(v) for v in m["themes_pred"]]
    gold_sets = list(m["themes_gold"])
    labels = list(theme_labels) if theme_labels else sorted(set().union(*gold_sets, *pred_sets)) if n else []
    per_theme, theme_scores = multilabel_scores(pred_sets, gold_sets, labels)

    def micro_f1(idx: np.ndarray) -> float:
        tp = fp = fn = 0
        for i in idx:
            pp, gg = pred_sets[i], gold_sets[i]
            tp += len(pp & gg)
            fp += len(pp - gg)
            fn += len(gg - pp)
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")

    sp, sg = list(m["sentiment_pred"]), list(m["sentiment_gold"])
    up, ug = list(m["urgency_pred"]), list(m["urgency_gold"])
    s_kappa = cohen_kappa(sg, sp, SENTIMENTS)
    u_kappa = cohen_kappa(ug, up, URGENCY_LEVELS, weights="quadratic")
    rank = {u: i for i, u in enumerate(URGENCY_LEVELS)}
    within_one = float(np.mean([abs(rank.get(a, 0) - rank.get(b, 0)) <= 1 for a, b in zip(ug, up)])) if n else 0.0

    gold_sens = m["sensitive_category_gold"].fillna("none") != "none"
    model_sens = m["sensitive_category_pred"].fillna("none") != "none"
    routed = m["confidential"].fillna(False).astype(bool) if "confidential" in m else model_sens
    n_sens = int(gold_sens.sum())
    rec_model = float((model_sens & gold_sens).sum() / n_sens) if n_sens else float("nan")
    rec_routed = float((routed & gold_sens).sum() / n_sens) if n_sens else float("nan")
    prec_routed = float((routed & gold_sens).sum() / routed.sum()) if routed.sum() else float("nan")
    missed_cols = [c for c in ("comment_id", "language_pred", "text", "sensitive_category_gold", "sensitive_category_pred", "urgency_pred") if c in m]
    missed = m.loc[gold_sens & ~routed, missed_cols]

    lang_col = "language_gold" if "language_gold" in m else ("language_pred" if "language_pred" in m else None)
    if lang_col is None:
        m["language_pred"] = "und"
        lang_col = "language_pred"
    by_lang_rows = []
    for lang, grp in m.groupby(m[lang_col].fillna("und")):
        idx = grp.index.to_numpy()
        f1 = micro_f1(idx)
        acc = float(np.mean([sp[i] == sg[i] for i in idx]))
        g_s = gold_sens.iloc[idx]
        by_lang_rows.append({
            "language": lang,
            "language_name": LANGUAGE_NAMES.get(lang, lang),
            "n": len(idx),
            "theme_f1": f1,
            "sentiment_accuracy": acc,
            "urgency_accuracy": float(np.mean([up[i] == ug[i] for i in idx])),
            "sensitive_in_sample": int(g_s.sum()),
            "sensitive_routed": int((routed.iloc[idx] & g_s).sum()),
        })
    by_language = pd.DataFrame(by_lang_rows).sort_values("n", ascending=False) if by_lang_rows else pd.DataFrame()

    dis_mask = [(pred_sets[i] != gold_sets[i]) or sp[i] != sg[i] or up[i] != ug[i] for i in range(n)]
    dis = m.loc[dis_mask].copy()
    if len(dis):
        dis["themes_model"] = [", ".join(sorted(pred_sets[i])) for i in dis.index]
        dis["themes_human"] = [", ".join(sorted(gold_sets[i])) for i in dis.index]
    keep = [c for c in ("comment_id", lang_col, "text", "themes_human", "themes_model", "sentiment_gold", "sentiment_pred",
                        "urgency_gold", "urgency_pred", "summary_en") if c in dis.columns]
    disagreements = dis[keep] if len(dis) else pd.DataFrame(columns=keep)

    report = ValidationReport(
        n=n,
        n_missing=n_missing,
        themes=theme_scores,
        themes_ci=bootstrap_ci(n, micro_f1),
        per_theme=per_theme,
        sentiment_accuracy=float(np.mean([a == b for a, b in zip(sp, sg)])) if n else 0.0,
        sentiment_kappa=s_kappa,
        sentiment_kappa_ci=bootstrap_ci(n, lambda idx: cohen_kappa([sg[i] for i in idx], [sp[i] for i in idx], SENTIMENTS)),
        sentiment_confusion=pd.crosstab(pd.Series(sg, name="human"), pd.Series(sp, name="model")).reindex(
            index=list(SENTIMENTS), columns=list(SENTIMENTS), fill_value=0),
        urgency_accuracy=float(np.mean([a == b for a, b in zip(up, ug)])) if n else 0.0,
        urgency_within_one=within_one,
        urgency_weighted_kappa=u_kappa,
        urgency_weighted_kappa_ci=bootstrap_ci(
            n, lambda idx: cohen_kappa([ug[i] for i in idx], [up[i] for i in idx], URGENCY_LEVELS, "quadratic")),
        urgency_confusion=pd.crosstab(pd.Series(ug, name="human"), pd.Series(up, name="model")).reindex(
            index=list(URGENCY_LEVELS), columns=list(URGENCY_LEVELS), fill_value=0),
        n_sensitive_gold=n_sens,
        sensitive_recall_model=rec_model,
        sensitive_recall_routed=rec_routed,
        sensitive_precision_routed=prec_routed,
        missed_sensitive=missed,
        by_language=by_language,
        disagreements=disagreements,
    )

    checks = [
        GateCheck("Human-coded sample size", n, th.min_sample, n >= th.min_sample,
                  f"{n} comments matched" + (f"; {n_missing} ids not found in this run" if n_missing else "")),
        GateCheck("Theme agreement (micro F1)", _safe(theme_scores["micro_f1"]), th.theme_micro_f1,
                  _safe(theme_scores["micro_f1"]) >= th.theme_micro_f1),
        GateCheck("Sentiment agreement (Cohen's kappa)", _safe(s_kappa), th.sentiment_kappa, _safe(s_kappa) >= th.sentiment_kappa),
        GateCheck("Urgency agreement (weighted kappa)", _safe(u_kappa), th.urgency_weighted_kappa,
                  _safe(u_kappa) >= th.urgency_weighted_kappa),
    ]
    if n_sens:
        checks.append(GateCheck("Sensitive comments routed to review (recall)", _safe(rec_routed), th.sensitive_recall,
                                _safe(rec_routed) >= th.sensitive_recall,
                                f"{int((routed & gold_sens).sum())} of {n_sens} routed; model alone caught {int((model_sens & gold_sens).sum())}"))
    if len(by_language):
        eligible = by_language[by_language["n"] >= th.min_language_n]
        if len(eligible):
            worst = eligible.loc[eligible["theme_f1"].idxmin()]
            checks.append(GateCheck("Weakest language, theme F1", _safe(worst["theme_f1"]), th.language_theme_f1,
                                    _safe(worst["theme_f1"]) >= th.language_theme_f1,
                                    f"{worst['language_name']} (n={int(worst['n'])})"))
    report.checks = checks
    return report


def compare(reports: dict[str, ValidationReport]) -> pd.DataFrame:
    """Side-by-side table, e.g. Muse Spark vs open weights vs keyword baseline."""
    rows = []
    for name, r in reports.items():
        rows.append({
            "run": name,
            "theme micro F1": r.themes["micro_f1"],
            "sentiment kappa": r.sentiment_kappa,
            "urgency weighted kappa": r.urgency_weighted_kappa,
            "sensitive recall": r.sensitive_recall_routed,
            "weakest language F1": float(r.by_language["theme_f1"].min()) if len(r.by_language) else float("nan"),
            "gate": "PASS" if r.passed else "FAIL",
        })
    return pd.DataFrame(rows)


def gold_template(results: pd.DataFrame, n: int = 150, seed: int = 11) -> pd.DataFrame:
    """A blank coding sheet, stratified by language, for the HR team to fill in by hand."""
    df = results[results["skipped"].fillna("") == ""] if "skipped" in results.columns else results
    per_lang = max(1, n // max(1, df["language"].nunique()))
    parts = [grp.sample(min(len(grp), per_lang), random_state=seed) for _, grp in df.groupby("language")]
    sample = pd.concat(parts).sample(frac=1, random_state=seed).head(n) if parts else df.head(0)
    out = sample[["comment_id", "language", "text"]].copy()
    for col in ("themes", "sentiment", "urgency", "sensitive_category", "coder"):
        out[col] = ""
    return out


__all__ = [
    "Thresholds", "ValidationReport", "GateCheck", "validate", "compare", "load_gold", "gold_template",
    "cohen_kappa", "multilabel_scores", "SENSITIVE_CATEGORIES",
]
