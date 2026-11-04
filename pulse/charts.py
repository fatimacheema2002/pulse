"""Altair charts for the dashboard. Colours follow one validated reference palette:
sequential = one blue ramp; sentiment = blue/red diverging with a gray neutral midpoint;
urgency = the fixed status palette, always shown with a text label."""

from __future__ import annotations

import altair as alt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
RED = "#e34948"
SENTIMENT_ORDER = ["negative", "mixed", "neutral", "positive"]
SENTIMENT_COLORS = ["#e34948", "#f0a19f", "#c3c2b7", "#2a78d6"]
URGENCY_COLORS = {"critical": "#d03b3b", "high": "#ec835a", "medium": "#fab219", "low": "#c3c2b7"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _style(chart: alt.Chart | alt.LayerChart, height: int | None = None, step: int | None = None) -> alt.Chart:
    if step:
        chart = chart.properties(height=alt.Step(step))
    elif height:
        chart = chart.properties(height=height)
    c = chart.configure(background=SURFACE, font=FONT, padding={"left": 4, "right": 12, "top": 8, "bottom": 4})
    c = c.configure_view(stroke=None)
    c = c.configure_axis(
        labelColor=INK_2, titleColor=INK_2, gridColor=GRID, domainColor=AXIS, tickColor=AXIS,
        labelFontSize=12, titleFontSize=12, titleFontWeight="normal", labelLimit=220,
    )
    c = c.configure_legend(labelColor=INK_2, titleColor=INK_2, labelFontSize=12, titleFontSize=12,
                           orient="top", direction="horizontal", symbolType="square")
    return c


def explode_themes(df: pd.DataFrame) -> pd.DataFrame:
    e = df.explode("themes").rename(columns={"themes": "theme"})
    return e[e["theme"].notna()]


def sentiment_by_theme(df: pd.DataFrame, labels: dict[str, str]) -> alt.Chart:
    """Diverging stacked bar: concerns to the left of zero, praise to the right, neutral centred."""
    e = explode_themes(df)
    e = e[e["theme"] != "other"]
    counts = e.groupby(["theme", "sentiment"]).size().unstack(fill_value=0).reindex(columns=SENTIMENT_ORDER, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts = counts.sort_values(["negative", "mixed"], ascending=False)
    rows = []
    for theme, r in counts.iterrows():
        half = r["neutral"] / 2
        edges = {
            "negative": (-(r["negative"] + r["mixed"] + half), -(r["mixed"] + half)),
            "mixed": (-(r["mixed"] + half), -half),
            "neutral": (-half, half),
            "positive": (half, half + r["positive"]),
        }
        for s, (a, b) in edges.items():
            if r[s]:
                rows.append({"theme": labels.get(theme, theme), "sentiment": s, "start": a, "end": b,
                             "count": int(r[s]), "share": r[s] / r["total"], "total": int(r["total"])})
    data = pd.DataFrame(rows)
    order = [labels.get(t, t) for t in counts.index]
    bars = alt.Chart(data).mark_bar(stroke=SURFACE, strokeWidth=2, height={"band": 0.7}).encode(
        y=alt.Y("theme:N", sort=order, title=None),
        x=alt.X("start:Q", title="Comments  (← concerns · praise →)", axis=alt.Axis(labelExpr="abs(datum.value)")),
        x2="end:Q",
        color=alt.Color("sentiment:N", scale=alt.Scale(domain=SENTIMENT_ORDER, range=SENTIMENT_COLORS),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("theme:N", title="Theme"), alt.Tooltip("sentiment:N", title="Sentiment"),
                 alt.Tooltip("count:Q", title="Comments"), alt.Tooltip("share:Q", title="Share of theme", format=".0%"),
                 alt.Tooltip("total:Q", title="All comments on theme")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=AXIS, strokeWidth=1).encode(x="x:Q")
    return _style(alt.layer(bars, zero), step=26)


def concern_heatmap(df: pd.DataFrame, labels: dict[str, str], row: str = "site", min_comments: int = 10) -> alt.Chart:
    """Share of each site's comments that raise a concern about each theme."""
    e = explode_themes(df)
    e = e[(e["theme"] != "other")]
    totals = df.groupby(row).size().rename("n")
    neg = e[e["sentiment"].isin(["negative", "mixed"])].groupby([row, "theme"]).size().rename("concerns").reset_index()
    neg = neg.merge(totals, left_on=row, right_index=True)
    neg = neg[neg["n"] >= min_comments]
    neg["share"] = neg["concerns"] / neg["n"]
    neg["theme_label"] = neg["theme"].map(lambda t: labels.get(t, t))
    theme_order = (neg.groupby("theme_label")["concerns"].sum().sort_values(ascending=False).index.tolist())
    base = alt.Chart(neg).encode(
        x=alt.X("theme_label:N", sort=theme_order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=160)),
        y=alt.Y(f"{row}:N", title=None, sort=sorted(neg[row].unique())),
    )
    rect = base.mark_rect(stroke=SURFACE, strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("share:Q", scale=alt.Scale(range=BLUE_RAMP, domain=[0, max(0.3, float(neg["share"].max()) if len(neg) else 0.3)]),
                        legend=alt.Legend(title="Share raising a concern", format=".0%", orient="top",
                                          direction="horizontal", gradientLength=180, titleLimit=300)),
        tooltip=[alt.Tooltip(f"{row}:N", title=row.replace("_", " ").title()), alt.Tooltip("theme_label:N", title="Theme"),
                 alt.Tooltip("concerns:Q", title="Comments raising it"), alt.Tooltip("n:Q", title="Comments from here"),
                 alt.Tooltip("share:Q", format=".0%", title="Share")],
    )
    hot = base.transform_filter("datum.share >= 0.2").mark_text(fontSize=11, color="#ffffff").encode(
        text=alt.Text("share:Q", format=".0%"))
    return _style(alt.layer(rect, hot), step=30)


def engagement_drivers(df: pd.DataFrame, labels: dict[str, str], score_col: str = "engagement_score", min_n: int = 20) -> alt.Chart | None:
    if score_col not in df.columns:
        return None
    d = df.copy()
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d[d[score_col].notna()]
    if d.empty:
        return None
    overall = d[score_col].mean()
    e = explode_themes(d)
    e = e[e["theme"] != "other"]
    g = e.groupby("theme")[score_col].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_n]
    g["diff"] = g["mean"] - overall
    g["label"] = g["theme"].map(lambda t: labels.get(t, t))
    g["direction"] = g["diff"].map(lambda v: "Above average" if v >= 0 else "Below average")
    chart = alt.Chart(g).mark_bar(cornerRadiusEnd=4, height={"band": 0.6}).encode(
        y=alt.Y("label:N", sort=alt.EncodingSortField("diff", order="ascending"), title=None),
        x=alt.X("diff:Q", title=f"Score difference vs average ({overall:.2f}/5)"),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Below average", "Above average"], range=[RED, BLUE]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("label:N", title="Theme"), alt.Tooltip("mean:Q", format=".2f", title="Avg score when mentioned"),
                 alt.Tooltip("diff:Q", format="+.2f", title="Difference"), alt.Tooltip("size:Q", title="Comments")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=AXIS).encode(x="x:Q")
    return _style(alt.layer(chart, zero), step=24)


def language_bars(df: pd.DataFrame, names: dict[str, str]) -> alt.Chart:
    c = df["language"].map(lambda l: names.get(l, l)).value_counts().rename_axis("language").reset_index(name="comments")
    chart = alt.Chart(c).mark_bar(color=BLUE, cornerRadiusEnd=4, height={"band": 0.6}).encode(
        y=alt.Y("language:N", sort="-x", title=None),
        x=alt.X("comments:Q", title="Comments"),
        tooltip=["language", "comments"],
    )
    text = chart.mark_text(align="left", dx=4, color=INK_2, fontSize=12).encode(text="comments:Q")
    return _style(alt.layer(chart, text), step=30)


def theme_f1_bars(per_theme: pd.DataFrame, labels: dict[str, str], threshold: float) -> alt.Chart:
    d = per_theme[per_theme["support"] > 0].copy()
    d["label"] = d["theme"].map(lambda t: labels.get(t, t))
    d["f1"] = d["f1"].fillna(0)
    bars = alt.Chart(d).mark_bar(color=BLUE, cornerRadiusEnd=4, height={"band": 0.6}).encode(
        y=alt.Y("label:N", sort=alt.EncodingSortField("f1", order="ascending"), title=None),
        x=alt.X("f1:Q", scale=alt.Scale(domain=[0, 1]), title="Agreement with human coder (F1)"),
        tooltip=[alt.Tooltip("label:N", title="Theme"), alt.Tooltip("f1:Q", format=".2f", title="F1"),
                 alt.Tooltip("precision:Q", format=".2f"), alt.Tooltip("recall:Q", format=".2f"),
                 alt.Tooltip("support:Q", title="Human-coded mentions")],
    )
    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color=INK_2, strokeWidth=1).encode(x="x:Q")
    return _style(alt.layer(bars, rule), step=22)


def confusion(cm: pd.DataFrame, title: str) -> alt.Chart:
    d = cm.reset_index().melt(id_vars=cm.index.name or "index", var_name="model", value_name="count")
    d = d.rename(columns={cm.index.name or "index": "human"})
    order = list(cm.index)
    base = alt.Chart(d).encode(
        x=alt.X("model:N", sort=order, title="Model said", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("human:N", sort=order, title="Human coder said"),
    )
    rect = base.mark_rect(stroke=SURFACE, strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("count:Q", scale=alt.Scale(range=BLUE_RAMP), legend=None),
        tooltip=["human", "model", "count"],
    )
    text = base.mark_text(fontSize=12).encode(
        text="count:Q",
        color=alt.condition(f"datum.count > {max(1, d['count'].max()) * 0.45}", alt.value("#ffffff"), alt.value(INK)),
    )
    return _style(alt.layer(rect, text).properties(title=alt.TitleParams(title, color=INK, fontSize=13, anchor="start")),
                  height=220)
