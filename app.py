"""Pulse dashboard.  Run with:  streamlit run app.py"""

from __future__ import annotations

import html
import io
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from pulse import charts
from pulse.briefs import build_briefs
from pulse.config import BACKEND_LABELS, from_env
from pulse.local import RECOMMENDED, best_model, installed_models
from pulse.pipeline import ColumnMap, is_text, list_runs, load_run, load_survey, run_survey
from pulse.providers import ChatProvider
from pulse.safety import review_category, severity_rank
from pulse.taxonomy import DEFAULT_THEMES, LANGUAGE_NAMES, SENSITIVE_LABELS, theme_labels
from pulse.validation import Thresholds, compare, gold_template, load_gold, validate

ROOT = Path(__file__).resolve().parent
DEMO_SURVEY = ROOT / "data" / "synthetic_survey.csv"
DEMO_GOLD = ROOT / "data" / "human_coded_sample.csv"
LABELS = theme_labels(DEFAULT_THEMES)
URGENCY_BADGE = {"critical": "red", "high": "orange", "medium": "yellow", "low": "gray"}
SENTIMENT_BADGE = {"negative": "red", "mixed": "violet", "neutral": "gray", "positive": "blue"}
BRIEF_LANGUAGES = ["English", "Arabic", "Urdu", "Hindi"]

st.set_page_config(page_title="Pulse", page_icon="🫀", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1400px;}
      [data-testid="stMetricValue"] {font-size: 1.9rem;}
      .pulse-orig {font-size: 1.05rem; line-height: 1.6; unicode-bidi: plaintext;}
      .pulse-muted {color: #52514e; font-size: 0.9rem;}
      .pulse-evidence {background: #fff3c4; padding: 0 3px; border-radius: 3px;}
      [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {unicode-bidi: plaintext; text-align: start;}
    </style>
    """,
    unsafe_allow_html=True,
)

ss = st.session_state
for key, default in {"results": None, "manifest": None, "report": None, "gate": {}, "decisions": {}, "briefs": None,
                     "gold": None, "gold_name": None, "api_key": ""}.items():
    ss.setdefault(key, default)


# ---------------------------------------------------------------------------------------------------------------- helpers
def badge(text: str, color: str) -> str:
    return f":{color}-badge[{text}]"


def lang_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def theme_text(themes) -> str:
    return ", ".join(LABELS.get(t, t) for t in (themes or []))


def group_columns(df: pd.DataFrame) -> list[str]:
    skip = {"comment_id", "text", "text_redacted", "summary_en", "evidence", "language", "sentiment", "urgency",
            "sensitive_category", "primary_theme", "backend", "model", "prompt_version", "skipped", "error", "redactions"}
    out = []
    for c in df.columns:
        if c in skip or not is_text(df[c]):
            continue
        vals = df[c].dropna()
        if vals.map(lambda v: isinstance(v, str)).all() and 2 <= vals.nunique() <= 200:
            out.append(c)
    return out


def build_settings():
    backend = ss.get("backend_choice", "offline")
    s = from_env(backend)
    if backend == "meta":
        s = replace(s, model=ss.get("meta_model") or s.model, api_key=s.api_key or ss.api_key)
    elif backend == "open_weights":
        s = replace(s, base_url=ss.get("open_url") or s.base_url, model=ss.get("open_model") or s.model)
    elif backend == "local":
        s = replace(s, base_url=ss.get("local_url") or s.base_url, model=ss.get("local_model") or s.model)
    return s


def set_results(results: pd.DataFrame, manifest: dict) -> None:
    ss.results, ss.manifest = results, manifest
    ss.report, ss.gate, ss.briefs, ss.decisions = None, {}, None, {}


# ---------------------------------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## Pulse")
    st.caption("Every survey comment read in its own language, sensitive issues routed to people, "
               "and a release gate before anything goes to managers.")

    st.markdown("#### 1 · Model")
    env_default = from_env().backend
    if env_default == "offline" and not os.environ.get("PULSE_BACKEND"):
        env_default = "local"  # no API key: default to running Meta's open models on this computer
    options = list(BACKEND_LABELS)
    local_ready = True
    ss.backend_choice = st.radio("Backend", options, index=options.index(env_default),
                                 format_func=lambda b: BACKEND_LABELS[b], label_visibility="collapsed")
    if ss.backend_choice == "meta":
        base = from_env("meta")
        st.text_input("Model", value=base.model, key="meta_model")
        if not base.api_key:
            ss.api_key = st.text_input("MODEL_API_KEY", type="password", value=ss.api_key,
                                       help="Kept in this browser session only. Put it in .env to avoid retyping.")
        else:
            st.caption("API key loaded from environment.")
    elif ss.backend_choice == "open_weights":
        base = from_env("open_weights")
        st.text_input("Endpoint (OpenAI-compatible)", value=base.base_url, key="open_url")
        st.text_input("Model", value=base.model, key="open_model")
    elif ss.backend_choice == "local":
        base = from_env("local")
        url = st.text_input("Ollama address", value=base.base_url, key="local_url")
        models = installed_models(url)
        local_ready = False
        if models is None:
            st.warning("Ollama isn't running. Install it from ollama.com, open it, then in a terminal run "
                       "`ollama pull llama3.1:8b` (or `ollama pull muse-glimmer:30b` on a big GPU) and refresh this page.",
                       icon="⚙️")
        elif not models:
            st.warning("Ollama is running but no model is installed. In a terminal run `ollama pull llama3.1:8b`, "
                       "then refresh this page.", icon="⚙️")
        else:
            names = [m.name for m in models]
            by_name = {m.name: m for m in models}
            pick = best_model(models)
            if ss.get("local_model") not in names:
                ss.local_model = pick
            st.selectbox("Model", names, key="local_model",
                         format_func=lambda n: f"{n} · {by_name[n].parameters or '?'} · {by_name[n].size_gb} GB")
            if not by_name[ss.local_model].is_meta:
                st.caption("This is not a Meta model. The hackathon asks for Meta's models (Muse Glimmer or Llama).")
            local_ready = True
        with st.expander("Which model should I install?"):
            for tag, desc, size, needs in RECOMMENDED:
                st.markdown(f"**`ollama pull {tag}`**  \n{desc}, {size}. Needs: {needs}.")
    settings = build_settings()
    if settings.backend == "offline":
        st.info("No model: keyword rules only. Useful as a baseline to compare against.", icon="ℹ️")
    elif settings.backend == "local":
        st.success("Runs on this computer: no API key, and comment text never leaves this machine.", icon="🔒")
    elif settings.leaves_network:
        st.warning("Comments are sent to an external API. Names, phone numbers, emails and IDs are removed first; "
                   "original text stays on this machine.", icon="🌐")
    else:
        st.success("Self-hosted model: comment text stays inside your network.", icon="🔒")
    if settings.uses_model and local_ready and st.button("Test connection", width="stretch"):
        try:
            with st.spinner("Calling the model…"):
                reply = ChatProvider(settings).check()
            st.success(f"Connected. Model replied: {reply}")
        except Exception as err:  # noqa: BLE001
            st.error(f"Connection failed: {err}")

    st.markdown("#### 2 · Survey")
    source = st.radio("Data", ["Synthetic demo survey", "Upload a CSV"], label_visibility="collapsed")
    survey_df, survey_name, cols = None, None, None
    if source == "Upload a CSV":
        up = st.file_uploader("Survey export (CSV, UTF-8)", type=["csv"])
        if up is not None:
            survey_df, survey_name = load_survey(up), up.name
    elif DEMO_SURVEY.exists():
        survey_df, survey_name = load_survey(DEMO_SURVEY), DEMO_SURVEY.name
    if survey_df is not None:
        guess = ColumnMap.guess(survey_df)
        with st.expander("Columns", expanded=source == "Upload a CSV"):
            text_col = st.selectbox("Comment text", list(survey_df.columns), index=list(survey_df.columns).index(guess.text))
            id_opts = ["(generate)"] + list(survey_df.columns)
            id_col = st.selectbox("Comment / respondent id", id_opts, index=id_opts.index(guess.id) if guess.id else 0)
            name_cols = st.multiselect("Columns with people's names (redacted from comments)", list(survey_df.columns),
                                       default=guess.names)
        cols = ColumnMap(text=text_col, id=None if id_col == "(generate)" else id_col, names=name_cols)
        n_comments = int((survey_df[text_col].str.strip() != "").sum())
        st.caption(f"{len(survey_df):,} respondents · {n_comments:,} comments")
        scope_opts = ["All comments", "Human-coded sample only", "Random sample"]
        scope = st.radio("Run on", scope_opts, index=1 if settings.backend == "local" else 0,
                         help="The human-coded sample is the fastest live demo: it is small and the release gate can score it.")
        if settings.backend == "local":
            st.caption("Local models are slower than the API. Start with the human-coded sample. Progress shows the time "
                       "left, and finished comments are saved, so if you stop, the next run picks up where it left off.")
        n_random = st.number_input("Sample size", 50, max(50, n_comments), min(300, max(50, n_comments)), step=50) \
            if scope == "Random sample" else None

        run_disabled = not settings.is_configured or not local_ready
        if st.button("Analyse comments", type="primary", width="stretch", disabled=run_disabled):
            work = survey_df
            if scope == "Human-coded sample only" and DEMO_GOLD.exists():
                ids = set(load_gold(DEMO_GOLD)["comment_id"])
                key = cols.id or "respondent_id"
                work = survey_df[survey_df[key].isin(ids)] if key in survey_df else survey_df
            elif scope == "Random sample":
                work = survey_df[survey_df[text_col].str.strip() != ""].sample(int(n_random), random_state=1)
            bar = st.progress(0.0, text="Starting…")

            def progress(done: int, total: int, msg: str) -> None:
                bar.progress(done / max(total, 1), text=msg)

            try:
                results, manifest = run_survey(work, settings, cols, progress=progress, source_name=survey_name)
                set_results(results, manifest)
                bar.empty()
                st.toast(f"Analysed {len(results):,} comments", icon="✅")
            except Exception as err:  # noqa: BLE001
                bar.empty()
                st.error(f"Run failed: {err}")
        if run_disabled:
            st.caption("Install a model with Ollama (see above) to run." if settings.backend == "local"
                       else "Add an API key or endpoint above to run.")

    runs = list_runs()
    if runs:
        st.markdown("#### Saved runs")
        labels = {r["run_id"]: f"{r['created_utc'][:16].replace('T', ' ')} · {r['model']} · {r['n_comments']:,}" for r in runs}
        pick = st.selectbox("Open a saved run", list(labels), format_func=labels.get, label_visibility="collapsed")
        if st.button("Open run", width="stretch"):
            set_results(*load_run(pick))


# ---------------------------------------------------------------------------------------------------------------- empty state
res: pd.DataFrame | None = ss.results
if res is None:
    st.title("Pulse")
    st.markdown("##### Read every engagement-survey comment, in every language, before results go out.")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**Reads the original language**")
        st.caption("A Meta model (Muse Glimmer or Llama on this computer, or Muse Spark through the API) tags each "
                   "comment by theme, sentiment and urgency in Arabic, Urdu (including Roman Urdu), Hindi or English, "
                   "with no translation step and a verbatim evidence quote.")
    with c2.container(border=True):
        st.markdown("**Routes sensitive issues to people**")
        st.caption("Safety, harassment, discrimination and wellbeing risks go to a confidential review queue, caught by "
                   "the model and by an independent keyword backstop. They never reach a manager's brief.")
    with c3.container(border=True):
        st.markdown("**Checks itself before release**")
        st.caption("Tags are compared with a human-coded sample. If agreement is too low, manager briefs are blocked, "
                   "so you can see whether a small local model is good enough before anyone acts on it.")
    st.markdown("Choose a model and a survey in the sidebar, then **Analyse comments**, or open a saved run.")
    if DEMO_SURVEY.exists():
        demo = load_survey(DEMO_SURVEY)
        demo = demo[demo["comment"] != ""]
        st.markdown("###### A few comments from the synthetic survey")
        st.dataframe(demo.sample(8, random_state=3)[["country", "site", "comment"]], hide_index=True, width="stretch")
    st.stop()


# ---------------------------------------------------------------------------------------------------------------- shared data
man = ss.manifest or {}
for col in ("themes", "review_reasons", "keyword_flags"):
    res[col] = res[col].apply(lambda v: list(v) if isinstance(v, (list, tuple)) else [])
substantive = res[(res["skipped"].fillna("") == "") & (res["error"].fillna("") == "")]
public = substantive[~substantive["confidential"].astype(bool)]
confidential = res[res["confidential"].astype(bool)]
gcols = group_columns(res)

BUNDLED = "Bundled sample (synthetic survey)"


@st.cache_data(show_spinner=False, max_entries=16)
def validate_cached(run_id: str, gold_bytes: bytes, th: tuple, _res: pd.DataFrame):
    return validate(_res, load_gold(io.BytesIO(gold_bytes)), Thresholds(*th), theme_labels=[t.id for t in DEFAULT_THEMES])


def current_gold_bytes() -> bytes | None:
    if ss.get("gold_src", BUNDLED) == "Upload":
        up = ss.get("gold_up")
        return up.getvalue() if up is not None else None
    return DEMO_GOLD.read_bytes() if DEMO_GOLD.exists() else None


th = Thresholds(ss.get("th_theme", 0.70), ss.get("th_sent", 0.60), ss.get("th_urg", 0.60), ss.get("th_sens", 0.95))
th_key = (th.theme_micro_f1, th.sentiment_kappa, th.urgency_weighted_kappa, th.sensitive_recall)
gold_bytes = current_gold_bytes()
gold_error = None
rep = None
if gold_bytes:
    try:
        rep = validate_cached(man.get("run_id", "live"), gold_bytes, th_key, res)
    except Exception as err:  # noqa: BLE001
        gold_error = str(err)
ss.report = rep
reason_txt = (ss.get("gate_reason") or "").strip()
ss.gate = {} if rep is None or rep.n == 0 else {
    "passed": rep.passed, "override": bool(ss.get("gate_override")) and bool(reason_txt) and not rep.passed,
    "reason": reason_txt}

st.title("Pulse")
st.caption(f"Run {man.get('run_id', '')} · {BACKEND_LABELS.get(man.get('backend', ''), man.get('backend', ''))} · "
           f"model `{man.get('model', '')}` · {man.get('source', '')}")

tab_over, tab_comments, tab_review, tab_valid, tab_briefs, tab_run = st.tabs(
    ["Overview", "Comments", f"Review queue ({len(confidential)})", "Release gate", "Manager briefs", "Run details"])

# ---------------------------------------------------------------------------------------------------------------- overview
with tab_over:
    non_en = substantive[substantive["language"] != "en"]
    sens_non_en = confidential[confidential["language"] != "en"]
    k = st.columns(5)
    k[0].metric("Comments analysed", f"{len(res):,}", help=f"{len(substantive):,} with real content")
    k[1].metric("Not written in English", f"{len(non_en) / max(1, len(substantive)):.0%}",
                help="Comments an English-only reading would miss or need translated")
    k[2].metric("Negative or mixed", f"{substantive['sentiment'].isin(['negative', 'mixed']).mean():.0%}")
    k[3].metric("Sent to confidential review", f"{len(confidential):,}",
                help=f"{len(sens_non_en)} of them not in English")
    gate = ss.gate or {}
    k[4].metric("Release gate", "Passed" if gate.get("passed") else ("Failed" if gate else "Not run"),
                help="Agreement with the human-coded sample. See the Release gate tab.")

    if len(confidential):
        share = len(sens_non_en) / len(confidential)
        st.info(f"**{len(sens_non_en)} of {len(confidential)} comments routed to confidential review ({share:.0%}) were "
                f"written in a language other than English.**", icon="🔎")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### What people talk about, and how they feel about it")
        st.altair_chart(charts.sentiment_by_theme(substantive, LABELS), theme=None, width="stretch")
        with st.expander("Table view"):
            e = charts.explode_themes(substantive)
            tbl = e.groupby(["theme", "sentiment"]).size().unstack(fill_value=0)
            tbl.index = [LABELS.get(t, t) for t in tbl.index]
            st.dataframe(tbl, width="stretch")
    with right:
        st.markdown("##### Languages")
        st.altair_chart(charts.language_bars(substantive, LANGUAGE_NAMES), theme=None, width="stretch")
        drivers = charts.engagement_drivers(substantive, LABELS)
        if drivers is not None:
            st.markdown("##### Themes linked to lower engagement")
            st.altair_chart(drivers, theme=None, width="stretch")
            st.caption("Average engagement score of people who raised each theme, compared with everyone who commented.")

    if gcols:
        st.markdown("##### Where concerns concentrate")
        row = st.selectbox("Rows", gcols, index=gcols.index("site") if "site" in gcols else 0, key="heat_rows")
        st.altair_chart(charts.concern_heatmap(substantive, LABELS, row=row), theme=None, width="stretch")
        st.caption("Share of each group's comments raising a concern (negative or mixed) about each theme. "
                   "Groups with fewer than 10 comments are hidden.")

# ---------------------------------------------------------------------------------------------------------------- comments
with tab_comments:
    st.caption(f"{len(confidential)} confidential comments are not shown here; they are only in the Review queue.")
    f = st.columns(5)
    themes_sel = f[0].multiselect("Theme", [t.id for t in DEFAULT_THEMES], format_func=LABELS.get)
    sent_sel = f[1].multiselect("Sentiment", ["negative", "mixed", "neutral", "positive"])
    urg_sel = f[2].multiselect("Urgency", ["critical", "high", "medium", "low"])
    lang_sel = f[3].multiselect("Language", sorted(public["language"].unique()), format_func=lang_name)
    grp_col = f[4].selectbox("Group by", ["(none)"] + gcols)
    grp_sel = st.multiselect(f"{grp_col}", sorted(public[grp_col].dropna().unique())) if grp_col != "(none)" else []
    view = public
    if themes_sel:
        view = view[view["themes"].apply(lambda ts: any(t in ts for t in themes_sel))]
    if sent_sel:
        view = view[view["sentiment"].isin(sent_sel)]
    if urg_sel:
        view = view[view["urgency"].isin(urg_sel)]
    if lang_sel:
        view = view[view["language"].isin(lang_sel)]
    if grp_sel:
        view = view[view[grp_col].isin(grp_sel)]
    show = view.assign(
        Language=view["language"].map(lang_name),
        Themes=view["themes"].map(theme_text),
        Evidence=view.apply(lambda r: ("✓ " if r["evidence_grounded"] else "⚠ ") + str(r["evidence"]) if r["evidence"] else "", axis=1),
    )
    st.caption(f"{len(view):,} comments")
    st.dataframe(
        show[[c for c in ["comment_id", "site", "Language", "text_redacted", "summary_en", "Themes", "sentiment", "urgency",
                          "Evidence", "confidence"] if c in show.columns]],
        hide_index=True, width="stretch", height=520,
        column_config={
            "comment_id": st.column_config.TextColumn("ID", width="small"),
            "text_redacted": st.column_config.TextColumn("Comment (as written, personal details removed)", width="large"),
            "summary_en": st.column_config.TextColumn("English summary", width="medium"),
            "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "urgency": st.column_config.TextColumn("Urgency", width="small"),
            "Evidence": st.column_config.TextColumn("Evidence (✓ found verbatim)", width="medium"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f", width="small"),
        },
    )
    st.download_button("Download these comments (CSV)", show.drop(columns=["text"], errors="ignore").to_csv(index=False).encode("utf-8"),
                       "pulse_comments.csv", "text/csv")

# ---------------------------------------------------------------------------------------------------------------- review queue
with tab_review:
    st.markdown("Comments that may involve a risk to someone's safety or wellbeing, or an allegation of misconduct. "
                "Visible to the HR / HSE review team only. **Original text is shown here** so reviewers can act; it is "
                "never included in manager briefs.")
    if confidential.empty:
        st.success("Nothing was routed to confidential review.")
    else:
        q = confidential.copy()
        q["category"] = q.apply(lambda r: r["sensitive_category"] if r["sensitive_category"] != "none" else review_category(r["review_reasons"]), axis=1)
        q["rank"] = q.apply(lambda r: severity_rank(r["sensitive_category"], r["urgency"], r["review_reasons"]), axis=1)
        q = q.sort_values("rank")

        where = "team" if "team" in q.columns else ("site" if "site" in q.columns else None)
        if where:
            clusters = q.groupby([where, "category"]).size().reset_index(name="comments")
            clusters = clusters[clusters["comments"] >= 2].sort_values("comments", ascending=False)
            if len(clusters):
                st.markdown("##### Repeated reports from the same group")
                if len(clusters) > 4:
                    st.caption(f"Showing the 4 largest of {len(clusters)} groups with more than one report.")
                for _, c in clusters.head(4).iterrows():
                    extra = ""
                    if where == "team" and "site" in q.columns:
                        extra = f" ({q[q['team'] == c['team']]['site'].iloc[0]})"
                    st.warning(f"**{c['comments']} comments** about **{SENSITIVE_LABELS.get(c['category'], c['category']).lower()}** "
                               f"from {where} **{c[where]}**{extra}. Independent reports from one group usually warrant an investigation.",
                               icon="⚠️")

        cats = ["All"] + sorted(q["category"].unique(), key=lambda c: SENSITIVE_LABELS.get(c, c))
        cat = st.segmented_control("Category", cats, default="All", format_func=lambda c: SENSITIVE_LABELS.get(c, c) if c != "All" else "All")
        if cat and cat != "All":
            q = q[q["category"] == cat]
        status_opts = ["New", "Escalated to HSE", "Escalated to HR", "Under investigation", "Dismissed (not sensitive)"]
        for _, r in q.iterrows():
            with st.container(border=True):
                head = st.columns([5, 2])
                meta = " · ".join(str(r[c]) for c in ("country", "site", "team") if c in r and pd.notna(r[c]))
                head[0].markdown(
                    f"{badge(SENSITIVE_LABELS.get(r['category'], r['category']), 'red')} "
                    f"{badge('Urgency: ' + r['urgency'], URGENCY_BADGE.get(r['urgency'], 'gray'))} "
                    f"{badge(lang_name(r['language']), 'gray')} "
                    f"{badge('second look', 'blue') if r.get('rechecked') else ''}  \n"
                    f"<span class='pulse-muted'>{r['comment_id']} · {meta}</span>", unsafe_allow_html=True)
                text = str(r["text"])
                ev = str(r.get("evidence") or "")
                safe = html.escape(text)
                if ev and html.escape(ev) in safe:
                    safe = safe.replace(html.escape(ev), f"<span class='pulse-evidence'>{html.escape(ev)}</span>", 1)
                st.markdown(f"<div class='pulse-orig' dir='auto'>{safe}</div>", unsafe_allow_html=True)
                if r.get("summary_en"):
                    st.markdown(f"<span class='pulse-muted'>English summary: {html.escape(str(r['summary_en']))}</span>",
                                unsafe_allow_html=True)
                why = []
                for reason in r["review_reasons"]:
                    if reason.startswith("model:"):
                        why.append("model flagged " + SENSITIVE_LABELS.get(reason[6:], reason[6:]).lower())
                    elif reason.startswith("keyword:"):
                        why.append("keyword backstop: " + SENSITIVE_LABELS.get(reason[8:], reason[8:]).lower())
                    elif reason == "critical_urgency":
                        why.append("critical urgency")
                st.caption("Why it is here: " + "; ".join(why))
                with head[1]:
                    d = ss.decisions.get(r["comment_id"], {"status": "New", "note": ""})
                    new_status = st.selectbox("Status", status_opts, index=status_opts.index(d["status"]),
                                              key=f"st_{r['comment_id']}", label_visibility="collapsed")
                    note = st.text_input("Note", value=d["note"], key=f"nt_{r['comment_id']}", placeholder="Reviewer note",
                                         label_visibility="collapsed")
                    ss.decisions[r["comment_id"]] = {"status": new_status, "note": note}
        log = pd.DataFrame([{"comment_id": k, **v} for k, v in ss.decisions.items()])
        st.download_button("Download review log (CSV)", log.to_csv(index=False).encode("utf-8"), "pulse_review_log.csv", "text/csv")

    qa = res[res["needs_review"].astype(bool) & ~res["confidential"].astype(bool)]
    if len(qa):
        with st.expander(f"Quality checks ({len(qa)}): low confidence, unverified evidence or processing errors"):
            st.dataframe(qa.assign(reasons=qa["review_reasons"].map(", ".join))[
                ["comment_id", "text_redacted", "summary_en", "confidence", "reasons", "error"]], hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------------------------------- release gate
with tab_valid:
    st.markdown("Before results go to managers, Pulse compares its tags with comments that a person coded by hand. "
                "If agreement is below the thresholds, manager briefs stay blocked.")
    gsrc = st.radio("Human-coded sample", [BUNDLED, "Upload"], horizontal=True, label_visibility="collapsed", key="gold_src")
    if gsrc == "Upload":
        st.file_uploader("CSV with comment_id, themes, sentiment, urgency, sensitive_category", type=["csv"], key="gold_up")
        st.download_button("Download a blank coding sheet for this run", gold_template(res).to_csv(index=False).encode("utf-8"),
                           "pulse_coding_sheet.csv", "text/csv", help="A language-stratified sample to code by hand.")
    with st.expander("Thresholds"):
        tc = st.columns(4)
        tc[0].slider("Theme F1", 0.5, 0.95, 0.70, 0.05, key="th_theme")
        tc[1].slider("Sentiment kappa", 0.4, 0.9, 0.60, 0.05, key="th_sent")
        tc[2].slider("Urgency weighted kappa", 0.4, 0.9, 0.60, 0.05, key="th_urg")
        tc[3].slider("Sensitive recall", 0.8, 1.0, 0.95, 0.01, key="th_sens")
    if gold_error:
        st.error(f"Could not read the human-coded file: {gold_error}")
    elif rep is None:
        st.info("Add a human-coded sample to run the gate.")
    elif rep.n == 0:
        st.warning("None of the human-coded comment ids are in this run. Run Pulse on the same survey first.")
    else:
        if rep.passed:
            st.success(f"**Release gate passed** on {rep.n} human-coded comments. Manager briefs can be shared.", icon="✅")
        else:
            failed = [c.name for c in rep.checks if not c.passed]
            st.error(f"**Release gate failed**: {', '.join(failed)}. Manager briefs are blocked.", icon="⛔")
        m = st.columns(4)
        lo, hi = rep.themes_ci
        m[0].metric("Theme agreement (F1)", f"{rep.themes['micro_f1']:.2f}", help=f"95% CI {lo:.2f}–{hi:.2f}")
        lo, hi = rep.sentiment_kappa_ci
        m[1].metric("Sentiment (Cohen's κ)", f"{rep.sentiment_kappa:.2f}",
                    help=f"95% CI {lo:.2f}–{hi:.2f}; accuracy {rep.sentiment_accuracy:.0%}")
        lo, hi = rep.urgency_weighted_kappa_ci
        m[2].metric("Urgency (weighted κ)", f"{rep.urgency_weighted_kappa:.2f}",
                    help=f"95% CI {lo:.2f}–{hi:.2f}; within one level {rep.urgency_within_one:.0%}")
        m[3].metric("Sensitive comments routed", f"{rep.sensitive_recall_routed:.0%}" if rep.n_sensitive_gold else "n/a",
                    help=f"Of {rep.n_sensitive_gold} the coder marked sensitive. Model alone: {rep.sensitive_recall_model:.0%}")
        st.dataframe(pd.DataFrame([{"Check": c.name, "Result": "Pass" if c.passed else "Fail",
                                    "Value": round(c.value, 3), "Threshold": c.threshold, "Detail": c.detail}
                                   for c in rep.checks]), hide_index=True, width="stretch")

        a, b = st.columns([3, 2], gap="large")
        with a:
            st.markdown("##### Agreement by language")
            bl = rep.by_language.copy()
            st.dataframe(bl[["language_name", "n", "theme_f1", "sentiment_accuracy", "urgency_accuracy",
                             "sensitive_in_sample", "sensitive_routed"]].rename(columns={
                "language_name": "Language", "theme_f1": "Theme F1", "sentiment_accuracy": "Sentiment accuracy",
                "urgency_accuracy": "Urgency accuracy", "sensitive_in_sample": "Sensitive (human)",
                "sensitive_routed": "Sensitive routed"}),
                hide_index=True, width="stretch",
                column_config={"Theme F1": st.column_config.NumberColumn(format="%.2f"),
                               "Sentiment accuracy": st.column_config.NumberColumn(format="%.2f"),
                               "Urgency accuracy": st.column_config.NumberColumn(format="%.2f")})
            st.markdown("##### Agreement by theme")
            st.altair_chart(charts.theme_f1_bars(rep.per_theme, LABELS, th.theme_micro_f1), theme=None, width="stretch")
        with b:
            st.altair_chart(charts.confusion(rep.sentiment_confusion, "Sentiment"), theme=None, width="stretch")
            st.altair_chart(charts.confusion(rep.urgency_confusion, "Urgency"), theme=None, width="stretch")
        if len(rep.missed_sensitive):
            st.markdown("##### Sensitive comments that were not routed to review")
            st.dataframe(rep.missed_sensitive, hide_index=True, width="stretch")
        with st.expander(f"Disagreements with the human coder ({len(rep.disagreements)})"):
            st.dataframe(rep.disagreements, hide_index=True, width="stretch")

        others = [r for r in list_runs() if r["run_id"] != man.get("run_id")]
        if others:
            st.markdown("##### Compare with other runs on the same sample")
            opts = {r["run_id"]: f"{r['model']} · {r['created_utc'][:16].replace('T', ' ')}" for r in others}
            chosen = st.multiselect("Runs", list(opts), format_func=opts.get, max_selections=4)
            if chosen:
                reps = {f"{man.get('model')} (this run)": rep}
                for rid in chosen:
                    other, _ = load_run(rid)
                    reps[opts[rid]] = validate_cached(rid, gold_bytes, th_key, other)
                st.dataframe(compare(reps), hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                            ["theme micro F1", "sentiment kappa", "urgency weighted kappa",
                                             "sensitive recall", "weakest language F1"]})

        if not rep.passed:
            st.markdown("##### Override")
            st.checkbox("Release anyway (recorded in the export)", key="gate_override")
            st.text_input("Reason", key="gate_reason")
            if ss.get("gate_override") and not (ss.get("gate_reason") or "").strip():
                st.caption("A reason is required to override.")

# ---------------------------------------------------------------------------------------------------------------- briefs
with tab_briefs:
    gate = ss.gate or {}
    released = gate.get("passed") or gate.get("override")
    if not gate:
        st.warning("Run the release gate first (Release gate tab). Briefs can be previewed but not exported.", icon="⏳")
    elif not released:
        st.error("The release gate failed. Briefs can be previewed but not exported.", icon="⛔")
    elif gate.get("override"):
        st.warning(f"Released by override: {gate.get('reason')}", icon="⚠️")

    if not gcols:
        st.info("This survey has no grouping column (for example team or manager), so there is nothing to brief by.")
    else:
        c = st.columns([2, 2, 2, 3])
        by = c[0].selectbox("Brief for each", gcols, index=gcols.index("team") if "team" in gcols else 0)
        lang = c[1].selectbox("Brief language", BRIEF_LANGUAGES)
        min_n = c[2].number_input("Minimum comments per group", 3, 20, 5,
                                  help="Groups below this are not reported, to protect anonymity.")
        groups_all = sorted(res[by].dropna().astype(str).unique())
        only = c[3].multiselect("Only these groups (optional)", groups_all)
        use_model = settings.uses_model and settings.is_configured
        st.caption("Briefs are written by " + (f"`{settings.model}` from counts and English paraphrases only — never "
                   "original text, names or the language a comment was written in." if use_model else
                   "a fixed template (no model selected)."))
        if st.button("Write briefs", type="primary"):
            with st.spinner("Writing briefs…"):
                provider = ChatProvider(settings) if use_model else None
                ss.briefs = build_briefs(res, by, provider, language=lang, min_group_size=int(min_n),
                                         groups=only or None, max_workers=settings.max_workers)
                ss.briefs_by = by
        briefs = ss.briefs
        if briefs:
            ok = [b for b in briefs if b.status == "ok"]
            sup = [b for b in briefs if b.status != "ok"]
            st.caption(f"{len(ok)} briefs · {len(sup)} groups too small to report")
            mgr = {}
            if ss.get("briefs_by") == "team" and "manager_name" in res.columns:
                mgr = res.groupby("team")["manager_name"].first().to_dict()
            for b in ok:
                cnt = b.content
                title = f"{b.group}" + (f" · {mgr[b.group]}" if b.group in mgr else "") + f" · {b.n_comments} comments"
                with st.expander(title, expanded=len(ok) <= 3):
                    st.markdown(f"**{cnt.get('headline', '')}**")
                    x, y = st.columns(2)
                    with x:
                        st.markdown("**Top concerns**")
                        for item in cnt.get("concerns", []):
                            st.markdown(f"- *{LABELS.get(item.get('theme'), item.get('theme'))}*: {item.get('point', '')}")
                        if cnt.get("strengths"):
                            st.markdown("**What is working**")
                            for s_ in cnt["strengths"]:
                                st.markdown(f"- {s_}")
                    with y:
                        st.markdown("**Actions for the next 30 days**")
                        for i, a in enumerate(cnt.get("actions", []), 1):
                            st.markdown(f"{i}. {a.get('action', '')}  \n   *First step:* {a.get('first_step', '')}")
                    if cnt.get("conversation_starter"):
                        st.markdown(f"**Ask your team:** {cnt['conversation_starter']}")
                    st.caption(("Written by the model" if b.source == "model" else "Template brief") +
                               (" · checks: " + "; ".join(b.warnings) if b.warnings else " · all numbers match the team's statistics"
                                if b.source == "model" else ""))
            if sup:
                with st.expander(f"Not reported ({len(sup)} groups)"):
                    for b in sup:
                        st.markdown(f"- **{b.group}**: {b.note}")
            md = "\n".join(
                [f"# Pulse manager briefs\n\nRun {man.get('run_id')} · model {man.get('model')} · "
                 f"release gate: {'passed' if gate.get('passed') else 'override: ' + str(gate.get('reason')) if gate.get('override') else 'not passed'}\n"]
                + [b.to_markdown(LABELS, by.replace('_', ' ').title()) for b in briefs])
            st.download_button("Download all briefs (Markdown)", md.encode("utf-8"), "pulse_briefs.md", "text/markdown",
                               disabled=not released, help=None if released else "Blocked until the release gate passes.")

# ---------------------------------------------------------------------------------------------------------------- run details
with tab_run:
    c = st.columns(4)
    usage = man.get("usage") or {}
    stats = man.get("stats") or {}
    c[0].metric("Model calls", f"{usage.get('calls', 0):,}")
    c[1].metric("Tokens in / out", f"{usage.get('prompt_tokens', 0):,} / {usage.get('completion_tokens', 0):,}")
    cost = man.get("estimated_cost_usd")
    c[2].metric("Estimated cost", f"${cost:,.2f}" if isinstance(cost, (int, float)) else "n/a",
                help="List price per million tokens; check your billing page for actuals.")
    c[3].metric("Duration", f"{man.get('duration_s', 0):,.0f} s")
    st.markdown("##### What happened to each comment")
    st.dataframe(pd.DataFrame([
        {"Step": "Empty or 'no comment' (no model call)", "Comments": stats.get("skipped", 0)},
        {"Step": "Answered from cache (already analysed)", "Comments": stats.get("cached", 0)},
        {"Step": "Tagged by the model", "Comments": stats.get("model", 0)},
        {"Step": "…of which identical text shared one call", "Comments": stats.get("deduplicated", 0)},
        {"Step": "Keyword baseline (no model)", "Comments": stats.get("baseline", 0)},
        {"Step": "Second look at flagged comments, one at a time", "Comments": stats.get("rechecked", 0)},
        {"Step": "…where the second look changed a label", "Comments": stats.get("recheck_changed", 0)},
        {"Step": "Failed (kept for manual review)", "Comments": stats.get("failed", 0)},
    ]), hide_index=True, width="stretch")
    st.markdown("##### Privacy")
    st.markdown(
        f"- Comment text sent outside your network: **{'yes, to ' + str(man.get('endpoint_host')) if man.get('data_leaves_network') else 'no'}**\n"
        f"- Comments with personal details removed before analysis: **{man.get('n_redacted', 0)}** "
        f"(names from {', '.join(man.get('columns', {}).get('names_redacted_from') or []) or 'no columns'}, phone numbers, emails, IDs)\n"
        f"- Manager briefs use counts and English paraphrases only; groups under the minimum size are not reported.")
    red = res[res["text"] != res["text_redacted"]].head(3)
    if len(red):
        with st.expander("Example of what the model received"):
            for _, r in red.iterrows():
                st.markdown(f"<div dir='auto' class='pulse-muted'>Original: {html.escape(str(r['text']))}</div>"
                            f"<div dir='auto'>Sent: {html.escape(str(r['text_redacted']))}</div><hr/>", unsafe_allow_html=True)
    if man.get("provider_notes"):
        st.markdown("##### Endpoint notes")
        for n in man["provider_notes"]:
            st.markdown(f"- {n}")
    with st.expander("Full run manifest"):
        st.json(man)
    st.download_button("Download all results (JSON lines)", res.to_json(orient="records", lines=True, force_ascii=False).encode("utf-8"),
                       f"pulse_{man.get('run_id', 'run')}.jsonl", "application/json")
