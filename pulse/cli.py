"""Command-line runner.

    python -m pulse.cli models                        # Meta models installed in Ollama on this computer
    python -m pulse.cli check                         # test the model connection
    python -m pulse.cli run --scope sample            # tag the human-coded sample only (fast)
    python -m pulse.cli run                           # tag the whole survey
    python -m pulse.cli validate                      # score the latest run against the human-coded sample
    python -m pulse.cli briefs --by team --language Arabic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .briefs import build_briefs
from .config import BACKENDS, from_env
from .pipeline import ColumnMap, list_runs, load_run, load_survey, run_survey
from .providers import AuthError, ChatProvider
from .taxonomy import DEFAULT_THEMES, theme_labels
from .validation import load_gold, validate

ROOT = Path(__file__).resolve().parent.parent


def _progress(done: int, total: int, msg: str) -> None:
    sys.stderr.write(f"\r{msg:<60}")
    if done >= total:
        sys.stderr.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pulse", description="Multilingual survey-comment analysis with Muse Spark.")
    ap.add_argument("--backend", choices=BACKENDS, help="Overrides PULSE_BACKEND")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="Send one small request to the configured model")
    sub.add_parser("models", help="List models installed in Ollama on this computer")
    r = sub.add_parser("run", help="Analyse a survey export")
    r.add_argument("--input", default=str(ROOT / "data" / "synthetic_survey.csv"))
    r.add_argument("--text-col")
    r.add_argument("--id-col")
    r.add_argument("--name-cols", default="", help="Comma-separated columns holding people's names to redact")
    r.add_argument("--scope", choices=["all", "sample"], default="all", help="'sample' = only the human-coded comments")
    r.add_argument("--limit", type=int, default=0, help="Analyse a random subset of this many comments")
    r.add_argument("--gold", default=str(ROOT / "data" / "human_coded_sample.csv"))
    v = sub.add_parser("validate", help="Score a run against a human-coded sample")
    v.add_argument("--run", help="Run id (default: latest)")
    v.add_argument("--gold", default=str(ROOT / "data" / "human_coded_sample.csv"))
    b = sub.add_parser("briefs", help="Write manager briefs for a run")
    b.add_argument("--run", help="Run id (default: latest)")
    b.add_argument("--by", default="team")
    b.add_argument("--language", default="English")
    b.add_argument("--out", default="outputs/briefs.md")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows consoles may not be UTF-8
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    if args.cmd == "models":
        from .local import RECOMMENDED, installed_models

        base = from_env("local").base_url
        models = installed_models(base)
        if models is None:
            print(f"Ollama is not running at {base}. Install it from https://ollama.com and open it.")
            return 1
        for m in models:
            print(f"  {m.name:<28} {m.parameters:>8} {m.size_gb:>6} GB  {'Meta model' if m.is_meta else ''}")
        if not any(m.is_meta for m in models):
            print("No Meta model installed yet. Pick one:")
            for tag, desc, size, needs in RECOMMENDED:
                print(f"  ollama pull {tag:<20} {desc}, {size}. Needs: {needs}")
        return 0

    settings = from_env(args.backend)

    if args.cmd == "check":
        if not settings.uses_model or not settings.is_configured:
            print("No model found. To run without an API key: install Ollama (https://ollama.com), run "
                  "`ollama pull llama3.1:8b`, then try again. Or put MODEL_API_KEY in .env for Meta Model API.")
            return 1
        print(f"Backend {settings.backend} · model {settings.model} · {settings.base_url}")
        try:
            print("OK:", ChatProvider(settings).check())
        except AuthError:
            print("The endpoint rejected the API key (HTTP 401/403). Check MODEL_API_KEY.")
            return 1
        except Exception as err:  # noqa: BLE001
            print(f"Request failed: {err}")
            return 1
        return 0

    if args.cmd == "run":
        df = load_survey(args.input)
        cols = ColumnMap.guess(df)
        if args.text_col:
            cols.text = args.text_col
        if args.id_col:
            cols.id = args.id_col
        if args.name_cols:
            cols.names = [c.strip() for c in args.name_cols.split(",") if c.strip()]
        if args.scope == "sample":
            ids = set(load_gold(args.gold)["comment_id"])
            df = df[df[cols.id].isin(ids)]
        if args.limit:
            df = df[df[cols.text].str.strip() != ""]
            df = df.sample(min(args.limit, len(df)), random_state=1)
        print(f"Model: {settings.model} ({settings.backend}) · comments: {int((df[cols.text].str.strip() != '').sum())}",
              file=sys.stderr)
        res, man = run_survey(df, settings, cols, progress=_progress, source_name=Path(args.input).name)
        print(json.dumps({k: man[k] for k in ("run_id", "model", "n_comments", "n_confidential", "stats", "usage",
                                              "estimated_cost_usd", "duration_s", "provider_notes")}, indent=2))
        return 0

    runs = list_runs()
    if not runs:
        print("No saved runs. Use `run` first.")
        return 1
    run_id = args.run or runs[0]["run_id"]
    res, man = load_run(run_id)

    if args.cmd == "validate":
        rep = validate(res, load_gold(args.gold), theme_labels=[t.id for t in DEFAULT_THEMES])
        print(f"Run {run_id} · {man['model']}")
        for c in rep.checks:
            print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name}: {c.value:.3f} (threshold {c.threshold}) {c.detail}")
        print(rep.by_language[["language_name", "n", "theme_f1", "sentiment_accuracy", "sensitive_routed"]].to_string(index=False))
        print("RELEASE GATE:", "PASSED" if rep.passed else "FAILED")
        return 0 if rep.passed else 2

    if args.cmd == "briefs":
        provider = ChatProvider(settings) if settings.uses_model else None
        briefs = build_briefs(res, args.by, provider, language=args.language, min_group_size=settings.min_group_size)
        labels = theme_labels(DEFAULT_THEMES)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(b.to_markdown(labels, args.by.title()) for b in briefs), encoding="utf-8")
        print(f"Wrote {sum(b.status == 'ok' for b in briefs)} briefs to {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
