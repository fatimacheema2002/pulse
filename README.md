# Pulse

**Every engagement-survey comment, read in the language it was written in, before results go out.**

Pulse uses Meta's models to read open-text survey comments in Arabic, Urdu (including Urdu typed in Latin letters), Hindi and English. It tags each comment by theme, sentiment and urgency. Safety, harassment and other sensitive reports go to a confidential human review queue. Each manager gets a short action brief. Nothing is released until the model's tags agree with a human-coded sample.

**It runs entirely on your own computer with Meta's open-weights models (Muse Glimmer or Llama, through Ollama). No API key, no account, and no survey comment ever leaves the machine.** It can also use Muse Spark through Meta Model API when you have a key.

Built for the **Meta Global AI Developer Hackathon 2026** · Theme: **Build for Your World**.

## Why

Engagement surveys are how large employers find out what is going wrong, and the open-text comments are where the detail is. In a multilingual frontline workforce most of those comments are not written in English, so they are read by hand, skimmed or dropped, and the ones that get lost are often where the safety and fairness problems sit.

## What it does

| Step | What happens |
|---|---|
| **Redact** | Names (taken from the survey's own manager columns), phone numbers, emails, IDs and links are removed before any text reaches a model. The original stays on your machine. |
| **Tag** | The model reads each comment in its original language and returns themes (from a configurable taxonomy), sentiment, urgency, a sensitive-issue category, a neutral English summary and a **verbatim evidence quote**. Pulse checks the quote really appears in the comment. |
| **Second look** | Anything sensitive, critical, low-confidence or with an unverifiable quote is re-read on its own (with a higher reasoning effort when the model supports it). |
| **Route** | Sensitive comments go to a confidential review queue, triggered by the model **or** by an independent multilingual keyword backstop. Repeated reports from one team are surfaced as clusters. These comments never reach a manager's brief. |
| **Release gate** | Tags are compared with a human-coded sample: theme F1, Cohen's κ for sentiment, weighted κ for urgency, recall on sensitive comments, and agreement per language, with bootstrap confidence intervals. If a threshold fails, briefs are blocked (an override needs a written reason). |
| **Brief** | One short action brief per team, in English, Arabic, Urdu or Hindi. The model sees only counts and English paraphrases. Groups under the anonymity minimum get no brief, and every number in a brief is checked against the team's statistics. |

## How it uses Meta's models

- **On your computer, no API key (default):** Muse Glimmer 30B or Llama 3.1 8B through [Ollama](https://ollama.com). Pulse talks to Ollama's native API so it can raise the context window (Ollama's 4,096-token default silently cuts long prompts) and enforce the output JSON schema by constrained decoding, which keeps even small models from producing broken output.
- **Meta Model API (optional):** Muse Spark 1.3 at `https://api.meta.ai/v1`, with two reasoning budgets: `low` for every comment, `high` for the second look.
- **Open weights on a server (optional):** any OpenAI-compatible server (vLLM, SGLang, Baseten, Fireworks, Modal) running Muse Glimmer or Muse Spark Open.
- **Structured output** from a JSON schema generated from the taxonomy. If an endpoint rejects a parameter, the client steps down (schema, then JSON mode, then plain prompting) and remembers what worked.
- **Multilingual by design:** no translation step. The evidence quote stays in the original script, so a reviewer can verify it.
- **The release gate decides whether a model is good enough.** A small local model may not be, especially for Arabic and Urdu, and the gate will say so before anyone acts on its output.

## Quick start: on your computer, no API key

**1. Install Ollama and a Meta model.** Download Ollama from [ollama.com](https://ollama.com) (Windows, Mac or Linux) and open it. Then in a terminal:

| Your computer | Command | Download |
|---|---|---|
| NVIDIA GPU with 24 GB+ memory, or a Mac with 32 GB+ | `ollama pull muse-glimmer:30b` | about 18 GB |
| Any laptop with 16 GB RAM | `ollama pull llama3.1:8b` | about 5 GB |
| Older or 8 GB laptop | `ollama pull llama3.2:3b` | about 2 GB |

Muse Glimmer is Meta's newest open model but is very slow without a big GPU. If unsure, start with `llama3.1:8b`.

**2. Install and start Pulse.** With Anaconda on Windows, open *Anaconda Prompt* in the `pulse` folder:

```bat
conda create -n pulse python=3.11 -y
conda activate pulse
pip install -r requirements.txt
streamlit run app.py
```

Or with plain Python on any system:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**3. Run it.** Pulse finds Ollama and picks the best Meta model you have installed. Keep **Human-coded sample only** (230 comments) for the first run and click **Analyse comments**. The progress bar shows the time left. Finished comments are saved, so if you stop, the next run continues where it left off. The **Release gate** tab scores the run against the human-coded sample as soon as it finishes.

For long runs, the command line is more convenient than the browser:

```bash
python -m pulse.cli models                        # Meta models installed in Ollama
python -m pulse.cli check                         # one small request to the model
python -m pulse.cli run --scope sample            # the 230 human-coded comments
python -m pulse.cli run --limit 400               # a random 400 comments
python -m pulse.cli run                           # all 1,302 comments (about 840 unique)
python -m pulse.cli validate                      # release gate on the latest run (exit code 2 if it fails)
python -m pulse.cli briefs --by site --language Arabic
python -m pulse.cli --backend offline run         # keyword baseline, for comparison
```

Then open the saved runs in the dashboard.

## Other ways to run

**Meta Model API.** Put your key in `.env` as `MODEL_API_KEY` (copy `.env.example`) and choose *Meta Model API* in the sidebar. Hackathon participants get free credits.

**Open weights on a server.**

```bash
vllm serve meta-models/Muse-Glimmer-30B --port 8000     # on a GPU server inside your network
```

Then set `PULSE_BACKEND=open_weights`, `PULSE_OPEN_BASE_URL=http://your-gpu-host:8000/v1` and `PULSE_OPEN_MODEL=meta-models/Muse-Glimmer-30B`. Hosted providers work the same way; Pulse labels a run "data leaves your network" whenever the endpoint is not a private address.

**Keyword baseline.** Choose *Keyword baseline* to explore the dashboard without any model. It is also the baseline the models are measured against.

## Your own survey

Upload any UTF-8 CSV export in the sidebar. Pulse guesses the comment, id and name columns, and you can change them. Any other column (country, site, team, job level) becomes a filter and a grouping for briefs. To score a run on real data, download the **blank coding sheet** from the Release gate tab. It is a language-stratified sample. Have someone code it, then upload it back.

The theme taxonomy is in `pulse/taxonomy.py`. Edit it to match your organisation's engagement model; the prompt, the structured-output schema and the dashboard all follow it.

## The synthetic data

`data/` holds a synthetic survey: 2,853 respondents, 1,302 comments, 62 teams across 9 countries, written in English, Arabic, Urdu (one third of it in Roman Urdu) and Hindi. Every sentence was written by hand with its own labels, so the correct answer for every comment is known. Some patterns are planted on purpose: crowded camp accommodation in Qatar, safety hazards in Iraq, workload on an Abu Dhabi shutdown, delayed overtime in Jubail, and a cluster of harassment reports from one Kuwait team. `human_coded_sample.csv` is the coding an analyst would do by hand: 50 random comments per language plus 30 extra sensitive ones, so the safety-recall check has enough cases to mean something.

Regenerate it with `python scripts/generate_synthetic.py`. No real people, companies or responses are used.

**Keyword baseline on this data:** theme F1 0.86, sentiment κ 0.57 (below the 0.60 bar), and 35 of 35 sensitive comments routed. **The gate fails.**

## Privacy and safety choices

- By default nothing leaves your computer. If you use a cloud API, comment text is redacted before it is sent.
- Do not use the discounted `muse-spark-1.3-contributor` model with real employee data, because it allows Meta to train on prompts and completions.
- Confidential comments appear only in the review queue, with the original text for the people who must act on them.
- Briefs never contain original text, quotes, names, or the language a comment was written in, because a single Hindi speaker on a team would be identifiable.
- Groups below the minimum size (default 5) are not reported.

## Project layout

```
app.py                 Streamlit dashboard
pulse/providers.py     model client: Ollama (local), Meta Model API, open-weights servers; fallbacks, retries
pulse/local.py         finds Ollama and the installed Meta models
pulse/tagger.py        batching, validation, cache, de-duplication, second look
pulse/safety.py        multilingual keyword backstop and routing
pulse/validation.py    release gate: F1, kappa, recall, per-language, bootstrap CIs
pulse/briefs.py        manager briefs with anonymity threshold and number checks
pulse/redact.py        personal-detail removal
pulse/synthetic.py     synthetic survey generator with gold labels
pulse/cli.py           command line
tests/                 60+ tests; fake Ollama and OpenAI-compatible servers stand in for the model
```

Run the tests with `pytest`.

## Limitations

- The synthetic comments are built from about 50 sentences per language, so they are cleaner and more repetitive than real comments. Real surveys need their own human-coded sample, which is why the gate exists.
- The keyword backstop is deliberately simple. It is a safety net under the model, not a replacement for it.
- Its nationality phrases were added after the release gate showed Llama 3.1 8B missing discrimination by nationality when it came second in a comment, after another topic. They were found on the human-coded sample, so safety recall on that sample is optimistic; score a real survey on its own sample.
- Review decisions are kept for the session and exported as CSV. There is no user management yet.
- Llama 3.1 and 3.2 officially support Hindi and English but not Arabic or Urdu, so expect weaker results there; check the per-language rows of the release gate.
