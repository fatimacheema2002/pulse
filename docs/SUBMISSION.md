# Submission

Paste into the Cerebral Valley submission form. Check the track line, then add your repo and video links.

**Project name:** Pulse

**Track:** Indie developer track

**Theme:** Build for Your World

**One line:** Pulse reads every engagement-survey comment in the language it was written in, using Meta's Llama 3.1 on a laptop with no API key, routes safety and harassment reports to people, and blocks results from going out until they agree with a human-coded sample.

## Problem

Multinational employers survey thousands of people across countries and languages. Open-text comments are the most useful part of a survey and the least used. Analysts read them by hand, in whatever languages the team can read, and run out of time. I co-led a survey of 3,000 employees across 9 countries in 4 languages, and most comments never reached the findings. The ones that were lost were disproportionately not in English, and that is where safety and fairness concerns often sit.

## What Pulse does

- **Tags every comment in its original language** (Arabic, Urdu including Roman Urdu, Hindi, English) with themes, sentiment, urgency, a sensitive-issue category, an English summary, and a verbatim evidence quote that Pulse checks really appears in the comment.
- **Routes sensitive comments** (safety hazards, harassment, discrimination, wellbeing risk, misconduct, retaliation) to a confidential review queue. Both the model and an independent multilingual keyword backstop can trigger this. Repeated reports from one team are surfaced as a cluster. None of these comments reach a manager.
- **Checks itself before release**: theme F1, sentiment kappa, weighted urgency kappa, sensitive-comment recall and per-language agreement against a human-coded sample, with bootstrap confidence intervals. Manager briefs stay blocked until the gate passes.
- **Writes manager action briefs** in the manager's language, from counts and English paraphrases only, with an anonymity threshold and a check that every number matches the data.

## Use of Meta AI tooling

- Runs **Llama 3.1 8B** locally through Ollama on a MacBook Air (M5, 16 GB). No API key, no cloud: confidential survey comments never leave the machine.
- Uses Ollama's native API to raise the context window (the 4,096-token default silently truncates prompts) and to enforce a JSON schema generated from the theme taxonomy by constrained decoding, so the model always returns valid, complete tags. The full survey ran with zero failed comments.
- No translation step. The model judges tone in the original script, and the evidence quote stays in that script so a reviewer can verify it.
- Sensitive, critical, low-confidence and unverifiable items get a second, single-comment read: 59 comments in the full survey.
- The same code runs Muse Spark through Meta Model API, or Muse Glimmer / Muse Spark Open on a vLLM or SGLang server, and the release gate compares any of them on the same human-coded sample. Muse Glimmer 30B was out of reach on a 16 GB laptop.

## Results

**Release gate on the 230-comment human-coded sample**

| | Theme F1 | Sentiment kappa | Urgency weighted kappa | Sensitive routed | Gate |
|---|---|---|---|---|---|
| Llama 3.1 8B (local) | 0.83 | **0.76** | 0.80 | 35 / 35 | **Pass** |
| Keyword baseline | 0.86 | 0.57 | 0.86 | 35 / 35 | Fail (sentiment) |

Theme F1 by language for Llama 3.1 8B: Arabic 0.82, English 0.84, Urdu 0.82, Hindi 0.85.

**Full survey (1,302 comments, scored against the synthetic answer key)**

| Sentiment accuracy | Urdu | Arabic | Hindi | English |
|---|---|---|---|---|
| Llama 3.1 8B | **86%** | **83%** | **81%** | 80% |
| Keyword baseline | 46% | 66% | 72% | 80% |

Keyword rules do as well as the model on English comments but read tone poorly in the other languages, worst in Urdu, the largest language group (404 comments). On themes the keyword rules score slightly higher, because the synthetic comments are built from a fixed set of sentences that keywords match easily; real comments will be messier.

The full survey took 105 minutes on the laptop at $0: 816 comments tagged by the model, 171 identical comments sharing one call, 88 empty comments skipped, 50 routed to confidential review, 0 failures.

**The gate earned its place.** The first model run failed it: 30 of 35 sensitive comments routed. All five misses were the same discrimination report ("workers of certain nationalities get better rooms") written as a second sentence after another topic. I added nationality phrases to the keyword backstop and the next run routed 35 of 35. Because that fix was found on the same sample, the sample recall figure is optimistic; a fresh human-coded sample is the honest test on real data.

## Technical notes

- Python, Streamlit, pandas, Altair. One HTTP client for Ollama, Meta Model API and OpenAI-compatible servers, with automatic fallback when a server rejects structured output, reasoning effort or temperature, plus retries with backoff.
- SQLite cache keyed by model, prompt version and taxonomy, so interrupted runs resume and nothing is analysed twice. Identical comments in a run share one call.
- Personal details (names from the survey's own manager columns, phone numbers, emails, IDs) are removed before any text reaches a model.
- 69 automated tests, including fake Ollama and OpenAI-compatible servers that inject malformed output, missing ids, rate limits, missing models and rejected parameters.
- Synthetic multilingual survey with known labels, built for this project: 2,853 respondents, 1,302 comments, 9 countries, 62 teams, planted site-level patterns. No real employee data.

## What's next

Pilot with a real survey and its own human-coded sample, add roles and access control for the review queue, try Muse Glimmer on a larger machine, and support more languages (Tagalog, Malay, Bengali) common in the same workforces.

## Built during the hackathon

All code and data in this repository were created between 26 October and 4 November 2026.
