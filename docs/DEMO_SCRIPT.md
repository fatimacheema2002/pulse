# Pulse: 3-minute demo script

Judging weights: Impact 30%, Use of Meta AI tooling 25%, Demo 25%, Technical depth 20%.

## Before you record (10 minutes)

1. Open Ollama, then `streamlit run app.py` in `~/Projects/pulse`.
2. In the sidebar, open the full-survey model run **20261030-181617-241e**.
3. Check three numbers on screen and say them as you see them: the "Not written in English" figure, the blue banner about sensitive comments, and the largest Kuwait cluster in the Review queue.
4. Pick one Urdu, one Roman Urdu and one Arabic comment in the Review queue. In Manager briefs, write the brief for one Jubail team in Arabic once, so it appears instantly on camera.
5. Close the Columns panel, zoom the browser to 110%, hide notifications. Record at 1440×900 or larger.

## Script

**0:00–0:20 · The problem**
"In my last role I co-led an engagement survey at an engineering company: 3,000 people, 9 countries, 4 languages. We read thousands of comments by hand, and most never made it into the findings. The ones we lost were the ones not written in English."

**0:20–0:45 · Run it**
Show the sidebar: *On this computer, no API key*, Llama 3.1 8B selected, and the green "never leaves this machine" note. Start a short run so the progress bar moves, then switch to the finished full run.
"Pulse runs Meta's Llama 3.1 right here on my MacBook Air. No API key, no cloud: survey comments are confidential, so they never leave the machine. It read all 1,302 comments of this synthetic survey in under two hours, in the language each was written in, with no translation step."

**0:45–1:15 · What it found**
Overview tab. Point at "Not written in English" and the blue banner.
"[Seven in ten] comments aren't in English, and so are [most] of the safety and harassment reports." Then the heatmap: "Accommodation complaints at the Ras Laffan camp stand out far above every other site."

**1:15–1:50 · Nobody falls through the cracks**
Review queue. Show the Kuwait cluster: "[N] people on one team independently describe a supervisor shouting and threatening to send them home." Open an Urdu safety comment, then a Roman Urdu one. Point at the highlighted evidence quote in the original script and the *second look* badge.
"Fifty comments went to this confidential queue. Anything sensitive gets a second, closer read, and an independent keyword backstop catches it even if the model doesn't. None of this reaches a manager."

**1:50–2:25 · Proving it before anyone acts on it**
Release gate tab. Show PASS, the four headline numbers, agreement by language, then the comparison with the keyword baseline.
"Before results go out, Pulse checks itself against comments a person coded by hand. Keyword rules fail this gate on tone: across the 404 Urdu comments they get sentiment right 46% of the time; Llama gets 86%. And the gate earned its place: my first run failed it, missing five discrimination reports buried in a second sentence. I fixed the backstop, and it now routes all 35 sensitive comments in the sample."

**2:25–2:50 · What managers get**
Manager briefs, the Jubail team in Arabic.
"Each manager gets three actions for the next 30 days, in their own language, written from counts and paraphrases only. Small teams don't get a brief, so nobody can be identified, and every number is checked against the data."

**2:50–3:00 · Close**
"Pulse: every comment read, every risk routed, nothing released until it's checked. Built on Meta's Llama, running on a laptop."

## Recording tips

- Keep the mouse still while talking. Move it only to point.
- If anything is slow on camera, cut. Don't wait.
- Say once that the survey is synthetic ("a synthetic survey I built in four languages").
- Replace every [bracketed] phrase with what you see on screen.
