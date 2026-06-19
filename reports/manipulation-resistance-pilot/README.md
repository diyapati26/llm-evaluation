# Manipulation Resistance in LLMs — Pilot Study

A controlled pilot measuring whether language models abandon a correct, well-reasoned
answer under social pressure, across **7 models / 3 families**.

| File | What it is |
|------|------------|
| **`report.pdf`** | Research-paper-style writeup — abstract, methodology, figures, findings. Share this. |
| **`report.html`** | Same report, self-contained single HTML (charts embedded as inline SVG, no external deps — opens in any browser). |
| **`tables.md`** | The raw analysis tables (every per-model / per-attack number). |
| **`results.json`** | The structured analysis output the figures are generated from. |

## At a glance

- **Run:** `cheaper-pilot-20260619T003531Z` · seed 42 · single judge Claude Opus 4.8 · ~$32.67 compute.
- **Models:** GPT-5.4-nano / -mini / GPT-5.5 · Claude Haiku 4.5 / Sonnet 4.6 · Gemini 3.1 Flash-Lite / 3.5 Flash.
- **Modes:** stateless, stateful, repeat, gauntlet (29-turn), drift + standard benchmark + moral/empathy.

## Headline findings (pilot, n≈10 — directional)

1. **Capability → robustness, significantly** (McNemar p<0.001): top tier (Sonnet 4.6, GPT-5.5, Gemini 3.5 Flash) resist ~98–100%; lite tier folds 30–40%.
2. **The ceiling is cheap** — the three strongest models are statistically tied yet ~6× apart in price; GPT-5.5's premium buys no extra resistance.
3. **Attack potency is lopsided** — confident assertion + fabricated evidence drive almost all folds; emotional/consensus appeals are near-inert.
4. **Length is an attack vector for small models** — they erode under repetition and the long gauntlet even when they pass single-turn tests.

See `report.pdf` §5 for limitations (pilot sample, single judge, 4/3,353 excluded trials).
