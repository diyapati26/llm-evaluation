# CLAUDE.md — project context for AI agents

Read this first. It tells a new Claude what this repo is, what was built and why,
how to run and extend it, and the rules that keep it correct.

## What this is

A **reproducible LLM evaluation framework** for a research paper on **manipulation
resistance** — *does a model abandon a correct, well-reasoned answer under social
pressure?* It runs three modules: **manipulation resistance** (5 modes), a
**standard benchmark** (MMLU / HellaSwag / TruthfulQA), and **moral/empathy**
(LLM-as-judge), across OpenAI / Anthropic / Gemini / Groq / OpenRouter.

## Ground rules (important)

- **`latest/` is the source of truth.** All real code lives there. Only `README.md`
  and `CLAUDE_EFFORT.md` sit at the repo root.
- **`Archive/` is retired** — two old prototypes (`evals/`, `Simpler Arch/`) kept for
  reference only. **Do not run, import, or edit them.**
- **Run from the repo root** as `python -m latest.…`. Run outputs go to
  `latest/runs/` (gitignored). Secrets in `latest/.env` (gitignored).
- **Architecture = ledger-first:** `plan` (freeze the design) → `collect` (call APIs,
  append one `fsync`'d `CallRecord` per call to `ledger.jsonl`) → `analyze` (pure,
  network-free: ledger → tables/CIs/tests). Collection produces facts; analysis
  produces findings; they never share code. Every number traces to a raw call.

## Goals → how they were achieved

| Goal | How it was achieved |
|---|---|
| Review the two prototypes, pick a direction | 32-agent analysis workflow → `latest/ARCHITECTURE_REVIEW.md`; verdict: consolidate, retire `Archive/`, go ledger-first |
| A publication-grade rewrite | Built `latest/` layer by layer (config → records → ledger/cache → providers → plan → collect → analyze → CLI), each verified before the next |
| Cover everything | 3 modules; manipulation 2×2 of modes (`stateless`/`stateful`/`repeat`/`gauntlet`) + `drift`; benchmark 4 datasets; moral with ≥2 judges + Cohen's κ |
| Reproducibility | content-addressed cache (re-runs ~free), append-only fsync'd ledger + resume, `manifest.json` provenance, `verify-ledger`, pinned dataset revisions + locked model snapshots (`lock-snapshots`) |
| Robustness | tenacity retry; budget gate that asks the user (resume/stop); preflight (modules → keys → offline tests → live connectivity) |
| Prove it works | a full smoke run (haiku + gpt-5.4-nano, all modules) produced a real report; offline test suite is green |
| Harden it | 33-agent adversarial review → 28 issues confirmed & fixed (log in `latest/CODEBASE.md` §14) |

## How to run

```bash
pip install -r latest/requirements.txt
cp latest/.env.example latest/.env        # fill in API keys

python -m latest.main                     # interactive: tests? → smoke? → comprehensive?
python -m latest.main --smoke             # tiny run across everything (config/latest/config/smoke.yaml)
python -m latest.main --comprehensive     # full run (config/eval.yaml)
python -m latest.cli analyze --run-id <id>        # re-analyze a run (no network)
python -m latest.cli verify  --run-id <id>        # ledger integrity + provenance gate
python -m latest.cli lock-snapshots               # pin dataset SHAs + model snapshots
```

Tests (offline, no keys; also run by `main.py` preflight):
```bash
python -m pytest latest/tests/test_offline.py latest/tests/test_plan.py \
                 latest/tests/test_analysis.py latest/tests/test_infra.py -q
```

## Config (all in `latest/config/`)

`eval.yaml` (the full run: models, judges, modules, n, modes, arms, subjects) ·
`smoke.yaml` (`--smoke` overrides) · `pricing.yaml` (authoritative cost) ·
`datasets.yaml` (HF ids + pinned revisions) · `snapshots.lock.yaml` (alias → snapshot).

## Where to read more

- `README.md` — overview + why `Archive/` vs `latest/`.
- `latest/CODEBASE.md` — **file-by-file, function-by-function** deep dive + review log.
- `latest/ARCHITECTURE_REVIEW.md` — analysis of the prototypes + the science to-dos.
- `CLAUDE_EFFORT.md` — the build session log (timeline, tokens, decisions, handoff).

## Conventions for editing

- Keep collection and analysis separate; analysis stays **pure** (no network) and
  unit-tested on fixtures.
- `CallRecord.call_id` is the **cache key** (content address) — shared across trials
  with identical requests; the *logical turn* identity is
  `(model_alias, trial_id, turn_index, judged_model, role, condition)`.
- Provider model strings are `provider:model`; `CallRecord.model_alias` is the **bare**
  model — aggregate/report lookups use the bare alias.
- New model = a line in `eval.yaml`; new provider/attack/mode/scorer = see
  `latest/CODEBASE.md` §12. Add a test for any new pure function.

## Current state & pre-publication to-dos

Working and green; `validate()` clean (revisions pinned, snapshots locked). Before the
paper: run the full `--comprehensive` matrix at larger `n` (the smoke had `nano`
all-invalid on 2 hard questions, so no McNemar pairs there); validate the LLM judge
against human labels; apply the statistics hardening in `latest/ARCHITECTURE_REVIEW.md`
(power, CIs, multiple-comparison correction, selection-bias handling).
