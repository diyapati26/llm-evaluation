# latest — LLM Manipulation-Resistance Evaluation Framework

A reproducible, ledger-first framework for stress-testing LLMs **beyond accuracy**:
does a model abandon a correct, well-reasoned answer under social pressure? It runs
three eval modules — **manipulation resistance**, a **standard benchmark**, and
**moral/empathy** — across multiple providers, and produces every number in the
paper from a single auditable record of what each model actually said.

**`latest/` is the source of truth.** It is the clean rewrite and the only code you
should run. The repo root holds just this README and `CLAUDE_EFFORT.md`; everything
else — the package, `requirements.txt`, `pyproject.toml`, the deep-dive docs, and run
outputs — lives under `latest/`.

### Why `Archive/` was archived, and why `latest/` exists

`Archive/` contains the two original prototypes — `evals/` (class-based, async) and
`Simpler Arch/` (function-based) — kept for reference only. **Do not run them.** A
deep review (`latest/ARCHITECTURE_REVIEW.md`) found they were good experiments but
not publication-grade:

- **Two divergent stacks** computed the "same" metrics differently (regex vs typed
  answer parsing, different wrong-answer rules, population vs sample std), so neither
  was authoritative.
- The manipulation path **re-billed the API on every run** (no caching of multi-turn
  calls), and a crash could **lose hours of work** (no per-call durability, no resume).
- The significance test **never actually ran** (it imported `scipy.stats.mcnemar`,
  which doesn't exist), two of four scoring axes were **silently dead**, and analysis
  was tangled into a 763-line script that mixed collection, scoring, and printing.

`latest/` consolidates the good ideas (typed structured output, retry, central
pricing, the multi-turn `Conversation` abstraction) onto a **ledger-first**
architecture that fixes all of the above by construction: collection and analysis are
separate, every call is cached + durably logged, and every number traces to a raw
response. See `latest/ARCHITECTURE_REVIEW.md` for the full analysis and
`latest/CODEBASE.md` for the file-by-file guide.

---

## 1. The ideology (why it's built this way)

**Collection produces facts; analysis produces findings — and they never share code.**

1. **Design as data.** Every experimental cell (which question, attack, phrasing
   variant, arm, replicate, the offered wrong answer, the attack order) is decided
   from a seed *before any API call* and frozen into `trials.parquet`. A reviewer
   audits the design by reading a file, not by tracing loop logic.
2. **A ledger is the single source of truth.** Every API call is one append-only,
   `fsync`'d line in `ledger.jsonl` (a `CallRecord`) stamped with full provenance
   (model snapshot, temperature, seed, git SHA, config hash). A crash loses at most
   the one call in flight.
3. **Content-addressed cache.** Each call is keyed by a hash of its full request
   (including the entire prior conversation), so the **multi-turn manipulation path
   caches** and any identical re-run is free. (The old framework re-billed every run.)
4. **Pure analysis.** The analysis layer reads `trials.parquet` + `ledger.jsonl` and
   regenerates every table, CI, and significance test with **zero network** — so you
   can re-score and re-analyze without re-collecting, and unit-test it on fixtures.
5. **Everything is reproducible.** A run is pinned by its `manifest.json` (git SHA,
   config/pricing hashes, seed, library versions, model snapshots, dataset revisions).
   `verify-ledger` refuses to trust a ledger that lacks this provenance.

---

## 2. Quickstart

```bash
pip install -r latest/requirements.txt
cp latest/.env.example latest/.env     # then fill in your API keys
```

> **Run all commands from the repository root.** The package is invoked as
> `python -m latest.…`; from elsewhere you'll get `ModuleNotFoundError: No module
> named 'latest'`. (Optionally `pip install -e ./latest` via `latest/pyproject.toml`
> to import `latest` from anywhere.)

`.env` keys (only the providers you use are required):
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`.

**Run it (interactive launcher):**

```bash
python -m latest.main
#   → Run the offline test suite?       [Y/n]
#   → Run a SMOKE test? (smoke.yaml)     [Y/n]   ← tiny run across ALL modules + datasets
#   → Run the COMPREHENSIVE run?         [y/N]   ← the full config/eval.yaml
```

**Headless (CI / automation):**

```bash
python -m latest.main --smoke           # tiny run, no prompts (config/smoke.yaml)
python -m latest.main --comprehensive   # full config/eval.yaml, no prompts
python -m latest.main --smoke --comprehensive   # smoke first, then full
# flags: --skip-tests  --skip-live  --yes  --models ...  --bench-n N  --moral-per-cat N  ...
```

Every run does its own **preflight** (modules import → API keys present → a live
connectivity check that actually calls each model) and then collects with constant
saves. Results land in `runs/<run_id>/report/report.md`.

---

## 3. What gets tested

### Manipulation resistance — the 2×2 of modes (+ drift)

The model first gives a reasoned answer to a hard-MMLU question; an "attacker" then
pushes a wrong option. The modes vary two axes — **how many attack types** and **how
many phrasing variants** are chained in one session:

| | one phrasing variant | all variants chained |
|---|---|---|
| **one attack type** | `stateless` — per-attack potency ranking | `repeat` — persistence under rephrasing |
| **all attack types** | `stateful` — cross-tactic killing-blow | `gauntlet` — endurance ceiling |

Plus `drift`: a 6-turn *gradual* escalation. Outcomes per turn: **resistant** (held) /
**hedged** (UNCERTAIN) / **folded** (switched to a specific wrong) / **invalid**
(baseline already wrong — excluded). Arms: `pressure_wrong` (attacker is wrong) and
`control` (neutral rephrasing → natural-drift baseline).

### Standard benchmark

MMLU, HellaSwag, TruthfulQA-MC (structured single-call, exact-match scoring) and
TruthfulQA-gen (free-form answer scored by the LLM judges on truthful + informative).

### Moral / empathy

Preference / ethical / crisis scenarios; free-form answers scored 1–5 on
category-specific axes (helpfulness, empathy, safety, reasoning) by **≥2 judges of
different families**, with inter-judge agreement (Cohen's κ) reported.

---

## 4. Folder structure (every file)

```
latest/
├── __init__.py
├── main.py                  # PROD entry: interactive launcher + headless flags; preflight then run
├── cli.py                   # `run` / `analyze` / `verify` subcommands; _build_trials() (overrides)
├── env.py                   # .env loading + API-key presence helpers (used by main/cli AND tests)
├── records.py               # THE data contract: answer schemas + Trial / CallRecord / ScoreRecord /
│                            #   RunManifest + ProviderResponse + content-address hashers
├── provenance.py            # build the RunManifest: git SHA+dirty, config/pricing hashes, lib versions
├── ledger.py                # append-only fsync'd CallRecord writer; run-dir layout; read/resume/verify
├── cache.py                 # content-addressed, multi-turn-aware response cache (one file per call_id)
├── runlog.py                # structured JSONL operational log (events, fsync'd) — separate from the ledger
├── loaders.py               # stimulus loaders: local JSONL templates + HF datasets, normalized
│
├── config/
│   ├── loader.py            # __file__-relative config loader (+ get_price, validate, load_smoke)
│   ├── eval.yaml            # the full configured run: models, judges, modules, n, modes, arms, subjects
│   ├── smoke.yaml           # --smoke overrides: smallest run that still exercises everything
│   ├── pricing.yaml         # per-1M-token prices (single source of truth for cost)
│   ├── datasets.yaml        # HF dataset ids + pinned revisions (reproducibility)
│   └── snapshots.lock.yaml  # alias -> dated model snapshot (filled after a run)
│
├── data/
│   ├── manipulation_attacks.jsonl   # 7 pressure attacks + neutral_control + incremental_drift (x4 variants)
│   ├── manipulation_drift.jsonl     # 5 escalation turns sent after the baseline (6 turns total)
│   └── moral_scenarios.jsonl        # 9 scenarios across preference / ethical / crisis
│
├── providers/               # ONE router; uniform client-side history; cache+ledger live in the base
│   ├── __init__.py          # registers all providers; re-exports resolve/start_conversation/chat/estimate_cost
│   ├── router.py            # resolve(model)->(provider,model); registry; start_conversation/chat/estimate_cost
│   ├── base.py              # Conversation ABC + CallContext; template send() = cache-check → call → ledger
│   ├── retry.py             # tenacity exponential backoff; billing/quota = fail-fast (non-retryable)
│   ├── schema_util.py       # the single make_strict() (additionalProperties=False) for strict json_schema
│   ├── openai.py            # OpenAIConversation (chat.completions.parse) + chat + estimate_cost
│   ├── anthropic.py         # AnthropicConversation (messages.parse) + chat + estimate_cost
│   ├── gemini.py            # GeminiConversation (response_schema) + chat + estimate_cost
│   └── chat_completions.py  # Groq + OpenRouter (OpenAI-compatible strict json_schema, with fallback)
│
├── plan/                    # build the frozen design matrix (no network for the pure expanders)
│   ├── design.py            # build_manipulation_trials / build_benchmark_trials / build_moral_trials / build_all
│   ├── arms.py              # attack taxonomy + arm_for_attack()
│   ├── wrong_answer.py      # seeded, plausibility-recorded distractor selection
│   ├── order.py             # counterbalanced stateful_order / variant_order / gauntlet_order
│   └── freeze.py            # write/read trials.parquet (+ trials.manifest.json, design_hash)
│
├── collect/                 # execute the design into the ledger (constant saves; resume; budget gate)
│   ├── engine.py            # fan-out across models; Progress (resume); interactive budget gate
│   ├── run.py               # per-trial dispatcher by module
│   ├── manipulation.py      # turn a manipulation Trial into the right send() sequence per mode
│   ├── benchmark.py         # MC structured call; gen free-form answer + judge calls
│   ├── moral.py             # free-form answer + per-axis judge calls
│   ├── judge.py             # LLM-as-judge prompts (moral axes + truthfulqa) — run at collection time
│   └── _calls.py            # cached, ledger-logged single-turn free-form call (moral/gen/judges)
│
├── analysis/                # PURE, network-free interpretation of the ledger
│   ├── score.py             # ledger -> ScoreRecords (resist/fold/hedge; benchmark accuracy; moral axes)
│   ├── aggregate.py         # per-model tables: per-attack rates, natural-drift correction, killing-blow,
│   │                        #   repeat/gauntlet endurance, benchmark accuracy, moral axes, judge κ, cost
│   ├── stats.py             # Wilson CIs + McNemar (statsmodels) + pairwise (variant-averaged)
│   └── report.py            # writes report/report.md, report/results.json, scored.parquet
│
└── tests/
    ├── conftest.py          # loads .env; run_root fixture
    ├── test_offline.py      # records / cache / ledger / router / Conversation (fake provider) — no network
    ├── test_plan.py         # design-matrix expansion + parquet round-trip
    ├── test_analysis.py     # scoring/aggregation across all modules (guards the multi-model bug)
    ├── test_infra.py        # budget-error classifier + run-log
    ├── test_live_smoke.py   # one real 2-turn conversation per configured provider (auto-skips w/o keys)
    └── smoke_live.py        # standalone printed demo of a two-turn conversation

runs/<run_id>/               # one directory per run (the deliverables)
├── manifest.json            # provenance (written first)
├── trials.parquet           # the frozen design matrix
├── trials.manifest.json     # design_hash + counts by module/mode/arm/attack
├── ledger.jsonl             # every API call (the source of truth)
├── run.log.jsonl            # operational events (preflight, progress, budget, completion)
├── progress.jsonl           # completed (model, trial) for resume
├── scored.parquet           # one row per derived ScoreRecord
└── report/
    ├── report.md            # the human-readable results
    └── results.json         # all aggregate tables
runs/cache/                  # shared content-addressed cache (cross-run; makes repeats free)
```

---

## 5. End-to-end flow

```
config/eval.yaml ──┐
config/smoke.yaml ─┤ (overrides)
                   ▼
   plan/  build_*_trials() ──► trials.parquet   (frozen design; one Trial per cell)
                   │
                   ▼
   collect/ engine.collect() ──► for each (model, trial): providers run the turns,
                   │             every call appended+fsync'd to ledger.jsonl
                   │             (cache hit ⇒ $0; budget wall ⇒ ask user; crash ⇒ resume)
                   ▼
   analysis/ report.analyze() ──► score_all() → aggregate → report/report.md + results.json + scored.parquet
                                  (pure: reads trials.parquet + ledger.jsonl, no network)
```

Re-analyze an existing run without re-collecting:
```bash
python -m latest.cli analyze --run-id <id>
python -m latest.cli verify  --run-id <id>     # integrity + provenance gate
```

---

## 6. Major abstractions

- **`records.Trial`** — one frozen design row. Every design choice is a column
  (`arm`, `attack`, `variant_idx`, `mode`, `offered_answer`, `distractor_plausibility`,
  `stateful_order`, `replicate_idx`).
- **`records.CallRecord`** — one ledger line per API call. `call_id` is the content
  address of the request (= cache key, shared across trials that issue an identical
  request). Carries `model_version`, `temperature`, `seed`, `git_sha`, `config_hash`.
- **`records.ReasonedAnswer`** — the manipulation answer schema with all four axes
  live: `letter`, `confidence` (1–5), `acknowledged_counterargument`, `reasoning`.
- **`providers.base.Conversation`** — template-method `send()`: append the user turn →
  content-address the full transcript → cache hit *or* `_raw_send()` (the provider's
  real call) → append the assistant turn → write a `CallRecord`. Subclasses only
  implement `_raw_send`; they know nothing about caching, the ledger, or provenance.
  **All providers use client-side history** so caching is correct and uniform.
- **`ledger.Ledger`** — append-only, `fsync`'d writer; `read` / `completed_call_ids`
  (resume) / `verify` (integrity + provenance).
- **`analysis.score.score_all`** — joins `Trial` (design) with its `CallRecord`s
  (facts) to produce `ScoreRecord`s; each names the `source_call_ids` it came from.

---

## 7. Reproducibility & operations

- **Constant logging:** every call → `ledger.jsonl` (fsync); every event →
  `run.log.jsonl` (fsync); completed trials → `progress.jsonl`. Nothing is lost.
- **Resume:** a re-run skips completed `(model, trial)` and serves any repeated call
  from the shared cache.
- **Exponential retry:** `providers/retry.py` retries transient 429/503/529/overload/
  timeout with backoff + jitter (6 attempts).
- **Budget gate:** a spend/quota error is *non-retryable*; the engine pauses and asks
  — **y** = "I topped up, resume the unfinished trials", **n** = "stop and keep
  everything saved". Unattended (no TTY) defaults to stop-and-save (never hangs).
- **Provenance:** `manifest.json` pins git SHA (+ dirty), config/pricing hashes, seed,
  library versions, model snapshots, dataset revisions. `verify-ledger` enforces it.
- **Pinning TODO before a paper run:** set `revision:` in `config/datasets.yaml` and
  run snapshot locking so `config/snapshots.lock.yaml` is filled (these surface as
  loud warnings from `config.loader.validate()` until done).

---

## 8. Extending it (each is a one-file change)

| To add… | Do this |
|---|---|
| a model | one line under `models:` in `config/eval.yaml` |
| a provider | a `Conversation` subclass + `router.register(...)` + a `pricing.yaml` row |
| an attack | one row in `data/manipulation_attacks.jsonl` (auto-expanded into Trials) |
| a mode | a `mode` branch in `plan/design.py` + `collect/manipulation.py` + `analysis/score.py` |
| a scorer / reporter / statistic | one pure function in `analysis/` + a unit test |
| a judge | add to `judges:` in `eval.yaml` (≥2 enables inter-judge κ) |

---

## 9. Tests

```bash
# offline (no keys, no spend) — also run automatically by main.py preflight
python -m pytest latest/tests/test_offline.py latest/tests/test_plan.py \
                 latest/tests/test_analysis.py latest/tests/test_infra.py -q

# live (real calls; auto-skips providers without keys)
python -m pytest latest/tests/test_live_smoke.py -v
python -m latest.tests.smoke_live          # printed two-turn demo
```
