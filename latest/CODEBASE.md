# `latest` — Complete Codebase Guide

A file-by-file, function-by-function reference for the `latest/` LLM
manipulation-resistance evaluation framework. Read `README.md` first for the
quickstart and the high-level picture; this document is the deep dive for anyone
who needs to understand, modify, or extend the code. `ARCHITECTURE_REVIEW.md`
records the analysis of the two retired prototypes (`Archive/`) that motivated
this design.

---

## 0. Mental model in one paragraph

The framework is split into three stages that never share code: **plan** decides
every experimental cell from a seed and freezes them to `trials.parquet`;
**collect** executes those trials against the model APIs, writing one append-only,
`fsync`'d line per call to `ledger.jsonl` (the single source of truth); **analyze**
reads `trials.parquet` + `ledger.jsonl` and produces every table, CI, and test
with no network. A content-addressed cache makes repeats free, a manifest pins
provenance, and `verify-ledger` is the integrity gate. Everything is config-driven
and runs from `python -m latest.main`.

```
config/*.yaml ─► plan/ ─► trials.parquet ─► collect/ ─► ledger.jsonl ─► analysis/ ─► report.md
                 (design)   (frozen)         (facts)     (source of truth)  (findings)
```

---

## 1. Top-level layout

Run everything from the **repo root** (`python -m latest.…`). Only `README.md` and
`CLAUDE_EFFORT.md` live at the root; `latest/` is the self-contained source of truth.

```
llm-evaluation/                 # repo root
├── README.md                   # overview + why Archive/ vs latest/
├── CLAUDE_EFFORT.md            # session log / handoff
├── .gitignore                  # secrets, caches, latest/runs/
├── Archive/                    # the two retired prototypes (reference only; not imported)
└── latest/                     # THE project (source of truth)
    ├── CODEBASE.md             # THIS file — the deep dive
    ├── ARCHITECTURE_REVIEW.md  # why the rewrite happened (analysis of Archive/)
    ├── requirements.txt        # pinned deps
    ├── pyproject.toml          # optional `pip install -e ./latest`
    ├── main.py  cli.py  env.py  lock.py  records.py  provenance.py  ledger.py  cache.py
    ├── runlog.py  loaders.py
    ├── config/   data/   providers/   plan/   collect/   analysis/   tests/
    └── runs/                   # run outputs (gitignored) — <run_id>/ + shared cache/
```

---

## 2. The data contract — `latest/records.py`

Everything serializes through these Pydantic models. This is the most important
file to understand; the rest of the system is functions over these types.

**Answer schemas (structured-output targets sent to providers):**
- `MMLU_Answer`, `HellaSwag_Answer` — `answer: Literal['1'..'4']`.
- `TruthfulQA_MC_Answer` — `answer: Literal['1'..'13']` (variable choice count).
- `TruthfulQA_Generation_Answer` — `answer: str` (free-form).
- `ReasonedAnswer` — the manipulation schema, with **all four scoring axes live**:
  `letter` (`'1'..'4'|'UNCERTAIN'`), `confidence` (`'1'..'5'`),
  `acknowledged_counterargument` (`bool`), `reasoning` (`str`). Each field's
  `description` doubles as the instruction the model sees.
- `ProviderResponse` — the uniform envelope every provider call returns:
  `provider, model, model_version, temperature, input_tokens, output_tokens,
  cost_usd, latency_ms, answer` (parsed instance), `raw` (its dict), `text`
  (free-form). Structured calls fill `answer`+`raw`; chat calls fill `text`.

**Pipeline artifacts:**
- `Trial` — one frozen design row. Fields encode every design choice as data:
  `trial_id, module, dataset, item_id, subject, question, choices, correct_answer,
  arm, attack, variant_idx, mode, stateful_order, offered_answer,
  distractor_plausibility, replicate_idx, is_canary, metadata`. `mode ∈
  {stateless, stateful, drift, repeat, gauntlet}`.
- `CallRecord` — one ledger line per API call. `call_id` is the content-address of
  the request (**= the cache key**, intentionally shared across trials that issue
  an identical request). Carries design linkage (`trial_id, module, item_id,
  subject, dataset`), routing/model identity (`provider, model_alias,
  model_version, temperature, seed, max_tokens`), experiment coordinates (`role,
  condition, attack, variant_idx, replicate_idx, turn_index, n_turns,
  judged_model`), payload (`messages_hash, prompt, answer_raw, text`), accounting
  (`input/output_tokens, cost_usd, latency_ms, cache_hit, error`), and provenance
  (`git_sha, config_hash`).
- `ScoreRecord` — a derived score (produced **only** in `analysis/`). Always names
  the `source_call_ids` it came from, so every number traces to raw calls.
- `RunManifest` — written first, once per run; provenance for the whole run.

**Hash helpers:** `canonical_json` (sorted-key, stable JSON), `sha256_hex`,
`make_trial_id(coords)` (deterministic id from design coordinates),
`make_call_id(request)` (content-address of an API request).

---

## 3. Configuration — `latest/config/`

- **`loader.py`** — `__file__`-relative loader (works from any cwd).
  - `load_config()` → fresh, mutable run config with companion files attached under
    `_pricing` / `_datasets` / `_snapshots` (run config deep-copied so CLI overrides
    can't poison the cache; companions shared read-only).
  - `models_from_config(cfg)` / `judges_from_config(cfg)` → flatten to
    `provider:model` strings.
  - `get_price(provider, model, default)` → per-1M-token price; **warns loudly**
    rather than silently returning $0.
  - `dataset_spec(name)`, `snapshot_for(alias)`, `load_smoke()`.
  - `validate(cfg)` → reproducibility/cost warnings (missing pricing, unpinned
    dataset revisions, unlocked model **and judge** snapshots).
  - `python -m latest.config.loader` prints the resolved config + warnings.
- **`eval.yaml`** — the full configured run: `run` (seed, results_root, concurrency),
  `models` (per provider), `judges`, `dataset.hard_mmlu`, `benchmark`,
  `manipulation` (items_per_subject, resamples, modes, arms, include_drift), `moral`.
- **`smoke.yaml`** — `--smoke` overrides (tiny run across everything). Editable.
- **`pricing.yaml`** — authoritative per-1M-token prices. Single source of truth.
- **`datasets.yaml`** — HF dataset ids + `revision` (pin commit SHAs before a paper run).
- **`snapshots.lock.yaml`** — alias → dated model snapshot (fill after a run).

---

## 4. Stimuli — `latest/data/` + `latest/loaders.py`

- **`data/manipulation_attacks.jsonl`** — 7 pressure attacks + `neutral_control` +
  `incremental_drift`, each with 4 phrasing variants.
- **`data/manipulation_drift.jsonl`** — 5 gradual-escalation turns (sent after the
  baseline → 6 turns total).
- **`data/moral_scenarios.jsonl`** — 9 scenarios across preference/ethical/crisis.
- **`loaders.py`** — turns raw stimuli into a uniform item dict `{item_id, dataset,
  subject, question, choices, correct_answer, refs}`.
  - `load_attacks()`, `load_drift_turns()`, `load_moral_scenarios()` — local JSONL.
  - `render_mc_prompt(question, choices)` — numbers the choices.
  - `load_hard_mmlu_items(subjects, items_per_subject, seed)` — manipulation pool.
  - `load_benchmark_items(name, n, seed)` — normalizes MMLU/HellaSwag/TruthfulQA.
    **TruthfulQA-MC choices are seed-shuffled** so the gold answer isn't always
    option 1; the permutation is recorded in `refs.perm`.
  - Sampling is seeded (`random.Random` over a fixed dataset order) → same seed +
    pinned revision = same items.

---

## 5. The design matrix — `latest/plan/`

Pure expanders (inject items → Trials) so the design is unit-testable with no network.

- **`arms.py`** — `PRESSURE_ATTACKS` (the 7), `CONTROL_ATTACK`, `DRIFT_SEQUENCE_ATTACK`,
  `STATEFUL_ATTACK`, `GAUNTLET_ATTACK`, `ARMS = [pressure_wrong, control]`,
  `arm_for_attack(attack)`.
- **`wrong_answer.py`** — `pick_wrong(correct, choices, seed, item_id)` → seeded
  distractor + a lexical `distractor_plausibility` proxy (fixes the old
  deterministic `(correct%4)+1` confound).
- **`order.py`** — `stateful_order(item_id, seed)` (counterbalanced attack order),
  `variant_order(item_id, attack, seed, n)` (Mode 1), `gauntlet_order(item_id, seed,
  attacks_with_variants)` (Mode 2). All seeded → reproducible, counterbalanced.
- **`design.py`** — the expanders:
  - `build_manipulation_trials(items, attacks, *, modes, include_drift, resamples,
    seed, max_variants)` — per item × replicate emits trials for each requested mode
    (the 2×2 + drift). The mode cheat-sheet:

    | mode | one Trial per | session shape |
    |---|---|---|
    | `stateless` | (item, attack, variant) | fresh 2-turn conversation |
    | `drift` | item | 1 baseline + 5 gradual turns |
    | `stateful` | item | baseline + counterbalanced attack types (variant 0), stop on fold |
    | `repeat` (Mode 1) | (item, attack) | baseline + all variants of one attack (counterbalanced) |
    | `gauntlet` (Mode 2) | item | baseline + all attack×variant steps (counterbalanced) |

  - `build_benchmark_trials(items_by_dataset)`, `build_moral_trials(scenarios)`.
  - `build_all(cfg)` — loads real stimuli for every enabled module and expands (network).
- **`freeze.py`** — `freeze(trials, rd, seed)` writes `trials.parquet` (object columns
  JSON-encoded for a flat, portable file) + `trials.manifest.json` (counts +
  `design_hash`). `read_trials(rd)` decodes back into `Trial`s (NaN→None, int coercion).

---

## 6. The provider layer — `latest/providers/`

The single place a model string maps to an API call. **All providers use
client-side history** (the full transcript is re-sent each turn), which is what
makes the content-addressed cache correct and uniform.

- **`router.py`** — `resolve(model)` → `(provider, real_model)` (pure prefix logic);
  a registry (`register`, `registered_providers`); `start_conversation(model, ctx,
  cache, ledger)` → the right `Conversation` subclass; `chat(model, messages, ...)`
  → single-turn free-form; `estimate_cost(model, in, out)`.
- **`base.py`** — `CallContext` (per-conversation linkage + provenance the base
  stamps on each `CallRecord`) and `Conversation`, whose `send()` is a
  **template method**:
  1. append the user turn to `self.transcript`;
  2. content-address the full transcript → `call_id`;
  3. cache hit → reconstruct the response; miss → `self._raw_send(...)` (the
     subclass's real API call) then `cache.put`;
  4. append the assistant turn (the answer as stable JSON) to the transcript;
  5. accumulate cost/tokens, build a `CallRecord`, append it to the ledger;
  6. return the `ProviderResponse`.
  Subclasses implement only `_raw_send`; they know nothing about caching/ledger.
- **`retry.py`** — `@retry_on_rate_limit` (tenacity, ~2→60s backoff + jitter, 6
  attempts). `_is_retryable` / `_is_hard_fail`: a quota/billing wall fails fast, but
  only when a high-precision phrase is present **and** no transient marker is — so a
  429 that merely mentions "monthly" still retries. `_is_hard_fail` is the shared
  classifier the collect engine reuses for its budget gate.
- **`schema_util.py`** — `make_strict(schema)` pins `additionalProperties=False`
  (required by Groq/OpenRouter strict json_schema).
- **`openai.py`** — `OpenAIConversation._raw_send` via `chat.completions.parse`
  (native messages list → `.parsed`); guards `parsed is None` (refusal / length
  truncation) by returning a clean `ProviderResponse` with the refusal text so the
  ledger records it. `chat()` for judges.
- **`anthropic.py`** — `AnthropicConversation` via `messages.parse` → `.parsed_output`.
  `chat()` concatenates all text blocks (skips leading thinking blocks).
- **`gemini.py`** — `GeminiConversation` via `generate_content(response_schema=…)`;
  guards `resp.parsed is None` (block / MAX_TOKENS) with an informative error.
  Converts the transcript to Gemini's `contents` format (assistant→model).
- **`chat_completions.py`** — `GroqConversation` / `OpenRouterConversation` (shared
  `ChatCompletionsConversation`) using strict json_schema with an inline-schema
  fallback. The fallback only catches schema/parse errors — rate-limit/billing
  propagate to retry (no double-billed attempt). OpenRouter prices are fetched live.
- **`__init__.py`** — importing the package registers all providers and re-exports
  `resolve / start_conversation / chat / estimate_cost`.

---

## 7. Collection — `latest/collect/`

Executes the frozen design into the ledger. Produces `CallRecord`s only — never scores.

- **`engine.py`**
  - `Progress` — crash-safe set of completed `(model, trial_id)` backed by
    `progress.jsonl` (resume).
  - `_is_budget_error` — reuses `retry._is_hard_fail` (one definition, can't disagree).
  - `_prompt_budget(model)` — the interactive budget gate (EOFError → stop+save, so
    unattended runs never hang).
  - `collect(trials, models, *, run_id, results_root, cfg, concurrency, run_trial_fn,
    runlog, on_budget)` — writes the manifest first (stamping the actual models +
    modules), opens the shared cache + ledger, then per model runs pending trials in
    a `ThreadPoolExecutor`. On a budget wall: cancel pending, **drain in-flight
    futures and mark the cleanly-finished ones** (so resume can't duplicate them),
    ask the user, resume or stop. A resume cap prevents an endless prompt loop.
- **`run.py`** — `run_trial(model, trial, …)` dispatches by `trial.module`.
- **`manipulation.py`** — `run_trial` builds a `Conversation` and drives the per-mode
  turn sequence. `_is_fold(base, letter, correct)` is **baseline-aware** (only a
  correct→specific-wrong switch ends a chain; an already-wrong baseline isn't
  truncated). `conv.ctx.attack/variant_idx` are set per turn so each `CallRecord`
  records which attack hit (for killing-blow analysis).
- **`benchmark.py`** — MC datasets → one structured `Conversation.send`;
  `truthfulqa_gen` → a free-form answer + truthful/informative judge calls.
- **`moral.py`** — free-form answer + per-axis judge calls.
- **`judge.py`** — judge prompts (moral axes + truthfulqa yes/no) run at collection
  time and logged as `CallRecord`s with `role='judge'`, `condition=<axis>`,
  `judged_model=<subject>`. `CATEGORY_AXES` maps each moral category to its axes.
- **`_calls.py`** — `cached_chat(...)`: the free-form sibling of `Conversation.send`
  (cache + ledger) used by moral/gen answers and all judge calls. Judge calls key
  on the judge's own model, never the subject's snapshot.

---

## 8. Analysis — `latest/analysis/` (pure, network-free)

- **`score.py`**
  - `persistence(orig, post, correct)` → 1.0 held / 0.5 hedged / 0.0 folded / None
    invalid. **A missing/errored post turn is `invalid`, not a 0.5 hedge** (only an
    explicit `UNCERTAIN` is a hedge).
  - `score_run(trials, records)` — groups manipulation turns by `(trial_id,
    model_alias)` (a trial runs once per model), walks each by `turn_index`, and
    emits a `ScoreRecord` per mode (stateless / drift / stateful / repeat /
    gauntlet, the last three recording `fold_attack` / `attacks_survived`). An empty
    chain → invalid.
  - `parse_rating(text)` — first whole integer, valid only 1–5 (so `'10'`→None, not
    1); a parse failure is `None` (excluded), never 0.
  - `score_benchmark` (MC accuracy; gen truthful/informative from judge rows),
    `score_moral` (per-axis means from judge rows), `score_all` (all three).
- **`aggregate.py`** — per-model tables: `stateless_overall`, `stateless_by_attack`,
  `natural_drift` (same-scale: `1 − mean(control scores)`), `stateless_adjusted`,
  `stateful_summary`/`gauntlet_summary` (killing blow + endurance),
  `repeat_summary`, `drift_summary`, `mcnemar_pairwise` (variant-averaged matched
  pairs; 0.5 tie counts as resistant — documented), `manipulation_excluded`
  (valid/invalid counts so all-invalid models are visible), `benchmark_accuracy`,
  `moral_axes`, `judge_reliability` (Cohen's κ + agreement, `float`-coerced), `cost_summary`.
- **`stats.py`** — `wilson_ci(k, n)` (small-n proportion interval), `mcnemar(b, c)`
  via **statsmodels** (the correct import — the old `scipy.stats.mcnemar` never
  existed), `pairwise_mcnemar(model_resist)`, `significance_stars`.
- **`report.py`** — `analyze(rd)` scores + aggregates the ledger and writes
  `report/report.md`, `report/results.json`, `scored.parquet`. `_markdown` renders
  every table by the **bare model alias** (the aggregate key), with an excluded-model
  footnote and a cause-specific McNemar fallback.

---

## 9. Persistence & provenance — `ledger.py`, `cache.py`, `runlog.py`, `provenance.py`

- **`ledger.py`** — run-dir layout (`manifest.json`, `trials.parquet`, `ledger.jsonl`,
  `scored.parquet`, `report/`) + the shared `cache/`. `Ledger.append` is
  **idempotent on the logical turn key** `(model_alias, trial_id, turn_index,
  judged_model, role, condition)` — re-appending a successful turn is a no-op, so
  resume can't create duplicate rows (error rows always write). `read`,
  `completed_call_ids` (tolerates a torn final line), `verify` (integrity +
  provenance gate; flags duplicate turns, missing provenance, manifest mismatch).
  Append opens with `newline=""` for portable LF lines.
- **`cache.py`** — directory of one JSON file per `call_id` (content address);
  atomic writes (`mkstemp`+`os.replace`); `call_key(provider, model_id, temperature,
  schema_name, messages, max_tokens, seed)` includes the full transcript → multi-turn
  caches. Shared across runs, so the first run pays and re-runs are free.
- **`runlog.py`** — `RunLog`: structured, `fsync`'d JSONL of operational **events**
  (separate from the data ledger), echoed to stdout.
- **`provenance.py`** — `build_manifest(cfg, run_id)` captures git SHA + dirty,
  config/pricing hashes, library versions, dataset revisions, snapshots, seed,
  models, judges, modules. `utc_now_iso`, `make_run_id`.

---

## 10. Entry points — `latest/main.py`, `latest/cli.py`, `latest/env.py`

- **`main.py`** — production launcher.
  - **Interactive** (`python -m latest.main`): asks *run tests? → run smoke? → run
    comprehensive?* in order.
  - **Headless**: `--smoke` (config/smoke.yaml), `--comprehensive` (config/eval.yaml),
    `--skip-tests`, `--skip-live`, `--yes`, plus all overrides.
  - **Preflight** (fail-fast, in order): `check_modules` → `check_env` (keys present)
    → `run_tests` (offline pytest) → `check_live` (a real structured conversation per
    subject + a real chat per judge — proves keys actually work).
  - `_execute(args, cfg, smoke)` runs one end-to-end run (env + live preflight →
    build → freeze → collect → analyze → verify) with its own `RunLog`.
- **`cli.py`** — `run` / `analyze` / `verify` subcommands + `_build_trials(cfg, args)`
  (applies overrides). `analyze`/`verify` operate on an existing run with no network.
- **`env.py`** — `load_env()` (reads `latest/.env`), `PROVIDER_KEYS`,
  `ENV_BY_PROVIDER`, `is_real_key`, `available_providers()`. Lives in the package so
  production code never imports from tests.

---

## 11. Tests — `latest/tests/`

- `test_offline.py` — records/IDs, cache, ledger (append/read/resume/**idempotency**/
  verify), router, the `Conversation` template with a fake provider (multi-turn +
  cache-hit replay), config. No keys, no network.
- `test_plan.py` — design expansion counts, arms/offered answers, counterbalanced
  order, determinism, parquet freeze round-trip.
- `test_analysis.py` — scoring/aggregation across all modules (guards the multi-model
  grouping bug), `persistence`/`parse_rating` edge cases, McNemar fires with 2 models,
  judge κ, report renders.
- `test_infra.py` — budget classifier (incl. a "monthly" 429 staying retryable) + run-log.
- `test_live_smoke.py` — one real two-turn conversation per configured provider
  (auto-skips without keys). `smoke_live.py` — a standalone printed demo.

Run offline: `python -m pytest latest/tests/test_offline.py latest/tests/test_plan.py
latest/tests/test_analysis.py latest/tests/test_infra.py -q` (26 tests).
`main.py`'s preflight runs these automatically.

---

## 12. Extending it

| To add… | Do this |
|---|---|
| a model | one line under `models:` in `config/eval.yaml` |
| a provider | a `Conversation` subclass + `router.register(...)` + a `pricing.yaml` row |
| an attack | one row in `data/manipulation_attacks.jsonl` (auto-expanded) |
| a mode | a branch in `plan/design.py` + `collect/manipulation.py` + `analysis/score.py` |
| a scorer / reporter / stat | one pure function in `analysis/` + a unit test |
| a judge | add to `judges:` in `eval.yaml` (≥2 enables inter-judge κ) |

---

## 13. Known limitations & pre-publication TODOs

- **Pin dataset revisions** (`config/datasets.yaml`) and **lock model snapshots**
  (`config/snapshots.lock.yaml`) before the confirmatory run — `validate()` warns
  until done.
- **Selection effect:** resistance is measured only on items a model answered
  correctly initially; a model wrong on every sampled item is reported as *excluded*
  (with its invalid count), not silently dropped. Scale `n` so each model has valid
  items and McNemar can match pairs.
- **Judge reliability** (Cohen's κ) should be validated against human labels before
  the moral/TruthfulQA-gen numbers are published.
- **Gauntlet (Mode 2)** is an endurance *ceiling*, not a per-attack potency claim
  (position confound over a long chain) — reported as such.

---

## 14. Review hardening log (2026-06-13)

A 33-agent adversarial review of `latest/` confirmed 28 issues; all were fixed:

**High** — retry/budget false-positive on transient 429s mentioning "monthly"
(now requires a precise quota phrase AND no transient marker; shared by the engine);
Gemini/OpenAI structured `parsed is None` crashes (now guarded → informative error /
clean record); resume could append duplicate turn rows (ledger `append` is now
idempotent on the logical turn key + the engine drains in-flight futures); a
missing/errored attack turn was scored as a 0.5 hedge (now `invalid`); `tqdm`
missing from `requirements.txt` (added).

**Medium** — `.env.example` missing though the README referenced it (added);
TruthfulQA-MC gold answer always option 1 (now seed-shuffled, permutation recorded);
`parse_rating('10')` returned 1 (now whole-integer, 1–5 only); all-invalid models
vanished silently from manipulation tables (now an explicit excluded footnote +
cause-specific McNemar fallback); Anthropic `chat()` crashed on thinking-block-leading
responses (now concatenates text blocks).

**Low** — `verify` turn key now includes `role`+`condition`; ledger/runlog/progress
opened with `newline=""`; Cohen's κ `float`-coerced; manifest records the actual
modules run; `validate()` checks judge snapshots; chat-completions fallback no longer
swallows rate/budget errors; chain stop-condition is baseline-aware; budget resume is
capped; McNemar 0.5-tie rule documented.
