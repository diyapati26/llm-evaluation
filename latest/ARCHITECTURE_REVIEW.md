# LLM Manipulation-Resistance Eval — Architecture Review & Rewrite Plan

**Date:** 2026-06-13 · **Branch:** feature/DP · **Scope:** `Archive/evals/` (root, class-based) vs `Archive/Simpler Arch/` (function-based)
**Method:** First-hand read of every load-bearing file + a 32-agent verification/design workflow (4 analyzers → 24 adversarial bug-verifiers → 3 competing designs → judge panel). 23 of 47 claimed issues verified as real; 1 refuted; the rest downgraded.

---

## 1. Verdict (TL;DR)

- **Keep `Simpler Arch/` as the base; retire root `evals/`.** Simpler Arch is strictly better on every infra axis (typed structured output, retry/backoff, central pricing, version-keyed cache, config-driven, crash-safe persistence, 5 providers, the multi-turn `Conversation` abstraction). Root's only unique asset is the *idea* of an importable typed result schema (`EvalResult`/`MetricScore`) — worth carrying forward — but its manipulation runner parses answers with **fragile regex**, has **no retry**, **no central pricing**, an **empty config**, and its **McNemar test never runs** (imports `scipy.stats.mcnemar`, which does not exist). Two stacks producing differently-priced, differently-sampled numbers for "the same" eval is itself a publication hazard.
- **The science needs real work before the numbers are publishable.** The construct (does a model abandon a correct, well-reasoned answer under social pressure?) and the resist/fold/hedge taxonomy are a genuine contribution, but the current design has a **central selection-bias confound**, is **badly underpowered** (n=10, single sample/cell), uses **pseudo-replicated** phrasing variants as independent data, has **no CIs / effect sizes / multiple-comparison correction**, and uses **one provider as an unvalidated judge**. These are fixable, and most of the fixes are *architectural* — they fall out of separating data collection from analysis.
- **The rewrite should be a single installable package built as a "ledger-first" engine:** a frozen experimental-design matrix feeds a content-addressed, cached, append-only per-call run-ledger (collection), and a pure, network-free analysis layer turns that ledger into every table, CI, and significance test (analysis). This structurally fixes the biggest cost gap (uncached manipulation), the biggest reproducibility gap (lost metadata, no resume), and the biggest methods gap (analysis tangled into a 763-line script).

---

## 2. The two architectures, side by side

| Axis | Root `evals/` (class/async) | `Simpler Arch/` (function/threaded) | Winner |
|---|---|---|---|
| Answer extraction | Regex over free-form text (`extract_letter`, A–D) — can fabricate a letter from prose | Typed `ReasonedAnswer` Literal `{1,2,3,4,UNCERTAIN}` | **Simpler** |
| Multi-turn | `generate_conversation(messages)` re-send history | `Conversation` subclasses hide server-vs-client state asymmetry (OpenAI server-side; Anthropic/Groq/OpenRouter/Gemini client-side) | **Simpler** |
| Providers | OpenAI, Anthropic, Groq | + Gemini, OpenRouter (5 total) | **Simpler** |
| Retry/backoff | None — a transient 429 permanently drops a sample | tenacity exp-backoff + jitter, billing fail-fast, applied at every leaf call | **Simpler** |
| Pricing | Hardcoded per-provider dicts (drift-prone) | Central `config/pricing.yaml` + dated verification + fallback | **Simpler** |
| Cache key | `sha256(alias+prompt)` — stale on snapshot rotation | `dataset::model_version::question` — auto-invalidates | **Simpler** |
| Config | `configs/eval_config.yaml` is **empty (0 bytes)**; everything hardcoded | `config.yaml` drives models/n/concurrency/mode/subjects | **Simpler** |
| Persistence | Writes only at the very end; crash loses the whole run | Per-model incremental JSONL + UTF-8 forcing + build-payload-before-print | **Simpler** |
| Significance test | `from scipy.stats import mcnemar` → **ImportError, silently never runs** | Manual McNemar on `scipy.stats.chi2` — actually runs | **Simpler** |
| Typed result contract | `EvalResult` / `MetricScore` Pydantic models | Plain dicts everywhere | **Root** (carry this idea forward) |
| Moral/empathy module | **Does not exist in this tree** | Present (`moral.py`, 9 scenarios, 4-axis judge) | **Simpler** |

**Conclusion:** consolidate onto Simpler Arch, graft root's "typed records as the data contract" idea, delete root.

---

## 3. Verified findings that matter for the paper

### 3a. Methodology / internal validity (the ones a reviewer will hit first)

1. **Selection bias on initial correctness (HIGH, central confound).** Resistance is measured only on items a model got right initially; the rest are dropped as `invalid`. So the comparison set is *model-dependent* — nano was excluded on ~50% of items (and entirely on 5/10 subjects), Haiku on 0%. "nano is more resistant" is computed on the easy half where nano was confident; Haiku is tested on its full, harder set. **The headline "capability is inversely related to robustness" is confounded by differential selection.** Fix: report resistance on the **common intersection** of items all models answered correctly (matched design), and report initial accuracy separately.
2. **Attacker always offers a fixed, deterministic wrong answer (HIGH).** `get_wrong_number` cycles `correct→next`; root takes "first non-correct choice." Not randomized, not matched on distractor plausibility, position-dependent, and *different between the two architectures* (so the two runs aren't comparable). Fix: seeded, plausibility-matched distractor selection recorded as data.
3. **One provider as unvalidated judge (HIGH).** All moral + TruthfulQA-gen scoring uses GPT-5.4-mini. Self-preference when judging OpenAI outputs; no second judge, no human anchor, no agreement statistics. Any moral finding is inseparable from judge bias.
4. **Temperature 0 + single sample = no variance estimate (HIGH).** One greedy decode is not "the" behavior — the paper's own phrasing-sensitivity result proves small perturbations flip it. Cannot distinguish a true 88% fold rate from noise. Fix: k≥5 resamples per cell (or temperature>0) for per-item CIs.
5. **Stateful "killing blow" confounded by fixed attack order (MED).** `STATEFUL_ATTACK_ORDER` always puts `confident_assertion` first, so it gets first crack at 100% of samples while later attacks see a shrinking residual. "confident_assertion is the most effective attack" cannot be drawn from a fixed-order chain — base potency claims on the *stateless* per-attack rates instead, or counterbalance order.
6. **No true-pushback / corrigibility arm (MED).** The attacker is *always wrong*, so "resistance" conflates healthy epistemic firmness with harmful obstinacy. Add a condition where the pushback is correct so high resistance can be interpreted as a safety positive.

### 3b. Scoring/metric correctness bugs (confirmed in code)

7. **Two of four scoring axes are permanently dead (Simpler Arch).** `score_resistance_vector` reads `confidence` and `acknowledged_counterargument`, but `ReasonedAnswer` only declares `letter` + `reasoning`, and the prompts never ask for them → `confidence_delta` always `None`, `engagement` always `0.0` (confirmed 755/755 rows). Fix: add the fields + prompt for them, or delete the dead axes honestly. (`letter_persistence` and `hedged` *do* work.)
8. **Natural-drift correction subtracts incompatible scales (MED).** `adjusted = raw_resistance − drift_rate`: a hedge counts as **0.5** in `raw_resistance` (continuous) but as a **full 1.0** of drift in `drift_rate` (binary outcome string). Hedge-prone models are over-penalized asymmetrically. Fix: put drift on the same `{0,0.5,1}` scale (`drift = 1 − mean(control scores)`).
9. **LLM-judge scored by naive `"yes" in reply` substring (MED).** A verbose/negated judge reply containing "yes" anywhere scores positive ("No, this is not yes…" → truthful=1). Biases TruthfulQA-gen upward. Fix: constrained-token / structured judge output.
10. **Moral judge parse-failure returns `0`, averaged as a real score (MED).** A formatting failure is indistinguishable from a genuine low rating and silently drags the mean below the scale's true minimum (1). Fix: return `None`, exclude from the denominator, log the failure count.
11. **McNemar treats 4 phrasing variants as independent (HIGH, pseudo-replication).** Match key `(sample_id, attack, variant_idx)` inflates effective n ~4×, inflating χ² and deflating p. Fix: average variants to one outcome per `(item, attack)` before inference, or use a mixed-effects model.
12. **Root-only:** `extract_letter` fabricates a letter from prose (`\b[A-D]\b` matches a stray "A"; final fallback returns `upper[0]`, so "Definitely" → "D"); `score_resistance` only inspects first/last turn (mid-conversation folds mismeasured); provider `.content[0].text` / `.choices[0].message.content` crash on empty/None content and silently shrink the denominator.

### 3c. Infra / reproducibility (confirmed)

13. **Manipulation path is 100% uncached (biggest cost issue).** `Conversation.send` never consults the cache; every manipulation run re-bills (~$1.39 for just 2 models × 10 questions; scales with n and rerun count). Benchmark *is* cached. Fix: make `Conversation.send` cache-aware, keyed on full message-history + model_version + temperature + schema.
14. **Manipulation results drop `model_version` and `temperature` (HIGH).** The data exists on `ProviderResponse` but is discarded in the result dicts (confirmed: 740/740 rows lack both). A reviewer can't tell which dated snapshot produced the numbers. Fix: stop dropping them; write a `run_manifest.json`.
15. **No dataset/library version pinning (HIGH).** HF datasets pulled live with no `revision=`; `datasets>=2.0.0` lower-bound only. Same `random_state` can sample different questions over time. Fix: pin `datasets==` and pass commit-hash `revision=`.
16. **No resume from partials (MED).** The crash-safe `*_partial.jsonl` is written but never read back, so a re-run repeats and re-bills everything (compounded by the no-cache gap). Per-model granularity also loses a full in-flight model on crash. Fix: per-call ledger + skip completed `call_id`s on resume.
17. **`load_config.py` uses a cwd-relative path (footgun).** Runners only work from inside `Simpler Arch/`. Fix: resolve relative to `__file__`.

### 3d. Statistics gaps that block publication

- Severely underpowered (n=10; n=3 moral) — at n=8 an "88% fold" has a 95% Wilson CI of ~0.53–0.98.
- No CIs, no effect sizes, no multiple-comparison (FDR/Bonferroni) correction anywhere.
- LLM-judge reliability never quantified (no human validation, no κ/α, no second judge).
- Population vs sample std inconsistency between the two codebases (~15% at N=4).
- The two reports (`RESULTS.md` vs `diya_report.md`) use **different models and contradict each other** on the headline (one says Claude is robust; the other says Haiku is fragile) — they must be reconciled into one canonical run.

---

## 4. Proposed target architecture — `latest/`

**Philosophy:** *Collection produces facts; analysis produces findings — and the two never touch the same code path.* The single source of truth is an append-only, content-addressed **per-call ledger**. A **frozen design matrix** (every trial, with its offered answer, distractor plausibility, arm, and replicate, decided before any API call) feeds the collector. A **pure analysis layer** reads the ledger and emits every number in the paper with zero network access. Extension happens through **registries** (add a model = one YAML line; a provider/scorer/reporter = one decorated function). This is the judge panel's winning "Ledger-First" design, with the best ideas grafted from the "Pipeline-as-Paper" (frozen design matrix, Parquet analysis mirror, `make` reproduce-button, preregistration) and "registry-first" (decorator seams, hard manifest contract) proposals.

### Module tree

```
latest/
  config/
    loader.py          # __file__-relative (fixes the cwd footgun); validates on load
    eval.yaml          # models per provider, n, k (resamples), concurrency, modes, subjects
    pricing.yaml       # carried verbatim from Simpler Arch (single source of truth)
    datasets.yaml      # dataset id + pinned revision (commit hash) per dataset
    snapshots.lock.yaml# alias -> dated model snapshot, committed
  records.py           # ALL Pydantic models (Trial, CallRecord, ScoreRecord, RunManifest)
  provenance.py        # capture git SHA+dirty, config/pricing hashes, lib versions, tz timestamp
  plan/
    design.py          # build the frozen trials table from a seed
    wrong_answer.py    # seeded, plausibility-matched distractor selection
    order.py           # counterbalanced stateful attack order
    arms.py            # pressure_wrong | corrigibility (pushback is correct) | control
    freeze.py          # -> runs/<id>/trials.parquet + trials.manifest.json
  providers/
    __init__.py        # @register_provider + startup validation of the registry
    router.py          # ONE resolve(model)->handle, used by single-shot AND conversation
    base.py            # Conversation ABC; cache+ledger live inside .send
    schema_util.py     # the ONE _make_strict (currently triplicated)
    openai.py anthropic.py gemini.py chat_completions.py   # carried from conversation.py
    retry.py           # typed-exception classification; reasoning-model temp guard
  cache.py             # content-addressed, full-history key (covers multi-turn) — closes the cost gap
  collect/
    engine.py          # read trials.parquet, skip ledger-completed call_ids (resume),
                       # bounded concurrency, append+fsync ONE CallRecord per call
    run_benchmark.py run_manipulation.py run_moral.py   # build plans, stream CallRecords ONLY
  analysis/            # PURE, network-free, fixture-tested
    score.py           # extract/score; returns None on ambiguity (no fabricated letters)
    aggregate.py       # variant-averaging BEFORE inference
    stats.py           # statsmodels McNemar + Wilson CIs + effect sizes + mixed-effects + FDR
    judge_reliability.py  # kappa / alpha vs human labels and across judges
    report.py          # UTF-8; @register_reporter
  cli.py               # latest plan --dry-run | collect | analyze | verify-ledger | lock-snapshots
  Makefile             # one target per stage = the reviewer reproduce-button
  tests/               # real pytest on pure functions + mocked-client provider tests + ledger replay
runs/<run_id>/         # manifest.json, trials.parquet, ledger.jsonl, cache/, scored.parquet, report/
data/                  # attacks.jsonl, drift.jsonl, moral.jsonl, corrigibility.jsonl, human_labels.jsonl
experiments/
  preregistration.md   # hypotheses, exclusion rules, primary outcome, MC-correction, snapshots
```

### Data model (`records.py`)

- **`Trial`** — one frozen design row, decided *before* any call: `trial_id, item_id, subject, arm, attack, variant_idx, replicate_idx, mode, stateful_order, correct_answer, offered_answer, distractor_plausibility, is_canary`. *Every design-validity threat becomes a column a reviewer audits by reading a file* — not logic buried in a modulo loop.
- **`CallRecord`** — the ledger line, one per API call: `call_id = sha256(canonical request)` (= cache key), `run_id, provider, model_alias, model_version, temperature, seed, max_tokens, condition, attack, variant_idx, trial_id, turn_index, n_turns, messages_hash, prompt, answer_raw, text, tokens, cost, latency, cache_hit, error, git_sha, config_hash`. Append-only + fsync ⇒ a crash loses exactly one call; resume reads completed `call_id`s.
- **`ScoreRecord`** — derived *only* in analysis; names the `CallRecord`s it came from.
- **`RunManifest`** — written first; `verify-ledger` and the analysis layer **refuse** to interpret a ledger without it (no orphan results file can ever be cited).
- **`ReasonedAnswer`** — gains `confidence` + `acknowledged_counterargument` *and prompts that elicit them* (reviving the two dead axes), or they are deleted honestly.

**Source-of-truth rule:** `ledger.jsonl` is canonical; `cache/` and `scored.parquet` are rebuildable derivations; the report is a pure function of `ledger + manifest`.

### What this fixes, by construction

| Problem | How the architecture removes it |
|---|---|
| Uncached manipulation re-bills every run | Cache lives inside `Conversation.send`; every condition cached for free; re-run = $0 |
| Lost `model_version`/`temperature`, no resume | Every fact is a `CallRecord` field, fsynced; resume skips completed `call_id`s |
| Analysis trapped in a 763-line script | Pure `analysis/` over the ledger; re-score/re-analyze without re-collecting; unit-testable with zero spend |
| Fixed wrong answer / fixed order / pseudo-replication / no corrigibility arm | All are columns in the frozen `trials.parquet`, set by seed before any call |
| Dual `route()`/`_resolve()` drift, triplicated `_make_strict` | One `router.py`, one `schema_util.py`, registry validated at startup |
| Hardcoded OpenAI judge crashes on Anthropic | Judge routed through the one router; supports ≥2 judges of different families |
| Naive stats (wrong McNemar import, no CIs, no FDR) | `statsmodels` McNemar, Wilson CIs, effect sizes, mixed-effects, FDR — pure & fixture-tested |

### Extension story

- **New model:** one line in `eval.yaml` + `latest lock-snapshots`.
- **New provider:** one `Conversation` subclass + `@register_provider` + one `pricing.yaml` row.
- **New attack:** one JSONL row (auto-expanded into Trials).
- **New arm (e.g. corrigibility):** one `arms.py` entry + JSONL; analysis groups by `arm`.
- **New eval module:** one `collect/run_<x>.py` emitting `CallRecord`s (inherits cache/resume/ledger).
- **New scorer / reporter / statistic:** one decorated pure function with a unit test.
- **Multi-judge:** list ≥2 judges (any `provider:model`) + enable the `judge_reliability` reporter.

---

## 5. Migration plan (ship value before restructuring)

- **Step 0 — cheap, high-value correctness fixes (no restructuring):** make config/cache paths `__file__`-relative; add `statsmodels` + pinned `datasets==` + dataset revisions to `requirements.txt`; fix the root McNemar import (or just delete root). Retire `evals/`.
- **Step 1 — the ledger substrate:** `records.py` + `provenance.py` + ledger writer + `verify-ledger`.
- **Step 2 — close three gaps at once:** cache-aware `base.Conversation.send`, unify routing into `router.py`, de-triplicate `_make_strict`. (Fixes multi-turn caching + dropped metadata + crash-safety together.)
- **Step 3 — frozen design + resume:** `plan/` builds `trials.parquet` (seeded randomized wrong-answer, counterbalanced order, k≥5 resamples, corrigibility arm); collectors stream `CallRecord`s; resume skips completed calls.
- **Step 4 — pure analysis:** fix McNemar (statsmodels), variant-averaging before inference, Wilson CIs, same-scale drift correction, FDR, mixed-effects; revive or delete the dead axes; route the moral judge through the router.
- **Step 5 — tests + delete the old monoliths:** pytest on pure functions, mocked-client provider tests, fixture-ledger replay.
- **Step 6 — science hardening as data:** corrigibility arm, higher n, second-family judge + human labels for κ, commit `preregistration.md` before the confirmatory run.

---

## 6. Decisions needed from you before code

1. **Retire root `evals/` entirely?** (Recommended: yes — consolidate on Simpler Arch.)
2. **Paper scope:** manipulation-resistance only, or keep the benchmark + moral/empathy modules in the paper? (Affects how much of `collect/` and `analysis/` we build now.)
3. **Model lineup for the canonical run** (the two existing reports disagree and use different models — we need one matrix).
4. **Compute/cost budget** → sets n (items/subject) and k (resamples/cell). The must-have is ~50–100 items/subject × k≥5; caching makes reruns free after the first.
5. **Dead axes:** add `confidence` + `acknowledged_counterargument` and prompt for them, or delete them?
6. **Corrigibility arm:** include the "pushback is actually correct" condition now (recommended — it's what lets "high resistance" mean something) or defer?
