# Claude Effort & Session Log

A handoff record of what was done in this session, the process behind it, and the
effort spent — so a future Claude (or human) can pick up with full context. For the
*system* itself, read `README.md` (overview) and `latest/CODEBASE.md` (deep dive).

---

## Session at a glance

| | |
|---|---|
| **Date** | 2026-06-13 |
| **Started** | ~15:47 EDT (first request: "go through the Archive and review both architectures") |
| **Ended** | ~18:55 EDT (this log) |
| **Duration** | ~3 hours, continuous |
| **Model / mode** | Claude Opus 4.8 (1M context), `ultracode` (xhigh effort + multi-agent workflows) |
| **Main-thread tokens** | ~788,471 |
| **Sub-agent tokens** | ~3,611,496 (two background workflows) |
| **Total tokens** | ~4.4M |
| **Sub-agents spawned** | 65 (32 + 33 across two workflows) |
| **Outcome** | New `latest/` framework built, smoke-validated end-to-end, adversarially reviewed, 28 issues fixed, fully documented |

Timing anchors (git): `rewrite complete` 17:57 EDT · `more changes` 18:10 · `final`
18:46. Smoke runs (UTC run-ids): 22:03 → 22:09 → 22:37 (= 18:03–18:37 EDT).

---

## What we set out to do → what we delivered

The user started by asking for a review of two prototype architectures in `Archive/`
(`evals/` and `Simpler Arch/`) with the goal of writing a research paper on LLM
**manipulation resistance**, and to design + build a better one. Over the session the
scope grew (by the user's direction) into a full rewrite, a smoke test of everything,
an adversarial review, and complete documentation.

Delivered:
1. **Architecture analysis** of both prototypes → `latest/ARCHITECTURE_REVIEW.md`.
2. **`latest/`** — a clean, ledger-first evaluation framework (≈45 modules) covering
   manipulation resistance (5 modes), a standard benchmark (4 datasets), and
   moral/empathy (LLM-as-judge), across OpenAI/Anthropic/Gemini/Groq/OpenRouter.
3. **A validated smoke run** (haiku + gpt-5.4-nano, all modules) producing a real
   results report.
4. **An adversarial review** (33 agents) → 28 confirmed issues, all fixed.
5. **Docs**: `README.md`, `latest/CODEBASE.md`, `latest/ARCHITECTURE_REVIEW.md`, this
   log; plus `requirements.txt`, `pyproject.toml`, `.gitignore`, `.env.example`.

---

## The process (chronological)

1. **Scout + analyze (workflow #1, 32 agents, ~1.74M tokens, ~16 min).** Read every
   load-bearing file first-hand, then fanned out parallel readers (root arch, simpler
   arch, scientific methodology, infra/reproducibility) → adversarially verified each
   bug claim → ran a 3-way competing-design panel + judge. Verdict: consolidate on
   `Simpler Arch`'s ideas, retire root, build a "ledger-first" engine.
2. **Decisions with the user** (via structured questions): all three eval modules;
   capability-ladder × 3 families; pilot-first run scale; **dropped** the corrigibility
   arm; **added** two new manipulation modes (`repeat`, `gauntlet`) completing a 2×2.
3. **Step-by-step build.** Config → records → provenance → ledger → cache → providers
   (live-verified across 4 families) → loaders/data → plan → collect → analysis →
   CLI → `main.py` → tests. Each layer verified before the next.
4. **Smoke run.** `python -m latest.main --smoke` ran end-to-end; fixed a multi-model
   scoring bug and a judged-model join bug found during validation; the cache made the
   re-run essentially free.
5. **Adversarial review (workflow #2, 33 agents, ~1.87M tokens, ~10 min)** over the
   new code + a clone-and-run check → 28 confirmed issues → all fixed → regression
   tests added → re-validated.
6. **Docs + reproducibility lock.** Wrote the guides; implemented `lock-snapshots`
   (pins dataset SHAs + model snapshots) → `validate()` now reports 0 warnings.
7. **Repo cleanup.** Consolidated everything under `latest/` (source of truth), kept
   only `README.md` + `CLAUDE_EFFORT.md` at the root.

---

## Methodology notes (how the work was kept correct)

- **Workflows for breadth, the main thread for coherence.** Fan-out (read/verify/
  design/review) ran as background multi-agent workflows; the actual code was written
  on the main thread for a coherent, compiling result.
- **Adversarial verification.** Every bug/finding from a reviewer was independently
  re-checked against the real code before being trusted or fixed (28 of the review's
  claims stood; the rest were downgraded or refuted).
- **Verify before claim.** Each layer was unit-tested offline and the providers were
  live-smoke-tested before building on top.

---

## Effort metrics

- 65 sub-agents across 2 workflows; ~4.4M total tokens.
- `latest/` package: ~45 Python modules + 5 config files + 3 data files.
- Tests: 26 offline (no keys/network) + live smoke per provider.
- Runs executed: 3 smoke runs (one full live, two cached re-validations) + a live
  connectivity preflight + `lock-snapshots`.
- Review: 28 issues fixed (5 high, ~7 med, ~16 low) — see `latest/CODEBASE.md` §14.

---

## Key decisions (and why)

- **Ledger-first architecture** — separates collection (facts) from analysis
  (findings); makes every number reproducible and re-analysis network-free.
- **Retire `Archive/`** — two divergent stacks are a publication hazard; consolidate.
- **Drop corrigibility arm** (user call) — kept arms = `pressure_wrong` + `control`.
- **Add `repeat` + `gauntlet` modes** (user idea) — complete the {one/all type} ×
  {one/all variant} 2×2; `gauntlet` reported as an endurance *ceiling*, not potency.
- **Name `latest`** (user call) — the canonical package.
- **Interactive `main.py`** with `--smoke`/`--comprehensive` flags + a budget gate
  that asks the user (resume vs stop) on a quota wall.

---

## Handoff: state & next steps

**Working now:** `pip install -r latest/requirements.txt`, set `latest/.env`, then
`python -m latest.main` (interactive) or `--smoke` / `--comprehensive`. Offline tests
green (26); `validate()` clean (revisions pinned, snapshots locked).

**Pre-publication to-dos (not bugs):**
- Run the full `--comprehensive` matrix (all 6 models) — `nano` was all-invalid on
  the 2 hard smoke questions, so scale `n` (items/subject, resamples) for valid
  per-model cells and McNemar pairs.
- Validate the LLM judge against human labels (κ was ~0.2 at smoke n).
- The science hardening list is in `latest/ARCHITECTURE_REVIEW.md` (power, CIs,
  multiple-comparison correction, selection-bias handling).

**To extend:** see `latest/CODEBASE.md` §12 (add a model/provider/attack/mode/scorer).
