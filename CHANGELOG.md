# Changelog — Manipulation Resistance Module

## [2026-05-09] — Research paper improvements: natural drift correction, prompt variants, statistical testing

---

### 1. Per-subject breakdown with natural drift correction

**Files:** `run_manipulation.py`, `evals/modules/manipulation_runner.py`

**What changed:**
- Results are now reported per MMLU subject (abstract_algebra, college_chemistry, etc.) instead of as a single aggregate score.
- A new control condition (`neutral_control`) measures each model's baseline answer volatility per subject.
- The final score is `resistance_rate − natural_drift_rate`, computed independently per model × subject.

**Why:**
A single aggregate resistance score conflates two different phenomena: (a) genuine resistance to manipulation pressure, and (b) general answer stability. A model that naturally changes its answer 20% of the time under any follow-up — even a neutral one — should not receive the same score as a model that only flips under adversarial pressure. Subtracting the per-subject natural drift rate isolates the net manipulation effect.

Per-subject reporting is also methodologically necessary because model confidence varies by domain: a model may be highly stable on college_chemistry but volatile on moral_scenarios. Aggregating these hides domain-specific failure modes that are meaningful to report in a research paper.

---

### 2. Four prompt variants per attack type (Item 5)

**Files:** `evals/datasets/manipulation_dataset.py`

**What changed:**
- `ATTACK_TEMPLATES` changed from `{attack_type: str}` to `{attack_type: list[str]}`.
- All 7 attack types and `neutral_control` now have 4 phrasing variants each.
- `build_manipulation_samples` expands each sample × attack_type × variant, stamping `variant_idx` on every test.
- `incremental_drift` intentionally keeps 1 variant — its attack_prompt is not used by `run_drift_test`, which drives turns 2–6 through `DRIFT_TURNS` directly.
- Test count: 40 samples × 7 attacks × 4 variants = 1,120 standard tests per model (up from 280). Control: 40 × 4 = 160 per model (up from 40), giving 20 observations per subject for natural drift.

**Why:**
With a single template per attack type, results reflect the model's response to *that specific phrasing*, not the attack class. A reviewer can always ask: "does this hold if you rephrase the authority pressure attack?" With 4 variants averaged, the reported resistance rate is an estimate of the model's resistance to the attack class as a whole. The standard deviation across variants is reported separately as a *phrasing sensitivity* metric — a high std means the model's resistance is brittle and depends heavily on wording.

---

### 3. `variant_idx` propagated through results

**Files:** `evals/modules/manipulation_runner.py`

**What changed:**
- `variant_idx` added to result dicts in both `_run_provider` (standard attacks) and `run_one_drift` (drift tests).
- Per-test print now shows `[v0]`/`[v1]`/`[v2]`/`[v3]` suffix so runs are traceable.
- `_print_summary` now groups results by `attack_type` and reports mean ± std across variants instead of a flat rate.

**Why:**
`variant_idx` is required in the result dict to (a) correctly aggregate mean/std across variants in the leaderboard, and (b) construct matched pairs for McNemar's test — pairing model A and model B on the exact same (sample, attack_type, variant) triple.

---

### 4. Temperature pinned to 0.0, model version logged (Item 3)

**Files:** `evals/schemas.py`, `evals/providers/anthropic_provider.py`, `evals/providers/openai_provider.py`, `evals/providers/groq_provider.py`

**What changed:**
- `LLMResponse` gains two new fields: `model_version: Optional[str]` and `temperature: float`. Both have defaults so cached responses without these fields deserialize without error.
- All three providers now populate `model_version` from the actual API response (e.g., `gpt-4.1-mini-2025-04-14` rather than just `gpt-4.1-mini`).
- **Bug fix:** `AnthropicProvider` stored `self.temperature` but never passed it to `client.messages.create()`. Both `generate()` and `generate_conversation()` now correctly pass `temperature=self.temperature`.

**Why:**
Temperature controls output randomness. Without pinning it, two runs of the same experiment can produce different results — especially for models with different API defaults. Temperature 0.0 (greedy decoding) makes runs deterministic and reproducible, which is a baseline requirement for a published evaluation.

Model version is separate from model name: `gpt-4.1-mini` is a routing alias that silently maps to a specific dated snapshot (e.g., `gpt-4.1-mini-2025-04-14`). That snapshot can change without the alias changing, making results from two different dates incomparable. Logging the version string returned by the API makes the paper's experimental setup reproducible months later.

---

### 5. McNemar's test for pairwise model comparison (Item 7)

**Files:** `run_manipulation.py`, `requirements.txt`

**What changed:**
- `scipy>=1.10.0` added to requirements.
- `_mcnemar_compare()` helper builds matched pairs on `(sample_id, attack_type, variant_idx)` keys and runs McNemar's test with Yates' continuity correction (`exact=False, correction=True`).
- A new section 12 in `main()` prints pairwise comparisons for all model pairs, reporting χ², p-value, significance stars (`*` p<0.05, `**` p<0.01, `***` p<0.001), and the direction of the effect.
- Degrades gracefully if scipy is not installed.

**Why:**
Without a significance test, a claim like "Claude resists 65% of attacks vs. GPT's 43%" is a descriptive observation, not a statistical finding. A reviewer will ask whether the difference is significant or within noise. McNemar's test is the correct choice here because:
- The same questions are posed to all models (paired design), so it is more powerful than an independent-samples test.
- The outcome is binary (resistant / not resistant), which is exactly the case McNemar's handles.
- With 1,120 matched pairs, it has sufficient power to detect differences of ~5 percentage points.

---

### 6. Other fixes

**Files:** `evals/providers/openai_provider.py`, `run_manipulation.py`

- **Model name typo fixed:** `gpt-5.4-mini` → `gpt-4.1-mini` (both in `OpenAIProvider.__init__` default and in `run_manipulation.py`). `gpt-5.4-mini` does not exist and would crash at runtime.
- **Removed unused `load_all` call:** `standard_samples = load_all(max_per_dataset=10)` loaded 30 samples that were never used, wasting API/HuggingFace calls on every run.
- **Merged duplicate provider lists:** `providers_standard` and `providers_drift` were identical; collapsed into a single `providers` list with a single `runner`.
- **Stale comment removed:** Footer note referenced `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` — models that were replaced in the code with `llama-4-scout-17b-16e-instruct`.
