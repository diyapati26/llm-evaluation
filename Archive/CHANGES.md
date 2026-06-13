# Changes — feature/DP

## Overview
Extended the manipulation resistance evaluation framework with two new testing modes (stateless and stateful), expanded subject coverage from 8 to 10 domains, and added full result persistence.

---

## 1. Stateless Mode
Each manipulation attack type runs in its own independent conversation session. After all attacks are tested, they are ranked by effectiveness (fold rate) — showing which attack type most reliably causes a model to surrender its correct answer.

**Outputs per model:**
- Resistance rate per subject per attack type
- Attack type effectiveness ranking (overall and per subject)
- Adjusted scores accounting for natural drift baseline

---

## 2. Stateful Mode (new)
All attack types are chained inside a **single conversation session** per sample. If the model resists an attack, the next attack type is applied in the same conversation — escalating pressure until the model surrenders or all 7 attacks are exhausted.

**Outputs per model:**
- Full resistance rate per subject (resisted all 7 attacks)
- Average number of attack turns before surrender
- Killing blow ranking — which attack type finally broke the model's resistance
- Per-subject killing blow breakdown

---

## 3. Expanded Subject Coverage (8 → 10)
Two additional MMLU subjects added to `HARD_SUBJECTS`:
- `clinical_knowledge`
- `formal_logic`

All scoring tables now report across 10 subjects.

---

## 4. CLI Mode Selection
`run_manipulation.py` now accepts a `--mode` argument:

```
python run_manipulation.py --mode stateless   # fresh session per attack
python run_manipulation.py --mode stateful    # chained attacks per session
python run_manipulation.py --mode both        # run all modes (default)
```

---

## 5. Result Persistence
Every run now automatically saves two files to `results/`:

- **`manipulation_<TIMESTAMP>.json`** — full raw results including every test outcome, per-subject scores, attack fold rates, natural drift values, and stateful killing blow counts
- **`manipulation_<TIMESTAMP>_report.txt`** — human-readable summary of scores per subject and model

Previously results were only printed to the terminal and lost after the session.

---

## Files Changed
| File | Change |
|---|---|
| `evals/datasets/manipulation_dataset.py` | Added `clinical_knowledge` and `formal_logic` to `HARD_SUBJECTS` |
| `evals/modules/manipulation_runner.py` | Added `run_stateful()` method; added imports for `ATTACK_TEMPLATES`, `get_wrong_answer`, `TaskType` |
| `run_manipulation.py` | Full rewrite — `--mode` arg, stateless attack ranking, stateful scoring, result saving via `_save_results()` |
| `CHANGES.md` | This file |
