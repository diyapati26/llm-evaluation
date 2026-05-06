# Simpler Arch

Function-based reimplementation of the LLM eval framework. No classes, no
abstract base providers, no inheritance — just typed Pydantic answer
schemas and plain functions.

Three modules, each with a top-level CLI runner:

| Module | Runner | What it tests |
| --- | --- | --- |
| **Standard benchmark** | `runner.py` | MMLU / HellaSwag / TruthfulQA-MC / TruthfulQA-gen accuracy |
| **Manipulation resistance** | `run_manipulation.py` | Does the model fold under social pressure? 8 attacks + 6-turn drift |
| **Moral / empathy** | `run_moral.py` | Preference dilemmas, ethical tradeoffs, crisis responses, scored by LLM-judge |

## Layout

```
Simpler Arch/
├── Output_Formats/output_format.py   # Pydantic answer schemas (MMLU, HellaSwag, TruthfulQA-MC, TruthfulQA-gen)
├── utils/load_dataset.py             # Typed HF dataset loaders
├── providers/
│   ├── openai_provider.py            # get_openai_response(parse) + get_openai_chat(multi-turn)
│   ├── anthropic_provider.py         # get_anthropic_response(parse) + get_anthropic_chat(multi-turn)
│   └── groq_provider.py              # get_groq_response(json_schema) + get_groq_chat(multi-turn)
├── scorers.py                        # exact_match, rouge_l, truthfulqa_mc_score, llm_judge_truthfulqa
├── manipulation.py                   # 8 attacks + 6-turn drift; run_single_attack, run_drift_attack, run_all_attacks
├── moral.py                          # 9 scenarios across 3 categories; 4-axis LLM-judge scoring
├── cache.py                          # File-backed JSON cache (interop with notebook)
├── runner.py                         # Standard benchmark CLI
├── run_manipulation.py               # Manipulation CLI (hard-MMLU)
├── run_moral.py                      # Moral / empathy CLI
├── compare.py                        # Read all results/*.json and print accuracy/cost/latency table
└── tests/
    └── test_providers.py             # 3-provider connectivity smoke test
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY
python "Simpler Arch/tests/test_providers.py"   # confirm all 3 providers reach
```

## Run

### Standard benchmark

```bash
# 20 samples per MMLU subject (160 total) across the 3 providers
python "Simpler Arch/runner.py"

# Pick datasets / models / sample size
python "Simpler Arch/runner.py" \
    --datasets mmlu hellaswag truthfulqa_mc truthfulqa_gen \
    --models gpt-5.4-mini claude-sonnet-4-6 openai/gpt-oss-120b \
    --n 20

# Quick smoke (5 samples per subject for MMLU)
python "Simpler Arch/runner.py" --datasets mmlu --n 5
```

### Manipulation resistance

```bash
# 8 attacks + 6-turn drift on 8 hard-MMLU questions per model
python "Simpler Arch/run_manipulation.py" --n 8

# Skip the 6-turn drift if you want a faster run
python "Simpler Arch/run_manipulation.py" --n 8 --no-drift
```

### Moral / empathy

```bash
# All 9 scenarios x N models, judged by gpt-5.4-mini by default
python "Simpler Arch/run_moral.py"

# Pick a different judge
python "Simpler Arch/run_moral.py" --judge-model claude-sonnet-4-6
```

### Compare runs

```bash
# Latest per (model, dataset)
python "Simpler Arch/compare.py"

# Filter by dataset substring
python "Simpler Arch/compare.py" --filter mmlu
```

Results land in `results/` as JSON. `runner.py` writes `results_<timestamp>.json`,
manipulation writes `manipulation_<timestamp>.json`, moral writes `moral_<timestamp>.json`.
Cache lives in `AI_Cache.json` at the repo root — same key shape as the notebook
(`subject::model::question`), so cached responses are interoperable.

## Why function-based

- Each provider is ~70 lines, three short functions; no inheritance to chase.
- Pydantic answer classes (MMLU_Answer, HellaSwag_Answer, etc.) live in
  one file and are reused across providers — they're the contract.
- The runner's `route(model)` dispatches on model-name prefix:
    `gpt-*` / `o1-*` / `o3-*` → OpenAI
    `claude-*` → Anthropic
    `openai/gpt-oss-*` / `meta-llama/*` / `llama*` → Groq

## Why gpt-oss-120b instead of Llama 4 Scout

gpt-oss-120b honors `response_format=json_schema` natively on Groq, so
no regex post-processing needed (Llama 4 Scout was returning
"Answer: B because..." even with the system-prompt hack — needed manual
letter extraction). Same `get_groq_response()` shape as the others.

## Scorers

| Scorer | Use for | Returns |
| --- | --- | --- |
| `exact_match` | MMLU, HellaSwag, TruthfulQA-MC fallback | 0.0 / 1.0 |
| `rouge_l` | TruthfulQA generation reference overlap | 0.0–1.0 |
| `truthfulqa_mc_score` | TruthfulQA-MC (uses `mc1_targets.labels`) | 0.0 / 1.0 |
| `llm_judge_truthfulqa` | TruthfulQA generation — truthful + informative axes | dict |
| `score_resistance` (manipulation) | 1.0 = held, 0.5 = hedged, 0.0 = folded, None = invalid baseline | float / None |
| Moral 4-axis judge | helpfulness / empathy / safety / reasoning, 1-5 scale | dict |
