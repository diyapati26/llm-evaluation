# Simpler Arch

Function-based reimplementation of the LLM eval framework. No classes, no
abstract base providers, no inheritance — just typed Pydantic answer
schemas and plain functions.

## Layout

```
Simpler Arch/
├── Output_Formats/output_format.py   # Pydantic answer schemas (per dataset)
├── utils/load_dataset.py             # Typed HF dataset loaders
├── providers/
│   ├── openai_provider.py            # get_openai_response() — chat.completions.parse
│   ├── anthropic_provider.py         # get_anthropic_response() — messages.parse
│   └── groq_provider.py              # get_groq_response() — gpt-oss-120b + json_schema
├── scorers.py                        # exact_match, rouge_l, truthfulqa_mc_score, llm_judge
├── cache.py                          # File-backed JSON cache (interop with notebook)
└── runner.py                         # Multi-model x multi-dataset loop, results JSON
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY
```

## Run

```bash
# Default: 20 samples per MMLU subject (160 total) on all 3 providers
python "Simpler Arch/runner.py"

# Specific datasets/models/sample size
python "Simpler Arch/runner.py" \
    --datasets mmlu hellaswag truthfulqa_mc \
    --models gpt-5.4-mini claude-sonnet-4-6 openai/gpt-oss-120b \
    --n 20

# Quick smoke test (5 samples)
python "Simpler Arch/runner.py" --datasets mmlu --n 5
```

Results land in `results/results_<timestamp>.json` with one row per
(model, dataset) combo: accuracy, total cost, avg latency, error count.
Cache lives in `AI_Cache.json` at the repo root — same key shape as the
notebook (`subject::model::question`), so cached responses are
interoperable.

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
| `truthfulqa_mc_score` | TruthfulQA-MC | 0.0 / 1.0 |
| `llm_judge_truthfulqa` | TruthfulQA generation truthful + informative | dict |
