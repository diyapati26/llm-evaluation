# Simpler Arch

Function-based LLM eval framework. No abstract base classes, no inheritance — just typed Pydantic schemas, plain functions, and a YAML config that drives everything.

Three eval modules, all configured from one place:

| Module | Runner | What it tests |
| --- | --- | --- |
| **Standard benchmark** | `runner.py` | MMLU / HellaSwag / TruthfulQA-MC / TruthfulQA-gen accuracy |
| **Manipulation resistance** | `run_manipulation.py` | 8 social-pressure attacks + 6-turn drift on hard-MMLU questions |
| **Moral / empathy** | `run_moral.py` | Preference dilemmas, ethical tradeoffs, crisis responses — 4-axis LLM-judge scoring |

Four providers supported, swappable from YAML:

| Provider | Models tested | API surface |
| --- | --- | --- |
| **OpenAI** | gpt-5.4-mini, gpt-5.4-nano, gpt-4.1-mini | Responses API + native Conversations (server-side state) |
| **Anthropic** | claude-sonnet-4-6, claude-haiku-4-5, claude-opus-4-7 | `messages.parse(output_format=...)` (client-side history) |
| **Groq** | openai/gpt-oss-120b, openai/gpt-oss-20b, llama-3.3-70b | OpenAI-compatible chat.completions + strict json_schema |
| **OpenRouter** | 200+ models (Anthropic, Meta, Mistral, DeepSeek, Qwen, xAI…) | OpenAI-compatible, strict-only json_schema |

## Layout

```
Simpler Arch/
├── config/config.yaml            # models per provider + dataset + benchmark/manipulation/moral settings
├── data/
│   ├── manipulation_attacks.jsonl   # 8 attack templates
│   ├── manipulation_drift.jsonl     # 5 progressive drift turns
│   └── moral_scenarios.jsonl        # 9 scenarios across 3 categories
├── schemas.py                       # Pydantic schemas: MMLU_Answer, …, ReasonedAnswer, ProviderResponse
├── providers/
│   ├── openai_provider.py           # get_openai_response (parse) + get_openai_chat (free-form)
│   ├── anthropic_provider.py        # get_anthropic_response + get_anthropic_chat
│   ├── groq_provider.py             # get_groq_response + get_groq_chat
│   ├── openrouter_provider.py       # strict-mode only; live pricing from /v1/models
│   └── conversation.py              # Conversation hierarchy + chat() dispatcher + _resolve()
├── utils/
│   ├── load_dataset.py              # HuggingFace loaders (MMLU/HellaSwag/TruthfulQA)
│   └── load_local.py                # JSONL loaders (attacks/drift/moral)
├── load_config.py                   # YAML config loader
├── cache.py                         # File-backed JSON cache (interop with notebook AI_Cache.json)
├── scorers.py                       # exact_match, rouge_l, truthfulqa_mc, llm_judge, multi-axis resistance
├── manipulation.py                  # run_single_attack / run_drift_attack / run_all_attacks
├── moral.py                         # 9 scenarios, 4-axis LLM-judge scoring
├── runner.py                        # Standard benchmark CLI
├── run_manipulation.py              # Manipulation CLI
├── run_moral.py                     # Moral / empathy CLI
├── test_providers.py                # 'Hi' smoke test across all 4 providers
├── ruff.toml                        # lint config (E,F,W,I,B,UP,SIM,C4,RET — line-length=120)
└── pyrightconfig.json               # type-check config (basic, ML-SDK noise suppressed)
```

## Setup

```bash
# From repo root
pip install -r requirements.txt
cp .env.example .env
# Fill OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY in .env
```

Smoke test all providers (sends "Hi", expects a non-empty response):

```bash
cd "Simpler Arch"
python test_providers.py
```

## Run

**Important:** every runner expects CWD = `Simpler Arch/` (so `config/config.yaml` and `data/*.jsonl` resolve via plain relative paths).

```bash
cd "Simpler Arch"
```

### Standard benchmark

```bash
python runner.py
# Uses config defaults: 4 providers × 4 datasets × 20 samples per dataset

# CLI flags override the YAML
python runner.py --datasets mmlu --n 5
python runner.py --models openai:gpt-5.4-nano anthropic:claude-haiku-4-5
```

### Manipulation resistance

```bash
python run_manipulation.py
# Default: 8 hard-MMLU questions × 8 attacks + 6-turn drift per model

python run_manipulation.py --no-drift           # skip the 6-turn drift
python run_manipulation.py --n 4                # smaller question set
```

### Moral / empathy

```bash
python run_moral.py
# 9 scenarios × N models, judged on 4 axes (helpfulness, empathy, safety, reasoning)

python run_moral.py --judge-model anthropic:claude-haiku-4-5
```

Results land in `results/` as JSON. `runner.py` writes `results_<ts>.json`, manipulation writes `manipulation_<ts>.json`, moral writes `moral_<ts>.json`. Cache lives in `AI_Cache.json` (same `subject::model::question` key shape as the notebook — cached responses are interoperable).

## Editing the config

`config/config.yaml` drives every runner. Add models per provider — every runner loops over the flattened list:

```yaml
models:
  openai:
    - gpt-5.4-mini
    - gpt-5.4-nano
  anthropic:
    - claude-sonnet-4-6
  groq:
    - openai/gpt-oss-120b
  openrouter:
    - meta-llama/llama-3.3-70b-instruct
    - qwen/qwen-2.5-72b-instruct
    - deepseek/deepseek-chat

judge_model: gpt-5.4-mini

dataset:
  random_state: 42
  max_samples: 20
  hard_mmlu: [professional_medicine, college_physics, …]

benchmark:
  datasets: [mmlu, hellaswag, truthfulqa_mc, truthfulqa_gen]
  n: 20

manipulation:
  n: 8
  include_drift: true
```

Each entry flattens to a `provider:model` string. The router (`providers/conversation._resolve`) dispatches by the prefix — so the same underlying model accessed via different hosts (e.g., `meta-llama/llama-3.3-70b` on Groq vs. OpenRouter) gets its own cache slot and its own row in the results JSON.

## Editing the datasets

- `data/manipulation_attacks.jsonl` — 8 attack templates. Add a row to define a new attack.
- `data/manipulation_drift.jsonl` — 5 progressive pressure turns for the drift sequence.
- `data/moral_scenarios.jsonl` — 9 moral scenarios (3 categories). Add rows with a `category` field to extend.

No code changes needed for any of these.

## Architecture

**Provider responses are unified by `ProviderResponse`** (`Output_Formats/output_format.py`) — every call (structured or free-form chat) returns the same Pydantic envelope with `answer`, `text`, `cost_usd`, `latency_ms`, `input_tokens`, `output_tokens`.

**Multi-turn flows go through `Conversation`** (`providers/conversation.py`) — three subclasses hide the API asymmetry:
- `OpenAIConversation` — server-side state via `conversations.create()` + `responses.parse(conversation=id, ...)`
- `AnthropicConversation` — client-side messages list, no server state (Anthropic API doesn't expose it)
- `ChatCompletionsConversation` — client-side messages + strict `response_format=json_schema`; used by Groq and OpenRouter

The Groq probe confirmed `/v1/responses` works on Groq but `/v1/conversations` does not — that's why Groq joins the `ChatCompletionsConversation` path.

**Single-turn free-form calls go through `chat()`** (`providers/conversation.chat`) — used by moral scenarios and LLM-judge scoring.

**Scoring is multi-axis** (`scorers.py`) for manipulation:
- `score_letter_persistence` — 1.0 / 0.5 / 0.0 — held / hedged (UNCERTAIN) / folded
- `score_confidence_delta` — change in self-reported 1–5 confidence
- `score_engagement` — did the model acknowledge the user's pushback?
- `score_resistance_vector` — combined dict across all axes

## Lint + type check

```bash
cd "Simpler Arch"
python -m ruff check .       # ruff.toml: E,F,W,I,B,UP,SIM,C4,RET; line-length=120; E501 ignored
python -m pyright .          # pyrightconfig.json: basic mode, ML-SDK noise suppressed
```

Both should be clean.

## Provider routing reference

| Model string in YAML / CLI | Routes to |
| --- | --- |
| `openai:gpt-5.4-mini` | OpenAI direct |
| `anthropic:claude-sonnet-4-6` | Anthropic direct |
| `groq:openai/gpt-oss-120b` | Groq endpoint |
| `openrouter:qwen/qwen-2.5-72b` | OpenRouter gateway |
| `gpt-5.4-mini` (unprefixed) | Prefix inference → OpenAI |
| `claude-sonnet-4-6` (unprefixed) | Prefix inference → Anthropic |
| `openai/gpt-oss-120b` (unprefixed) | Prefix inference → Groq |

Always prefer the explicit `provider:model` form for cross-host disambiguation — e.g., `meta-llama/llama-3.3-70b` exists on both Groq and OpenRouter at different prices.

## Cache

`cache.py` reads/writes `AI_Cache.json` (CWD-relative — created in `Simpler Arch/` when you run from there). Cached responses skip the API call entirely on repeat runs, which matters for paper reproducibility and cost.

If you want to share the notebook's existing `<repo-root>/AI_Cache.json` (1266 entries), either symlink it into `Simpler Arch/AI_Cache.json` or change `cache.DEFAULT_PATH` to point at the repo-root file.
