# Beyond Accuracy — LLM Robustness Evaluation Framework

A novel evaluation framework comparing frontier LLMs across standard 
benchmarks and three original research modules.

## Models evaluated
- GPT-5.4-mini (OpenAI, March 2026)
- Claude Sonnet 4.6 (Anthropic, February 2026)
- Llama 4 Scout (Meta via Groq — free)

## Modules

### 1. Standard Benchmark
MMLU, TruthfulQA, HellaSwag — 4 scorers: ExactMatch, ROUGE-L, 
BERTScore, LLM Judge

### 2. Manipulation Resistance ✅
8 attack types × hard MMLU subjects × true multi-turn conversations
with reasoning requirement. Includes 6-turn incremental drift.

Key finding: Claude Sonnet achieves 95% combined resistance vs 
GPT-5.4-mini at 78% and Llama 4 Scout at 22%.

### 3. Moral & Empathy 🔄
Preference dilemmas, ethical trolley problems, mental health crisis 
scenarios. Dataset grounded in CounselChat and CounselBench (2025).

### 4. Adversarial Debate 📋
Coming soon — model vs model position consistency testing.

## Setup

### 1. Clone the repo
\```bash
git clone <your-repo-url>
cd llm-robustness-eval
\```

### 2. Create virtual environment
\```bash
python3 -m venv venv
source venv/bin/activate
\```

### 3. Install dependencies
\```bash
pip install -r requirements.txt
\```

### 4. Set up API keys
\```bash
cp .env.example .env
# Edit .env and fill in your own API keys
\```

### 5. Get API keys
| Provider | Link | Cost |
|---|---|---|
| OpenAI | https://platform.openai.com/api-keys | ~$2 for full run |
| Anthropic | https://console.anthropic.com/ | ~$1 for full run |
| Groq | https://console.groq.com/keys | Free |
| HuggingFace | https://huggingface.co/settings/tokens | Free |

### 6. Verify setup
\```bash
python3 -m tests.test_providers
\```

All 3 should pass:
\```
✓ PASS  openai       gpt-5.4-mini
✓ PASS  anthropic    claude-sonnet-4-6
✓ PASS  groq         meta-llama/llama-4-scout-17b-16e-instruct
\```

## Running the evals

\```bash
# Standard benchmark
python3 run_eval.py

# Manipulation resistance
python3 run_manipulation.py

# Moral & empathy
python3 run_moral.py
\```

## Project structure
\```
evals/
  datasets/         — loaders, manipulation dataset, moral dataset
  providers/        — OpenAI, Anthropic, Groq wrappers
  scorers/          — ExactMatch, ROUGE, BERTScore, LLM Judge
  modules/          — manipulation runner, moral runner
  schemas.py        — Pydantic data models
  cache.py          — response cache
  runner.py         — async eval runner
results/
  cache/            — cached API responses (gitignored)
run_eval.py         — standard benchmark
run_manipulation.py — manipulation resistance
run_moral.py        — moral & empathy
tests/
  test_providers.py — verify all 3 providers work
\```

## Cost estimates per full run
\```
Standard benchmark (30 samples):
  GPT-5.4-mini:    ~$0.004
  Claude Sonnet:   ~$0.027
  Llama 4 Scout:   $0.000

Manipulation (280 tests + drift):
  GPT-5.4-mini:    ~$0.035
  Claude Sonnet:   ~$0.120
  Llama 4 Scout:   $0.000
\```

## Important notes for collaborators
- Cache is gitignored — each person builds their own locally
- First run makes real API calls, subsequent runs load from cache
- Groq has 100k daily token limit for large models
- Use llama-3.1-8b-instant for drift tests to preserve daily budget
- Never commit your .env file

## Research findings so far

### Standard benchmark
| Model | ExactMatch | LLM Judge | Cost/run |
|---|---|---|---|
| Claude Sonnet 4.6 | 0.700 | 0.933 | $0.027 |
| GPT-5.4-mini | 0.633 | 0.775 | $0.004 |
| Llama 4 Scout | 0.567 | 0.658 | $0.000 |

### Manipulation resistance (true multi-turn + reasoning)
| Model | Standard | Drift (6T) | Combined |
|---|---|---|---|
| Claude Sonnet 4.6 | 92% | 97% | 95% |
| GPT-5.4-mini | 74% | 81% | 78% |
| Llama 4 Scout | 36% | 13% | 22% |