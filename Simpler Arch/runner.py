"""Function-based runner — no classes.

Loops models x datasets x samples, dispatches to the right provider,
caches responses, scores, and writes a results JSON for comparison.

Usage (from repo root, with venv active):
    python "Simpler Arch/runner.py" --datasets mmlu --models gpt-5.4-mini claude-sonnet-4-6 openai/gpt-oss-120b --n 20
"""
import argparse
import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

import cache as cache_mod
import scorers
from load_config import load_config, models_from_config
from Output_Formats.output_format import (
    HellaSwag_Answer,
    MMLU_Answer,
    ProviderResponse,
    TruthfulQA_Generation_Answer,
    TruthfulQA_MC_Answer,
)
from providers import (
    anthropic_provider,
    groq_provider,
    openai_provider,
    openrouter_provider,
)
from utils.load_dataset import (
    load_hellaswag_data_sample,
    load_mmlu_data_sample,
    load_truthfulqa_data_sample,
)

load_dotenv()

# ── Provider routing ──────────────────────────────────────────────


_EXPLICIT_PROVIDERS = {
    "openai":     (openai_provider,     "openai"),
    "anthropic":  (anthropic_provider,  "anthropic"),
    "groq":       (groq_provider,       "groq"),
    "openrouter": (openrouter_provider, "openrouter"),
}


def route(model):
    """Return (provider_module, provider_name, real_model) for a given model id.

    Explicit form wins: 'openrouter:anthropic/claude-3.5-sonnet' → openrouter provider,
    real model = 'anthropic/claude-3.5-sonnet'. The 'openai/' and 'meta-llama/'
    prefixes contain '/' but no ':', so they hit prefix inference (Groq-hosted).

    Unprefixed names fall back to prefix inference for back-compat with existing
    --models arguments and cached entries.
    """
    # Explicit "provider:model" — disambiguates collisions between hosts
    if ":" in model:
        provider_name, real_model = model.split(":", 1)
        if provider_name in _EXPLICIT_PROVIDERS:
            mod, name = _EXPLICIT_PROVIDERS[provider_name]
            return mod, name, real_model
        raise ValueError(
            f"Unknown provider prefix '{provider_name}'. "
            f"Known: {sorted(_EXPLICIT_PROVIDERS)}"
        )

    # Prefix inference (back-compat for unprefixed names)
    if model.startswith(("gpt-", "o1", "o3")):
        return openai_provider, "openai", model
    if model.startswith("claude-"):
        return anthropic_provider, "anthropic", model
    if model.startswith(("openai/gpt-oss", "meta-llama/", "llama")):
        return groq_provider, "groq", model
    raise ValueError(
        f"Don't know how to route model: {model}. "
        f"Use 'provider:model' syntax (e.g. 'openrouter:qwen/qwen-2.5-72b')."
    )


def call(model, prompt, output_format):
    provider, name, real_model = route(model)
    if name == "openai":
        return provider.get_openai_response(prompt, real_model, output_format)
    if name == "anthropic":
        return provider.get_anthropic_response(prompt, real_model, output_format)
    if name == "groq":
        return provider.get_groq_response(prompt, real_model, output_format)
    if name == "openrouter":
        return provider.get_openrouter_response(prompt, real_model, output_format)
    raise ValueError(f"Unknown provider name from route(): {name!r}")


# ── Prompt builders ───────────────────────────────────────────────


def mmlu_prompt(row):
    return (
        f"Answer the following multiple-choice question.\n\n"
        f"Question: {row['question']}\n"
        f"1. {row['choices'][0]}\n"
        f"2. {row['choices'][1]}\n"
        f"3. {row['choices'][2]}\n"
        f"4. {row['choices'][3]}\n\n"
        f"Reply with the single number 1, 2, 3, or 4."
    )


def hellaswag_prompt(row):
    endings = row["endings"]
    return (
        f"Pick the most plausible ending for this scenario.\n\n"
        f"Activity: {row.get('activity_label', '')}\n"
        f"Context: {row.get('ctx', row.get('ctx_a',''))}\n\n"
        f"1. {endings[0]}\n"
        f"2. {endings[1]}\n"
        f"3. {endings[2]}\n"
        f"4. {endings[3]}\n\n"
        f"Reply with the single number 1, 2, 3, or 4."
    )


def truthfulqa_mc_prompt(row):
    choices = row["mc1_targets"]["choices"]
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(choices))
    return (
        f"Pick the most truthful answer (avoid common misconceptions).\n\n"
        f"Question: {row['question']}\n\n"
        f"{numbered}\n\n"
        f"Reply with a single number from 1 to {len(choices)}."
    )


def truthfulqa_gen_prompt(row):
    return (
        f"Answer the question truthfully and concisely (1-2 sentences max).\n\n"
        f"Question: {row['question']}"
    )


# ── Scoring per-task ──────────────────────────────────────────────


def score_mmlu(row, parsed_answer):
    correct_idx = int(row["answer"])
    correct_str = str(correct_idx + 1)
    return scorers.exact_match(parsed_answer, correct_str)


def score_hellaswag(row, parsed_answer):
    correct_idx = int(row["label"])
    correct_str = str(correct_idx + 1)
    return scorers.exact_match(parsed_answer, correct_str)


def score_truthfulqa_mc(row, parsed_answer):
    return scorers.truthfulqa_mc_score(parsed_answer, row["mc1_targets"])


def score_truthfulqa_gen(row, parsed_answer):
    """Free-form TruthfulQA scoring: max ROUGE-L vs each acceptable correct_answer."""
    correct = row.get("correct_answers")
    # HF dataset stores correct_answers as a list (or sometimes single best_answer string)
    if correct is None or len(correct) == 0:
        correct = [row.get("best_answer", "")]
    return scorers.rouge_l_max(parsed_answer, list(correct))


# ── Datasets ──────────────────────────────────────────────────────

_DS_CFG = load_config().get("dataset", {})
HARD_MMLU = _DS_CFG.get("hard_mmlu", [])
RANDOM_STATE = _DS_CFG.get("random_state", 42)


def get_dataset(name, n):
    if name == "mmlu":
        df = load_mmlu_data_sample(
            subjects=HARD_MMLU,
            max_per_subject=max(1, n // len(HARD_MMLU)),
            random_state=RANDOM_STATE,
        )
        return df, MMLU_Answer, mmlu_prompt, score_mmlu, "subject"
    if name == "hellaswag":
        df = load_hellaswag_data_sample(max_samples=n, random_state=RANDOM_STATE)
        df["subject"] = "hellaswag"
        return df, HellaSwag_Answer, hellaswag_prompt, score_hellaswag, "subject"
    if name == "truthfulqa_mc":
        df = load_truthfulqa_data_sample(config="multiple_choice", max_samples=n, random_state=RANDOM_STATE)
        df["subject"] = "truthfulqa_mc"
        return df, TruthfulQA_MC_Answer, truthfulqa_mc_prompt, score_truthfulqa_mc, "subject"
    if name == "truthfulqa_gen":
        df = load_truthfulqa_data_sample(config="generation", max_samples=n, random_state=RANDOM_STATE)
        df["subject"] = "truthfulqa_gen"
        return df, TruthfulQA_Generation_Answer, truthfulqa_gen_prompt, score_truthfulqa_gen, "subject"
    raise ValueError(f"Unknown dataset: {name}")


# ── Main loop ─────────────────────────────────────────────────────


def run(datasets, models, n, results_dir="results"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    cache = cache_mod.load_cache()
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    all_results = []

    for ds_name in datasets:
        print(f"\n=== loading {ds_name} ===")
        df, output_format, prompt_fn, score_fn, subject_col = get_dataset(ds_name, n)
        print(f"   {len(df)} samples")

        for model in models:
            print(f"\n--- {model} on {ds_name} ---")
            scores_list = []
            costs = []
            latencies = []
            errors = 0

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model}"):
                subject = row.get(subject_col, ds_name)
                prompt = prompt_fn(row)
                cached = cache_mod.get(cache, subject, model, prompt)

                if cached is not None:
                    # Cache stores plain dicts (JSON-friendly). Wrap back into
                    # ProviderResponse so attribute access works downstream.
                    resp = ProviderResponse(**cached) if isinstance(cached, dict) else cached
                else:
                    try:
                        resp = call(model, prompt, output_format)
                    except Exception:
                        errors += 1
                        traceback.print_exc()
                        continue
                    # Dump to dict for JSON-friendly cache storage.
                    cache_mod.put(cache, subject, model, prompt, resp.model_dump())

                try:
                    s = score_fn(row, resp.answer)
                except Exception:
                    s = 0.0
                scores_list.append(s)
                costs.append(resp.cost_usd)
                latencies.append(resp.latency_ms)

            cache_mod.save_cache(cache)

            n_scored = len(scores_list)
            accuracy = sum(scores_list) / n_scored if n_scored else 0.0
            total_cost = sum(costs)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

            row_result = {
                "run_id": run_id,
                "model": model,
                "dataset": ds_name,
                "n_samples": len(df),
                "n_scored": n_scored,
                "errors": errors,
                "accuracy": round(accuracy, 4),
                "total_cost_usd": round(total_cost, 6),
                "avg_latency_ms": round(avg_latency, 2),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            all_results.append(row_result)
            print(
                f"   accuracy={row_result['accuracy']:.3f}  "
                f"cost=${row_result['total_cost_usd']:.4f}  "
                f"avg_latency={row_result['avg_latency_ms']:.0f}ms  "
                f"errors={errors}"
            )

    out_path = os.path.join(results_dir, f"results_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")
    return all_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None,
                   choices=["mmlu", "hellaswag", "truthfulqa_mc", "truthfulqa_gen"])
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n", type=int, default=None,
                   help="Samples per dataset (per subject for MMLU)")
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()

    # Config-driven defaults; CLI overrides.
    cfg = load_config()
    bench = cfg.get("benchmark", {})

    datasets = args.datasets or bench.get("datasets") or ["mmlu"]
    models = args.models or models_from_config(cfg) or [
        "gpt-5.4-mini", "claude-sonnet-4-6", "openai/gpt-oss-120b"
    ]
    n = args.n if args.n is not None else bench.get("n", 20)
    results_dir = args.results_dir or cfg.get("results_dir", "results")

    run(datasets, models, n, results_dir)


if __name__ == "__main__":
    main()
