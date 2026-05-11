"""Top-level manipulation runner.

Pulls hard-MMLU questions, runs every attack type (8 single-shot + 6-turn drift)
against each model, saves results JSON, prints resistance % per (model, attack).

Usage:
    python "Simpler Arch/run_manipulation.py" --models gpt-5.4-mini claude-sonnet-4-6 openai/gpt-oss-120b --n 5
"""
import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.load_dataset import load_mmlu_data_sample
from utils.load_config import load_config, models_from_config
import manipulation as mp

load_dotenv()

HARD_MMLU = [
    "professional_medicine",
    "international_law",
    "college_physics",
    "abstract_algebra",
    "professional_law",
    "college_chemistry",
    "high_school_statistics",
    "machine_learning",
]


def question_text(row):
    return (
        f"{row['question']}\n"
        f"1. {row['choices'][0]}\n"
        f"2. {row['choices'][1]}\n"
        f"3. {row['choices'][2]}\n"
        f"4. {row['choices'][3]}"
    )


def correct_number(row):
    # MMLU stores answer as 0-3 int; ReasonedAnswer uses '1'..'4' strings.
    return str(int(row["answer"]) + 1)


def aggregate(results):
    """Group results by attack and compute multi-axis stats.

    Returns per-attack dict with:
      n / held / folded / hedged           — letter-persistence counts
      resistance_pct                       — mean letter_persistence * 100
      avg_confidence_delta                 — mean (post - orig) confidence
      engagement_rate                      — % of responses that acknowledged the pushback
    """
    by_attack = defaultdict(list)
    by_attack_vec = defaultdict(list)  # full score vectors for the multi-axis stats
    for r in results:
        if r.get("error") or r.get("score") is None:
            continue
        by_attack[r["attack"]].append(r["score"])
        if "scores" in r:
            by_attack_vec[r["attack"]].append(r["scores"])

    summary = {}
    for attack, scores in by_attack.items():
        if not scores:
            continue
        n = len(scores)
        held = sum(1 for s in scores if s == 1.0)
        folded = sum(1 for s in scores if s == 0.0)
        hedged = sum(1 for s in scores if s == 0.5)

        # Multi-axis aggregates (skip None / missing fields defensively)
        vecs = by_attack_vec[attack]
        conf_deltas = [v["confidence_delta"] for v in vecs if v.get("confidence_delta") is not None]
        engagements = [v["engagement"] for v in vecs if v.get("engagement") is not None]
        avg_conf_delta = round(sum(conf_deltas) / len(conf_deltas), 2) if conf_deltas else None
        engagement_rate = round(100.0 * sum(engagements) / len(engagements), 1) if engagements else None

        summary[attack] = {
            "n": n,
            "held": held,
            "folded": folded,
            "hedged": hedged,
            "resistance_pct": round(100.0 * sum(scores) / n, 1),
            "avg_confidence_delta": avg_conf_delta,
            "engagement_rate": engagement_rate,
        }
    return summary


def run(models, n_questions, results_dir="results", include_drift=True):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    print(f"\n=== loading {n_questions} hard-MMLU questions per subject ===")
    df = load_mmlu_data_sample(subjects=HARD_MMLU, max_per_subject=max(1, n_questions // len(HARD_MMLU)))
    print(f"   total questions: {len(df)}")

    all_results = []
    by_model = {}

    for model in models:
        print(f"\n--- {model} ---")
        model_results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=model):
            qtext = question_text(row)
            correct = correct_number(row)
            try:
                attack_results = mp.run_all_attacks(model, qtext, correct, include_drift=include_drift)
                for r in attack_results:
                    r["question_id"] = row.name
                    r["subject"] = row.get("subject", "unknown")
                model_results.extend(attack_results)
            except Exception:
                traceback.print_exc()

        summary = aggregate(model_results)
        by_model[model] = summary
        all_results.extend(model_results)

        # Quick per-model print
        print(f"\n{model} resistance summary:")
        for attack, stats in sorted(summary.items()):
            cd = stats.get("avg_confidence_delta")
            er = stats.get("engagement_rate")
            cd_str = f"{cd:+.2f}" if cd is not None else "  n/a"
            er_str = f"{er:5.1f}%" if er is not None else "  n/a"
            print(
                f"  {attack:30s}  resistance={stats['resistance_pct']:5.1f}%  "
                f"(held={stats['held']}, folded={stats['folded']}, hedged={stats['hedged']}, n={stats['n']})  "
                f"conf_delta={cd_str}  engagement={er_str}"
            )
        if summary:
            combined = sum(s["resistance_pct"] for s in summary.values()) / len(summary)
            print(f"  {'COMBINED':30s}  resistance={combined:5.1f}%")

    # Write detailed results
    out_path = os.path.join(results_dir, f"manipulation_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "n_questions": len(df),
                "by_model": by_model,
                "raw_results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")

    return by_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="Path to YAML config (default: Simpler Arch/configs/eval_config.yaml)")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n", type=int, default=None,
                   help="Total questions sampled across hard-MMLU subjects")
    p.add_argument("--no-drift", action="store_true", help="Skip the 6-turn drift sequence")
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    manip = cfg.get("manipulation", {})

    models = args.models or models_from_config(cfg) or [
        "gpt-5.4-mini", "claude-sonnet-4-6", "openai/gpt-oss-120b"
    ]
    n = args.n if args.n is not None else manip.get("n", 8)
    results_dir = args.results_dir or cfg.get("results_dir", "results")
    # CLI --no-drift always wins; otherwise honor config.
    include_drift = (not args.no_drift) and manip.get("include_drift", True)

    run(models, n, results_dir, include_drift=include_drift)


if __name__ == "__main__":
    main()
