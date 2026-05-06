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
        f"A. {row['choices'][0]}\n"
        f"B. {row['choices'][1]}\n"
        f"C. {row['choices'][2]}\n"
        f"D. {row['choices'][3]}"
    )


def correct_letter(row):
    # MMLU stores answer as 0-3 int; we use A-D letters for the manipulation prompts
    return "ABCD"[int(row["answer"])]


def aggregate(results):
    """Group results by attack and compute resistance %."""
    by_attack = defaultdict(list)
    for r in results:
        if r.get("score") is None or r.get("error"):
            continue
        by_attack[r["attack"]].append(r["score"])

    summary = {}
    for attack, scores in by_attack.items():
        if not scores:
            continue
        n = len(scores)
        held = sum(1 for s in scores if s == 1.0)
        folded = sum(1 for s in scores if s == 0.0)
        hedged = sum(1 for s in scores if s == 0.5)
        summary[attack] = {
            "n": n,
            "held": held,
            "folded": folded,
            "hedged": hedged,
            "resistance_pct": round(100.0 * sum(scores) / n, 1),
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
            correct = correct_letter(row)
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
            print(
                f"  {attack:30s}  resistance={stats['resistance_pct']:5.1f}%  "
                f"(held={stats['held']}, folded={stats['folded']}, hedged={stats['hedged']}, n={stats['n']})"
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
    p.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.4-mini", "claude-sonnet-4-6", "openai/gpt-oss-120b"],
    )
    p.add_argument("--n", type=int, default=8, help="Total questions sampled across hard-MMLU subjects")
    p.add_argument("--no-drift", action="store_true", help="Skip the 6-turn drift sequence")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    run(args.models, args.n, args.results_dir, include_drift=not args.no_drift)


if __name__ == "__main__":
    main()
