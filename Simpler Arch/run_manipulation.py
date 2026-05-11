"""Top-level manipulation runner.

Pulls hard-MMLU questions, runs every (attack × phrasing variant) against each
model, plus the 6-turn drift sequence. Reports:

  - Per-subject leaderboard with natural-drift correction
        adjusted_resistance = raw_resistance − natural_drift_rate
    (drift rate is measured per subject from neutral_control attacks — no-pressure
     rephrasings that should produce no flips; any flips reveal baseline volatility.)
  - Phrasing-sensitivity std across the 4 variants of each attack (high std =
    model is brittle to how the attack is phrased).
  - Pairwise McNemar's test on matched (sample_id, attack, variant_idx) triples
    to detect statistically-significant differences between models.

Usage:
    python run_manipulation.py
    python run_manipulation.py --n 8 --no-drift
"""
import argparse
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

import manipulation as mp
from load_config import load_config, models_from_config
from utils.load_dataset import load_mmlu_data_sample
from utils.stats import (
    per_subject_drift_rate,
    per_subject_resistance_stats,
    print_pairwise_mcnemar,
)

load_dotenv()

_DS_CFG = load_config()["dataset"]
HARD_MMLU = _DS_CFG["hard_mmlu"]
RANDOM_STATE = _DS_CFG["random_state"]

CONTROL_ATTACK = "neutral_control"
DRIFT_ATTACK = "incremental_drift_6_turn"


def question_text(row):
    return (
        f"{row['question']}\n"
        f"1. {row['choices'][0]}\n"
        f"2. {row['choices'][1]}\n"
        f"3. {row['choices'][2]}\n"
        f"4. {row['choices'][3]}"
    )


def correct_number(row):
    """MMLU stores answer as 0-3 int; ReasonedAnswer uses '1'..'4' strings."""
    return str(int(row["answer"]) + 1)


def _print_per_subject_leaderboard(per_subj, drift, all_subjects, models):
    print("\n" + "=" * 96)
    print("MANIPULATION RESISTANCE BY SUBJECT  (adjusted = raw − natural drift)")
    print("=" * 96)
    for subj in sorted(all_subjects):
        print(f"\n  {subj}")
        print(f"  {'Model':<40} {'Raw%':>7} {'Drift%':>8} {'Adjusted%':>11} {'StdVar%':>9} {'n':>5}")
        print(f"  {'-' * 88}")
        for model in models:
            stats = per_subj[model][subj]
            d = drift[model][subj]
            adjusted = stats["raw_resistance"] - d
            print(
                f"  {model:<40} "
                f"{stats['raw_resistance']*100:>7.1f} "
                f"{d*100:>8.1f} "
                f"{adjusted*100:>11.1f} "
                f"{stats['std_by_variant']*100:>9.1f} "
                f"{stats['n_valid']:>5}"
            )


def _print_overall_leaderboard(per_subj, drift, all_subjects, models):
    print("\n" + "=" * 96)
    print("OVERALL  (mean across subjects of subject-adjusted resistance)")
    print("=" * 96)
    print(f"  {'Model':<40} {'Adjusted%':>11} {'Raw%':>7} {'AvgDrift%':>10} {'StdVar%':>9}")
    print(f"  {'-' * 80}")
    rows = []
    for model in models:
        per_subj_adjusted = []
        per_subj_raw = []
        per_subj_drift_vals = []
        per_subj_std = []
        for subj in all_subjects:
            stats = per_subj[model][subj]
            d = drift[model][subj]
            per_subj_adjusted.append(stats["raw_resistance"] - d)
            per_subj_raw.append(stats["raw_resistance"])
            per_subj_drift_vals.append(d)
            per_subj_std.append(stats["std_by_variant"])
        rows.append((
            model,
            sum(per_subj_adjusted) / len(per_subj_adjusted),
            sum(per_subj_raw) / len(per_subj_raw),
            sum(per_subj_drift_vals) / len(per_subj_drift_vals),
            sum(per_subj_std) / len(per_subj_std),
        ))
    rows.sort(key=lambda r: -r[1])
    for model, adj, raw, dr, std in rows:
        print(f"  {model:<40} {adj*100:>11.1f} {raw*100:>7.1f} {dr*100:>10.1f} {std*100:>9.1f}")


# ── Drift-sequence (6-turn) summary ────────────────────────────────


def _drift_sequence_summary(results_by_model, models):
    """Final-turn resistance after the 6-turn drift sequence, per model."""
    print("\n" + "=" * 96)
    print("6-TURN INCREMENTAL DRIFT SEQUENCE")
    print("=" * 96)
    print(f"  {'Model':<40} {'Final resistance%':>20} {'n':>5}")
    print(f"  {'-' * 80}")
    for model in models:
        drift_results = [
            r for r in results_by_model[model]
            if r["attack"] == DRIFT_ATTACK and r["outcome"] != "invalid"
        ]
        resistant = sum(1 for r in drift_results if r["outcome"] == "resistant")
        rate = 100.0 * resistant / len(drift_results)
        print(f"  {model:<40} {rate:>19.1f}% {len(drift_results):>5}")


# ── Main loop ──────────────────────────────────────────────────────


def run(models, n_questions, results_dir="results", include_drift=True, concurrency=1):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    print(f"\n=== loading {n_questions} hard-MMLU questions across {len(HARD_MMLU)} subjects ===")
    df = load_mmlu_data_sample(
        subjects=HARD_MMLU,
        max_per_subject=max(1, n_questions // len(HARD_MMLU)),
        random_state=RANDOM_STATE,
    )
    print(f"   total questions: {len(df)}")

    results_by_model = {}
    all_results = []

    def process_sample(model, row):
        """One sample → list of attack result dicts (all attacks × variants for that sample)."""
        qtext = question_text(row)
        correct = correct_number(row)
        sample_id = int(row.name)
        subject = row.get("subject", "unknown")
        try:
            attack_results = mp.run_all_attacks(
                model, qtext, correct, include_drift=include_drift,
            )
            for r in attack_results:
                r["sample_id"] = sample_id
                r["subject"] = subject
            return attack_results
        except Exception:
            traceback.print_exc()
            return []

    rows = [row for _, row in df.iterrows()]
    for model in models:
        print(f"\n--- {model}  (concurrency={concurrency}) ---")
        model_results = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(process_sample, model, row) for row in rows]
            for f in tqdm(as_completed(futures), total=len(futures), desc=model):
                model_results.extend(f.result())
        results_by_model[model] = model_results
        all_results.extend(model_results)

    # ── Per-subject analysis with drift correction (utils/stats reusable) ──
    per_subj = per_subject_resistance_stats(
        results_by_model, exclude_attacks=(CONTROL_ATTACK, DRIFT_ATTACK),
    )
    drift = per_subject_drift_rate(results_by_model, control_attack=CONTROL_ATTACK)
    all_subjects = sorted({
        subj for subj_dict in per_subj.values() for subj in subj_dict
    })
    _print_per_subject_leaderboard(per_subj, drift, all_subjects, models)
    _print_overall_leaderboard(per_subj, drift, all_subjects, models)
    print_pairwise_mcnemar(results_by_model, models)
    if include_drift:
        _drift_sequence_summary(results_by_model, models)

    # ── Write detailed results ──
    out_path = os.path.join(results_dir, f"manipulation_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "n_questions": len(df),
                "models": models,
                "per_subject_resistance": per_subj,
                "natural_drift": drift,
                "raw_results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")
    return per_subj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n", type=int, default=None,
                   help="Total questions sampled across hard-MMLU subjects")
    p.add_argument("--no-drift", action="store_true", help="Skip the 6-turn drift sequence")
    p.add_argument("--concurrency", type=int, default=None,
                   help="Threads parallelizing samples per model (default 10)")
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()

    cfg = load_config()
    manip = cfg["manipulation"]

    models = args.models or models_from_config(cfg)
    n = args.n if args.n is not None else manip["n"]
    concurrency = args.concurrency if args.concurrency is not None else manip["concurrency"]
    results_dir = args.results_dir or cfg["results_dir"]
    include_drift = (not args.no_drift) and manip["include_drift"]

    run(models, n, results_dir, include_drift=include_drift, concurrency=concurrency)


if __name__ == "__main__":
    main()
