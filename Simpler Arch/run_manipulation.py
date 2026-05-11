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
import statistics
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

import manipulation as mp
from load_config import load_config, models_from_config
from utils.load_dataset import load_mmlu_data_sample

# scipy is optional — McNemar's test degrades gracefully if not installed
try:
    from scipy.stats import chi2 as _scipy_chi2
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

load_dotenv()

_DS_CFG = load_config().get("dataset", {})
HARD_MMLU = _DS_CFG.get("hard_mmlu", [])
RANDOM_STATE = _DS_CFG.get("random_state", 42)

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


# ── McNemar's pairwise test ────────────────────────────────────────


def _mcnemar(results_a, results_b):
    """McNemar's chi-square (Yates' continuity correction) on matched pairs.

    Pairs are keyed by (sample_id, attack, variant_idx). Only counts pairs
    where BOTH models produced a valid (non-'invalid') outcome.
    Returns dict {chi2, p_value, n_pairs, n_a_better, n_b_better} or None if
    no common pairs exist.
    """
    def lookup(results):
        return {
            (r["sample_id"], r["attack"], r.get("variant_idx", 0)): (r["outcome"] == "resistant")
            for r in results
            if r.get("outcome") and r["outcome"] != "invalid"
        }

    la, lb = lookup(results_a), lookup(results_b)
    common = set(la) & set(lb)
    if not common:
        return None

    n10 = sum(1 for k in common if la[k] and not lb[k])   # A resistant, B folded
    n01 = sum(1 for k in common if not la[k] and lb[k])   # A folded, B resistant
    discordant = n10 + n01
    if discordant == 0:
        # No disagreement — perfectly tied
        return {"chi2": 0.0, "p_value": 1.0, "n_pairs": len(common),
                "n_a_better": n10, "n_b_better": n01}

    chi2_stat = (abs(n10 - n01) - 1) ** 2 / discordant  # Yates correction
    p_value = float(_scipy_chi2.sf(chi2_stat, 1)) if _HAS_SCIPY else None
    return {
        "chi2": round(chi2_stat, 4),
        "p_value": round(p_value, 4) if p_value is not None else None,
        "n_pairs": len(common),
        "n_a_better": n10,
        "n_b_better": n01,
    }


def _stars(p):
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── Per-subject aggregation with drift correction ─────────────────


def _per_subject_drift(results_by_model):
    """For each (model, subject), compute fold rate on neutral_control attacks.

    Returns {model: {subject: drift_rate}}. drift_rate = fraction of valid
    control outcomes that were NOT 'resistant'.
    """
    drift = {}
    for model, results in results_by_model.items():
        per_subj = defaultdict(list)
        for r in results:
            if r.get("attack") != CONTROL_ATTACK or r.get("outcome") in (None, "invalid"):
                continue
            per_subj[r["subject"]].append(r["outcome"])
        drift[model] = {
            subj: sum(1 for o in outs if o != "resistant") / len(outs)
            for subj, outs in per_subj.items()
        }
    return drift


def _per_subject_resistance(results_by_model):
    """For each (model, subject), compute raw resistance across non-control attacks.

    Returns dict {model: {subject: {raw_resistance, std_by_variant, n_valid, n_total}}}.
    std_by_variant is the std of per-(attack,variant) resistance rates in the subject —
    a phrasing-sensitivity metric.
    """
    out = {}
    for model, results in results_by_model.items():
        cells = defaultdict(list)  # (subject, attack, variant_idx) -> [scores]
        per_subj_scores = defaultdict(list)
        for r in results:
            if r.get("attack") in (CONTROL_ATTACK, DRIFT_ATTACK):
                continue
            if r.get("score") is None:
                continue
            cells[(r["subject"], r["attack"], r["variant_idx"])].append(r["score"])
            per_subj_scores[r["subject"]].append(r["score"])

        # Per (subject, attack, variant): mean resistance rate
        cell_means = {k: sum(v) / len(v) for k, v in cells.items()}

        model_out = {}
        all_subjects = {k[0] for k in cell_means}
        for subj in all_subjects:
            subj_cells = [v for k, v in cell_means.items() if k[0] == subj]
            scores = per_subj_scores[subj]
            model_out[subj] = {
                "raw_resistance": sum(scores) / len(scores) if scores else 0.0,
                "std_by_variant": statistics.stdev(subj_cells) if len(subj_cells) > 1 else 0.0,
                "n_valid": len(scores),
                "n_cells": len(subj_cells),
            }
        out[model] = model_out
    return out


def _print_per_subject_leaderboard(per_subj, drift, all_subjects, models):
    print("\n" + "=" * 96)
    print("MANIPULATION RESISTANCE BY SUBJECT  (adjusted = raw − natural drift)")
    print("=" * 96)
    for subj in sorted(all_subjects):
        print(f"\n  {subj}")
        print(f"  {'Model':<40} {'Raw%':>7} {'Drift%':>8} {'Adjusted%':>11} {'StdVar%':>9} {'n':>5}")
        print(f"  {'-' * 88}")
        for model in models:
            stats = per_subj.get(model, {}).get(subj)
            if not stats:
                print(f"  {model:<40} {'—':>7} {'—':>8} {'—':>11} {'—':>9} {'—':>5}")
                continue
            d = drift.get(model, {}).get(subj, 0.0)
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
            stats = per_subj.get(model, {}).get(subj)
            if not stats:
                continue
            d = drift.get(model, {}).get(subj, 0.0)
            per_subj_adjusted.append(stats["raw_resistance"] - d)
            per_subj_raw.append(stats["raw_resistance"])
            per_subj_drift_vals.append(d)
            per_subj_std.append(stats["std_by_variant"])
        if not per_subj_adjusted:
            rows.append((model, 0.0, 0.0, 0.0, 0.0))
            continue
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


def _print_pairwise_mcnemar(results_by_model, models):
    if not _HAS_SCIPY:
        print("\n[McNemar's pairwise: scipy not installed — skipped]")
        return
    print("\n" + "=" * 96)
    print("PAIRWISE McNEMAR'S TEST  (matched on sample_id × attack × variant_idx)")
    print("=" * 96)
    print(f"  {'A vs B':<60} {'χ²':>8} {'p':>8} {'sig':>5} {'A>B':>5} {'B>A':>5} {'n':>5}")
    print(f"  {'-' * 96}")
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            res = _mcnemar(results_by_model.get(a, []), results_by_model.get(b, []))
            if res is None:
                print(f"  {(a + ' vs ' + b):<60} {'no common pairs':>30}")
                continue
            print(
                f"  {(a + ' vs ' + b):<60} "
                f"{res['chi2']:>8.3f} "
                f"{(res['p_value'] if res['p_value'] is not None else 0):>8.4f} "
                f"{_stars(res['p_value']):>5} "
                f"{res['n_a_better']:>5} "
                f"{res['n_b_better']:>5} "
                f"{res['n_pairs']:>5}"
            )


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
            r for r in results_by_model.get(model, [])
            if r.get("attack") == DRIFT_ATTACK and r.get("outcome") not in (None, "invalid")
        ]
        if not drift_results:
            print(f"  {model:<40} {'—':>20} {'—':>5}")
            continue
        resistant = sum(1 for r in drift_results if r["outcome"] == "resistant")
        rate = 100.0 * resistant / len(drift_results)
        print(f"  {model:<40} {rate:>19.1f}% {len(drift_results):>5}")


# ── Main loop ──────────────────────────────────────────────────────


def run(models, n_questions, results_dir="results", include_drift=True):
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

    for model in models:
        print(f"\n--- {model} ---")
        model_results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=model):
            qtext = question_text(row)
            correct = correct_number(row)
            sample_id = int(row.name) if hasattr(row, "name") else 0
            subject = row.get("subject", "unknown")
            try:
                attack_results = mp.run_all_attacks(
                    model, qtext, correct, include_drift=include_drift,
                )
                for r in attack_results:
                    r["sample_id"] = sample_id
                    r["subject"] = subject
                model_results.extend(attack_results)
            except Exception:
                traceback.print_exc()
        results_by_model[model] = model_results
        all_results.extend(model_results)

    # ── Per-subject analysis with drift correction ──
    per_subj = _per_subject_resistance(results_by_model)
    drift = _per_subject_drift(results_by_model)
    all_subjects = sorted({
        subj for subj_dict in per_subj.values() for subj in subj_dict
    })
    _print_per_subject_leaderboard(per_subj, drift, all_subjects, models)
    _print_overall_leaderboard(per_subj, drift, all_subjects, models)
    _print_pairwise_mcnemar(results_by_model, models)
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
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()

    cfg = load_config()
    manip = cfg.get("manipulation", {})

    models = args.models or models_from_config(cfg) or [
        "openai:gpt-5.4-mini", "anthropic:claude-sonnet-4-6", "groq:openai/gpt-oss-120b"
    ]
    n = args.n if args.n is not None else manip.get("n", 8)
    results_dir = args.results_dir or cfg.get("results_dir", "results")
    include_drift = (not args.no_drift) and manip.get("include_drift", True)

    run(models, n, results_dir, include_drift=include_drift)


if __name__ == "__main__":
    main()
