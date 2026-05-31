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
    rank_attacks,
    stateful_stats,
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
            stats = per_subj.get(model, {}).get(subj)
            if stats is None:          # subject dropped out for this model (small n / all invalid)
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
            if stats is None:
                continue
            d = drift.get(model, {}).get(subj, 0.0)
            per_subj_adjusted.append(stats["raw_resistance"] - d)
            per_subj_raw.append(stats["raw_resistance"])
            per_subj_drift_vals.append(d)
            per_subj_std.append(stats["std_by_variant"])
        if not per_subj_adjusted:      # no scored subjects for this model
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


def _bar(frac, width=10):
    filled = int(max(0.0, min(1.0, frac)) * width)
    return "█" * filled + "░" * (width - filled)


# ── Stateless: attack-effectiveness ranking ────────────────────────


def _print_attack_ranking(results_by_model, models, all_subjects, exclude):
    print("\n" + "=" * 96)
    print("STATELESS — ATTACK TYPE EFFECTIVENESS RANKING")
    print("  Ranked by fold rate = how often the attack made the model surrender.")
    print("  Higher fold rate → more effective attack.")
    print("=" * 96)

    ranked_all = rank_attacks(results_by_model, exclude_attacks=exclude)
    for model in models:
        print(f"\n  {model} — overall (all subjects)")
        for i, (attack, fold, resist, n) in enumerate(ranked_all.get(model, []), 1):
            print(
                f"    #{i}  {attack:<22}  {_bar(fold)}  "
                f"fold={fold:>5.0%}  resist={resist:>5.0%}  n={n}"
            )

    print("\n" + "=" * 96)
    print("STATELESS — TOP-3 MOST EFFECTIVE ATTACKS BY SUBJECT")
    print("=" * 96)
    for subj in sorted(all_subjects):
        print(f"\n  {subj}")
        for model in models:
            ranked = rank_attacks(
                results_by_model, exclude_attacks=exclude, subject=subj
            ).get(model, [])
            if not ranked:
                continue
            top = "  ".join(
                f"#{i + 1} {at[:13]}({fr:.0%})"
                for i, (at, fr, _, _) in enumerate(ranked[:3])
            )
            print(f"    {model[:34]:<34}: {top}")


# ── Stateful reporting ─────────────────────────────────────────────


def _print_stateful(stateful_by_model, models, all_subjects):
    print("\n" + "=" * 96)
    print("STATEFUL — RESISTANCE BY SUBJECT")
    print("  Full resistance = model withstood ALL chained attacks in one session.")
    print("=" * 96)
    for subj in sorted(all_subjects):
        print(f"\n  {subj}")
        print(f"  {'Model':<40} {'FullResist%':>12} {'AvgTurnsToFold':>16} {'KillingBlow'}")
        print(f"  {'-' * 90}")
        per_subj = stateful_stats(stateful_by_model, subject=subj)
        for model in models:
            s = per_subj.get(model)
            if not s:
                print(f"  {model:<40} {'no data':>12}")
                continue
            avg = f"{s['avg_attacks_to_fold']:.1f}" if s["avg_attacks_to_fold"] is not None else "—"
            kb = s["top_killing_blow"] or "—"
            print(
                f"  {model:<40} "
                f"{s['full_resistance_rate'] * 100:>11.1f} "
                f"{avg:>16} "
                f" {kb}"
            )

    print("\n" + "=" * 96)
    print("STATEFUL — OVERALL  (across all subjects)")
    print("=" * 96)
    print(f"  {'Model':<40} {'FullResist%':>12} {'AvgTurnsToFold':>16} {'TopKillingBlow'}")
    print(f"  {'-' * 90}")
    overall = stateful_stats(stateful_by_model)
    for model in models:
        s = overall.get(model)
        if not s:
            continue
        avg = f"{s['avg_attacks_to_fold']:.1f}" if s["avg_attacks_to_fold"] is not None else "—"
        kb = s["top_killing_blow"] or "—"
        print(
            f"  {model:<40} "
            f"{s['full_resistance_rate'] * 100:>11.1f} "
            f"{avg:>16} "
            f" {kb}"
        )

    print("\n" + "=" * 96)
    print("STATEFUL — KILLING BLOW RANKING  (which attack finally broke resistance?)")
    print("=" * 96)
    for model in models:
        s = overall.get(model)
        kill = s["killing_blow_counts"] if s else {}
        total = sum(kill.values())
        if not total:
            print(f"\n  {model}: no surrenders recorded")
            continue
        print(f"\n  {model}  ({total} total surrenders)")
        for attack, count in sorted(kill.items(), key=lambda x: -x[1]):
            pct = count / total
            print(f"    {attack:<22}  {_bar(pct)}  {count}/{total} ({pct:.0%})")


# ── Cost & token accounting ────────────────────────────────────────


def _total_cost(results_by_model):
    """Sum cost_usd + tokens per model and overall.

    Result dicts always carry cost_usd/tokens from the provider layer, except
    error rows (run_all_attacks' exception handler) which omit them — hence the
    .get() defaults. Returns {per_model: {cost_usd, tokens}, total_cost_usd, total_tokens}.
    """
    per_model = {}
    grand_cost = 0.0
    grand_tokens = 0
    for model, results in results_by_model.items():
        c = sum(r.get("cost_usd", 0.0) or 0.0 for r in results)
        t = sum(r.get("tokens", 0) or 0 for r in results)
        per_model[model] = {"cost_usd": round(c, 6), "tokens": int(t)}
        grand_cost += c
        grand_tokens += int(t)
    return {
        "per_model": per_model,
        "total_cost_usd": round(grand_cost, 6),
        "total_tokens": grand_tokens,
    }


def _print_cost(label, cost):
    print("\n" + "=" * 96)
    print(f"{label} — COST & TOKENS")
    print("=" * 96)
    print(f"  {'Model':<40} {'Cost (USD)':>14} {'Tokens':>14}")
    print(f"  {'-' * 70}")
    for model, c in cost["per_model"].items():
        print(f"  {model:<40} {c['cost_usd']:>14.4f} {c['tokens']:>14,}")
    print(f"  {'-' * 70}")
    print(f"  {'TOTAL':<40} {cost['total_cost_usd']:>14.4f} {cost['total_tokens']:>14,}")


# ── Result persistence ─────────────────────────────────────────────


def _write_report(report_path, run_id, mode, models, all_subjects,
                  per_subj, drift, attack_ranking, stateful_by_model,
                  cost_summary=None):
    lines = [
        "LLM Manipulation Resistance Report",
        f"Run ID : {run_id}",
        f"Mode   : {mode}",
        f"Models : {', '.join(models)}",
        f"Subjects ({len(all_subjects)}): {', '.join(sorted(all_subjects))}",
        "",
    ]

    if per_subj is not None:
        lines += ["=" * 78, "STATELESS — SUBJECT SCORES (adjusted = raw − natural drift)", "=" * 78]
        for subj in sorted(all_subjects):
            lines.append(f"\n  {subj}")
            lines.append(f"  {'Model':<40} {'Raw%':>7} {'Drift%':>8} {'Adj%':>7} {'TopFoldAttack'}")
            lines.append(f"  {'-' * 76}")
            for model in models:
                stats = per_subj.get(model, {}).get(subj)
                if not stats:
                    continue
                d = drift.get(model, {}).get(subj, 0.0)
                ranked = attack_ranking.get(model, {}).get(subj, [])
                top = ranked[0][0] if ranked else "—"
                lines.append(
                    f"  {model:<40} "
                    f"{stats['raw_resistance'] * 100:>7.1f} "
                    f"{d * 100:>8.1f} "
                    f"{(stats['raw_resistance'] - d) * 100:>7.1f} "
                    f" {top}"
                )

    if stateful_by_model is not None:
        lines += ["", "=" * 78, "STATEFUL — SUBJECT SCORES", "=" * 78]
        for subj in sorted(all_subjects):
            lines.append(f"\n  {subj}")
            lines.append(f"  {'Model':<40} {'FullResist%':>12} {'AvgTurns':>9} {'KillingBlow'}")
            lines.append(f"  {'-' * 76}")
            per_subj_sf = stateful_stats(stateful_by_model, subject=subj)
            for model in models:
                s = per_subj_sf.get(model)
                if not s:
                    continue
                avg = f"{s['avg_attacks_to_fold']:.1f}" if s["avg_attacks_to_fold"] is not None else "—"
                lines.append(
                    f"  {model:<40} "
                    f"{s['full_resistance_rate'] * 100:>12.1f} "
                    f"{avg:>9} "
                    f" {s['top_killing_blow'] or '—'}"
                )

    if cost_summary:
        lines += ["", "=" * 78, "COST & TOKEN USAGE", "=" * 78]
        for label, cost in cost_summary.items():
            if not cost:
                continue
            lines.append(f"\n  {label}")
            lines.append(f"  {'Model':<40} {'Cost (USD)':>14} {'Tokens':>14}")
            lines.append(f"  {'-' * 70}")
            for model, c in cost["per_model"].items():
                lines.append(f"  {model:<40} {c['cost_usd']:>14.4f} {c['tokens']:>14,}")
            lines.append(f"  {'-' * 70}")
            lines.append(f"  {'TOTAL':<40} {cost['total_cost_usd']:>14.4f} {cost['total_tokens']:>14,}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main loop ──────────────────────────────────────────────────────


def _load_questions(n_questions):
    print(f"\n=== loading {n_questions} hard-MMLU questions across {len(HARD_MMLU)} subjects ===")
    df = load_mmlu_data_sample(
        subjects=HARD_MMLU,
        max_per_subject=max(1, n_questions // len(HARD_MMLU)),
        random_state=RANDOM_STATE,
    )
    print(f"   total questions: {len(df)}")
    return df


def run(models, n_questions, results_dir="results", include_drift=True,
        concurrency=1, mode="both"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    df = _load_questions(n_questions)
    rows = [row for _, row in df.iterrows()]

    # Crash-safe partial log: each (mode, model) is appended the moment it finishes,
    # so a hang / kill / crash loses at most the model currently in flight — never
    # the whole stage. The final structured JSON + report are still written at the end.
    partial_path = os.path.join(results_dir, f"manipulation_{run_id}_partial.jsonl")

    def _run_per_model(process_sample, desc):
        """Fan one per-sample function across all models. Flushes each model's
        results to the partial JSONL immediately on completion (incremental save)."""
        by_model = {}
        for model in models:
            print(f"\n--- {desc}: {model}  (concurrency={concurrency}) ---")
            model_results = []
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(process_sample, model, row) for row in rows]
                for f in tqdm(as_completed(futures), total=len(futures), desc=model):
                    model_results.extend(f.result())
            by_model[model] = model_results
            # ── incremental save: this model's results hit disk right now ──
            with open(partial_path, "a", encoding="utf-8") as pf:
                pf.write(json.dumps(
                    {"mode": desc, "model": model, "results": model_results},
                    default=str,
                ) + "\n")
                pf.flush()
            print(f"  ✔ saved {desc}/{model} → {os.path.basename(partial_path)} "
                  f"({len(model_results)} results)")
        return by_model

    # Shared per-sample metadata extraction.
    def _meta(row):
        return question_text(row), correct_number(row), int(row.name), row.get("subject", "unknown")

    per_subj = drift = None
    attack_ranking = None
    stateless_by_model = None
    stateful_by_model = None
    all_subjects = set()
    cost_summary = {}
    payload = {
        "run_id": run_id,
        "mode": mode,
        "n_questions": len(df),
        "models": models,
    }

    # ── STATELESS MODE ────────────────────────────────────────────
    if mode in ("stateless", "both"):
        print("\n" + "=" * 96)
        print("STATELESS MODE — fresh conversation per attack type")
        print("=" * 96)

        def process_stateless(model, row):
            qtext, correct, sample_id, subject = _meta(row)
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

        stateless_by_model = _run_per_model(process_stateless, "stateless")

        per_subj = per_subject_resistance_stats(
            stateless_by_model, exclude_attacks=(CONTROL_ATTACK, DRIFT_ATTACK),
        )
        drift = per_subject_drift_rate(stateless_by_model, control_attack=CONTROL_ATTACK)
        all_subjects |= {subj for d in per_subj.values() for subj in d}

        # ── Build data/payload FIRST, so display failures can't lose it ──
        attack_ranking = {
            model: {
                subj: rank_attacks(
                    stateless_by_model, exclude_attacks=(CONTROL_ATTACK, DRIFT_ATTACK),
                    subject=subj,
                ).get(model, [])
                for subj in all_subjects
            }
            for model in models
        }
        stateless_cost = _total_cost(stateless_by_model)
        cost_summary["STATELESS"] = stateless_cost
        payload["stateless"] = {
            "per_subject_resistance": per_subj,
            "natural_drift": drift,
            "attack_fold_rates": {
                model: {at: round(fr, 4) for at, fr, _, _ in ranked}
                for model, ranked in rank_attacks(
                    stateless_by_model, exclude_attacks=(CONTROL_ATTACK, DRIFT_ATTACK)
                ).items()
            },
            "cost": stateless_cost,
            "raw_results": [r for rs in stateless_by_model.values() for r in rs],
        }
        # ── Display is best-effort: never let a print abort the run ──
        try:
            _print_per_subject_leaderboard(per_subj, drift, sorted(all_subjects), models)
            _print_overall_leaderboard(per_subj, drift, sorted(all_subjects), models)
            print_pairwise_mcnemar(stateless_by_model, models)
            _print_attack_ranking(
                stateless_by_model, models, all_subjects,
                exclude=(CONTROL_ATTACK, DRIFT_ATTACK),
            )
            if include_drift:
                _drift_sequence_summary(stateless_by_model, models)
            _print_cost("STATELESS", stateless_cost)
        except Exception:
            print("[stateless display failed — data already captured for save]")
            traceback.print_exc()

    # ── STATEFUL MODE ─────────────────────────────────────────────
    if mode in ("stateful", "both"):
        print("\n" + "=" * 96)
        print("STATEFUL MODE — chained attacks in one conversation per sample")
        print("  Attacks escalate until the model surrenders or all are exhausted.")
        print("=" * 96)

        def process_stateful(model, row):
            qtext, correct, sample_id, subject = _meta(row)
            try:
                r = mp.run_stateful_attack(model, qtext, correct)
                r["sample_id"] = sample_id
                r["subject"] = subject
                return [r]
            except Exception:
                traceback.print_exc()
                return []

        stateful_by_model = _run_per_model(process_stateful, "stateful")
        all_subjects |= {
            r.get("subject", "unknown")
            for rs in stateful_by_model.values() for r in rs
        }
        stateful_cost = _total_cost(stateful_by_model)
        cost_summary["STATEFUL"] = stateful_cost
        payload["stateful"] = {
            "summary": stateful_stats(stateful_by_model),
            "cost": stateful_cost,
            "raw_results": [r for rs in stateful_by_model.values() for r in rs],
        }
        try:
            _print_stateful(stateful_by_model, models, all_subjects)
            _print_cost("STATEFUL", stateful_cost)
        except Exception:
            print("[stateful display failed — data already captured for save]")
            traceback.print_exc()

    # ── Grand total across modes ──────────────────────────────────
    grand_cost = round(sum(c["total_cost_usd"] for c in cost_summary.values()), 6)
    grand_tokens = sum(c["total_tokens"] for c in cost_summary.values())
    payload["cost_total_usd"] = grand_cost
    payload["tokens_total"] = grand_tokens
    print("\n" + "=" * 96)
    print(f"RUN TOTAL — cost=${grand_cost:.4f}   tokens={grand_tokens:,}")
    print("=" * 96)

    # ── Persist: JSON (raw) + text report (human-readable) ─────────
    payload["subjects"] = sorted(all_subjects)
    json_path = os.path.join(results_dir, f"manipulation_{run_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {json_path}")

    report_path = os.path.join(results_dir, f"manipulation_{run_id}_report.txt")
    _write_report(
        report_path, run_id, mode, models, all_subjects,
        per_subj, drift, attack_ranking, stateful_by_model,
        cost_summary=cost_summary,
    )
    print(f"Wrote {report_path}")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--n", type=int, default=None,
                   help="Total questions sampled across hard-MMLU subjects")
    p.add_argument("--no-drift", action="store_true", help="Skip the 6-turn drift sequence")
    p.add_argument("--concurrency", type=int, default=None,
                   help="Threads parallelizing samples per model (default 10)")
    p.add_argument("--mode", choices=["stateless", "stateful", "both"], default=None,
                   help="stateless: fresh session per attack + ranking. "
                        "stateful: chain attacks until the model folds. "
                        "both: run both (default).")
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()

    cfg = load_config()
    manip = cfg["manipulation"]

    models = args.models or models_from_config(cfg)
    n = args.n if args.n is not None else manip["n"]
    concurrency = args.concurrency if args.concurrency is not None else manip["concurrency"]
    results_dir = args.results_dir or cfg["results_dir"]
    include_drift = (not args.no_drift) and manip["include_drift"]
    mode = args.mode or manip.get("mode", "both")

    run(models, n, results_dir, include_drift=include_drift,
        concurrency=concurrency, mode=mode)


if __name__ == "__main__":
    main()
