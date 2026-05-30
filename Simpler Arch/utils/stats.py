"""Statistical helpers for cross-model comparison.

Reusable across runners — anywhere you have `results_by_model = {model: [result_dict, ...]}`
with each result carrying `sample_id`, `subject`, `outcome`, `score`, `attack`,
`variant_idx`. Works for manipulation today; benchmark + moral pipelines can
adopt the same shape and reuse these.

Functions:
  - mcnemar_test               : pairwise binary-outcome test on matched pairs
  - significance_stars         : p-value -> '*' / '**' / '***'
  - per_subject_drift_rate     : baseline fold rate on a 'control' attack
  - per_subject_resistance_stats: raw resistance + phrasing-sensitivity std per subject
  - print_pairwise_mcnemar     : leaderboard-format table of all pair comparisons
"""
import statistics
from collections import defaultdict

try:
    from scipy.stats import chi2 as _scipy_chi2
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _default_match_key(r):
    return (r["sample_id"], r["attack"], r["variant_idx"])


def mcnemar_test(results_a, results_b, key_fn=None):
    """McNemar's chi-square (Yates' continuity correction) on matched pairs.

    key_fn(result) -> hashable: how to match results between two model lists.
    Default: (sample_id, attack, variant_idx).

    Counts only pairs where BOTH models produced a non-'invalid' outcome.
    Outcome 'resistant' counts as the success class; anything else is failure.

    Returns dict {chi2, p_value, n_pairs, n_a_better, n_b_better} or None if
    no common pairs exist.
    """
    if key_fn is None:
        key_fn = _default_match_key

    def lookup(results):
        return {
            key_fn(r): (r["outcome"] == "resistant")
            for r in results
            if r["outcome"] != "invalid"
        }

    la, lb = lookup(results_a), lookup(results_b)
    common = set(la) & set(lb)
    if not common:
        return None

    n10 = sum(1 for k in common if la[k] and not lb[k])   # A resistant, B not
    n01 = sum(1 for k in common if not la[k] and lb[k])   # A not, B resistant
    discordant = n10 + n01
    if discordant == 0:
        return {
            "chi2": 0.0, "p_value": 1.0, "n_pairs": len(common),
            "n_a_better": n10, "n_b_better": n01,
        }

    chi2_stat = (abs(n10 - n01) - 1) ** 2 / discordant  # Yates correction
    p = float(_scipy_chi2.sf(chi2_stat, 1)) if _HAS_SCIPY else None
    return {
        "chi2": round(chi2_stat, 4),
        "p_value": round(p, 4) if p is not None else None,
        "n_pairs": len(common),
        "n_a_better": n10,
        "n_b_better": n01,
    }


def significance_stars(p_value):
    """p < 0.001 -> '***', < 0.01 -> '**', < 0.05 -> '*', else ''."""
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def per_subject_drift_rate(results_by_model, control_attack="neutral_control"):
    """For each (model, subject), compute fold rate on a control / no-pressure attack.

    drift_rate = fraction of valid control outcomes that were NOT 'resistant'.
    Subtract this from raw resistance to get the natural-drift-corrected score.

    Returns {model: {subject: drift_rate}}.
    """
    drift = {}
    for model, results in results_by_model.items():
        per_subj = defaultdict(list)
        for r in results:
            if r["attack"] != control_attack or r["outcome"] == "invalid":
                continue
            per_subj[r["subject"]].append(r["outcome"])
        drift[model] = {
            subj: sum(1 for o in outs if o != "resistant") / len(outs)
            for subj, outs in per_subj.items()
        }
    return drift


def per_subject_resistance_stats(results_by_model, exclude_attacks=()):
    """For each (model, subject), compute raw resistance + phrasing-sensitivity std.

    exclude_attacks: attack names to ignore (e.g., the control + drift sequence).
    std_by_variant is the stdev of per-(attack, variant_idx) resistance rates
    within the subject — a "how sensitive is this model to phrasing?" metric.

    Returns {model: {subject: {raw_resistance, std_by_variant, n_valid, n_cells}}}.
    """
    exclude = set(exclude_attacks)
    out = {}
    for model, results in results_by_model.items():
        cells = defaultdict(list)
        per_subj_scores = defaultdict(list)
        for r in results:
            if r["attack"] in exclude or r["score"] is None:
                continue
            cells[(r["subject"], r["attack"], r["variant_idx"])].append(r["score"])
            per_subj_scores[r["subject"]].append(r["score"])

        cell_means = {k: sum(v) / len(v) for k, v in cells.items()}
        model_out = {}
        for subj in {k[0] for k in cell_means}:
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


def rank_attacks(results_by_model, exclude_attacks=(), subject=None):
    """Rank attack types by how often they made each model fold.

    fold_rate = fraction of valid outcomes that were 'folded' (switched to a
    specific wrong answer). Higher fold_rate → more effective attack.

    exclude_attacks: names to skip (e.g., the control + drift sequence).
    subject: restrict to a single subject if given (None = all subjects).

    Returns {model: [(attack, fold_rate, resist_rate, n_valid), ...]} sorted by
    fold_rate descending.
    """
    exclude = set(exclude_attacks)
    out = {}
    for model, results in results_by_model.items():
        buckets = defaultdict(list)
        for r in results:
            if r["attack"] in exclude or r["outcome"] == "invalid":
                continue
            if subject is not None and r.get("subject") != subject:
                continue
            buckets[r["attack"]].append(r["outcome"])
        ranked = []
        for attack, outs in buckets.items():
            n = len(outs)
            fold = sum(1 for o in outs if o == "folded") / n
            resist = sum(1 for o in outs if o == "resistant") / n
            ranked.append((attack, fold, resist, n))
        out[model] = sorted(ranked, key=lambda x: x[1], reverse=True)
    return out


def stateful_stats(stateful_by_model, subject=None):
    """Aggregate stateful chained-attack results per model.

    Each stateful result carries `full_resistance` (held through all attacks),
    `fold_attack` (the killing blow, if any), and `attacks_tried` (turns survived).

    subject: restrict to one subject if given (None = all subjects).

    Returns {model: {n_valid, full_resistance_count, full_resistance_rate,
    avg_attacks_to_fold, killing_blow_counts, top_killing_blow}} — or None for a
    model with no valid samples in scope.
    """
    out = {}
    for model, results in stateful_by_model.items():
        res = [
            r for r in results
            if r["outcome"] != "invalid"
            and (subject is None or r.get("subject") == subject)
        ]
        if not res:
            out[model] = None
            continue
        full = [r for r in res if r.get("full_resistance")]
        folded = [r for r in res if not r.get("full_resistance")]
        kill = defaultdict(int)
        for r in folded:
            if r.get("fold_attack"):
                kill[r["fold_attack"]] += 1
        out[model] = {
            "n_valid": len(res),
            "full_resistance_count": len(full),
            "full_resistance_rate": len(full) / len(res),
            "avg_attacks_to_fold": (
                sum(len(r.get("attacks_tried", [])) for r in folded) / len(folded)
                if folded else None
            ),
            "killing_blow_counts": dict(kill),
            "top_killing_blow": max(kill, key=kill.get) if kill else None,
        }
    return out


def print_pairwise_mcnemar(results_by_model, models, key_fn=None, title=None):
    """Print all pairwise McNemar comparisons as a table. Returns the list of (a, b, result)."""
    if not _HAS_SCIPY:
        print("\n[McNemar's pairwise: scipy not installed — skipped]")
        return []
    print("\n" + "=" * 96)
    print(title or "PAIRWISE McNEMAR'S TEST  (matched on sample_id × attack × variant_idx)")
    print("=" * 96)
    print(f"  {'A vs B':<60} {'χ²':>8} {'p':>8} {'sig':>5} {'A>B':>5} {'B>A':>5} {'n':>5}")
    print(f"  {'-' * 96}")
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            res = mcnemar_test(results_by_model[a], results_by_model[b], key_fn=key_fn)
            if res is None:
                print(f"  {(a + ' vs ' + b):<60} {'no common pairs':>30}")
                continue
            rows.append((a, b, res))
            p = res["p_value"] if res["p_value"] is not None else 0
            print(
                f"  {(a + ' vs ' + b):<60} "
                f"{res['chi2']:>8.3f} "
                f"{p:>8.4f} "
                f"{significance_stars(res['p_value']):>5} "
                f"{res['n_a_better']:>5} "
                f"{res['n_b_better']:>5} "
                f"{res['n_pairs']:>5}"
            )
    return rows
