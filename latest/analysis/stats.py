"""analysis/stats.py — confidence intervals + significance tests.

- wilson_ci    : Wilson score interval for a proportion (good at small n, which is
                 exactly the regime here — far better than the normal approximation).
- mcnemar      : McNemar's test on discordant counts, via statsmodels
                 (statsmodels.stats.contingency_tables.mcnemar) — this is the
                 CORRECT import; the old code imported a non-existent
                 scipy.stats.mcnemar, so its significance test silently never ran.
- pairwise_mcnemar : all model pairs, matched on variant-AVERAGED resistance per
                 (item, attack) so the 4 phrasing variants are not treated as
                 independent observations (the pseudo-replication fix).
"""
from __future__ import annotations

import math
from itertools import combinations


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(point, lower, upper) Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (round(p, 4), round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def mcnemar(b: int, c: int) -> dict | None:
    """McNemar on discordant counts b (A resisted, B didn't) and c (B did, A didn't).

    Uses the exact binomial test when discordant total is small (<25), else the
    chi-square with continuity correction. Returns None if statsmodels is absent.
    """
    try:
        from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
    except ImportError:
        return None
    table = [[0, b], [c, 0]]
    res = _mcnemar(table, exact=(b + c < 25), correction=True)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue), "b": b, "c": c, "n_discordant": b + c}


def significance_stars(p: float | None) -> str:
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def pairwise_mcnemar(model_resist: dict[str, dict]) -> list[dict]:
    """model_resist: {model: {match_key: resistant_bool}}. Returns per-pair results.

    Matched on the keys both models share. `match_key` should already be
    variant-AVERAGED (e.g. (item_id, attack)) so variants aren't double-counted.
    """
    out = []
    for a, bm in combinations(sorted(model_resist), 2):
        ra, rb = model_resist[a], model_resist[bm]
        common = set(ra) & set(rb)
        if not common:
            continue
        b = sum(1 for k in common if ra[k] and not rb[k])
        c = sum(1 for k in common if rb[k] and not ra[k])
        res = mcnemar(b, c)
        out.append({
            "model_a": a, "model_b": bm, "n_pairs": len(common),
            "a_better": b, "b_better": c,
            "statistic": (res or {}).get("statistic"),
            "pvalue": (res or {}).get("pvalue"),
            "stars": significance_stars((res or {}).get("pvalue")),
        })
    return out
