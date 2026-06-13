"""analysis/report.py — assemble analysis artifacts for a run.

Reads trials.parquet + ledger.jsonl, scores, aggregates, and writes:
  report/results.json  — all aggregate tables
  report/report.md     — human-readable Markdown (UTF-8)
  scored.parquet       — one row per ScoreRecord (re-analyzable without re-scoring)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from latest import ledger
from latest.analysis import aggregate as A
from latest.analysis.score import score_all
from latest.plan.freeze import read_trials


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def analyze(rd: Path | str, write: bool = True):
    rd = Path(rd)
    trials = read_trials(rd)
    records = ledger.read(ledger.ledger_path(rd))
    manifest = ledger.read_manifest(rd)
    scores = score_all(trials, records)

    agg = {
        "stateless_overall": A.stateless_overall(scores),
        "stateless_by_attack": A.stateless_by_attack(scores),
        "stateless_adjusted": A.stateless_adjusted(scores),
        "natural_drift": A.natural_drift(scores),
        "drift": A.drift_summary(scores),
        "stateful": A.stateful_summary(scores),
        "repeat": A.repeat_summary(scores),
        "gauntlet": A.gauntlet_summary(scores),
        "mcnemar": A.mcnemar_pairwise(scores),
        "benchmark_accuracy": A.benchmark_accuracy(scores),
        "moral_axes": A.moral_axes(scores),
        "judge_reliability": A.judge_reliability(records),
        "cost": A.cost_summary(records),
    }
    if write:
        _write(rd, manifest, scores, agg)
    return scores, agg


def _write(rd: Path, manifest, scores, agg) -> None:
    rdir = ledger.report_dir(rd)
    rdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in scores:
        d = s.model_dump()
        d["source_call_ids"] = json.dumps(d["source_call_ids"])
        d["metadata"] = json.dumps(d["metadata"], default=str)
        rows.append(d)
    if rows:
        pd.DataFrame(rows).to_parquet(ledger.scored_path(rd), index=False)

    (rdir / "results.json").write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")
    (rdir / "report.md").write_text(_markdown(manifest, agg), encoding="utf-8")


def _markdown(manifest, agg) -> str:
    # manifest.models is prefixed (provider:model); aggregates are keyed by the
    # bare model_alias (the resolved model). Look up + display by the bare alias.
    models = [m.split(":", 1)[1] if ":" in m else m for m in manifest.models]
    out: list[str] = []
    out.append(f"# latest manipulation report — run {manifest.run_id}\n")
    out.append(f"- git: `{manifest.git_sha}` (dirty={manifest.git_dirty}) · seed: {manifest.seed}")
    out.append(f"- models: {', '.join(models)}")
    out.append("")

    out.append("## Stateless — overall (pressure attacks; adjusted = raw − natural drift)\n")
    out.append("| Model | Resist | Fold | Hedge | Raw | Drift | Adjusted | n |")
    out.append("|---|---|---|---|---|---|---|---|")
    for m in models:
        o = agg["stateless_overall"].get(m)
        adj = agg["stateless_adjusted"].get(m)
        if not o:
            continue
        out.append(f"| {m} | {_pct(o['resist'])} | {_pct(o['fold'])} | {_pct(o['hedge'])} | "
                   f"{_pct(adj['raw_resistance']) if adj else '—'} | {_pct(adj['natural_drift']) if adj else '—'} | "
                   f"{_pct(adj['adjusted_resistance']) if adj else '—'} | {o['n']} |")

    out.append("\n## Stateless — per-attack fold rate (pressure attacks)\n")
    for m in models:
        ba = agg["stateless_by_attack"].get(m) or {}
        if not ba:
            continue
        out.append(f"\n**{m}**\n")
        out.append("| Attack | Resist | Fold | Hedge | Resist 95% CI | n |")
        out.append("|---|---|---|---|---|---|")
        for a, b in sorted(ba.items(), key=lambda kv: -kv[1]["fold"]):
            ci = b["resist_ci"]
            out.append(f"| {a} | {_pct(b['resist'])} | {_pct(b['fold'])} | {_pct(b['hedge'])} | "
                       f"[{ci[1]*100:.0f}–{ci[2]*100:.0f}%] | {b['n']} |")

    out.append("\n## Drift — 6-turn gradual escalation (final-turn resistance)\n")
    out.append("| Model | Resist | Fold | Hedge | n |")
    out.append("|---|---|---|---|---|")
    for m in models:
        d = agg["drift"].get(m)
        if d:
            out.append(f"| {m} | {_pct(d['resist'])} | {_pct(d['fold'])} | {_pct(d['hedge'])} | {d['n']} |")

    out.append("\n## Stateful — chained attacks (full resistance + killing blow)\n")
    out.append("| Model | Full resist | 95% CI | Avg turns survived | Top killing blow | n |")
    out.append("|---|---|---|---|---|---|")
    for m in models:
        s = agg["stateful"].get(m)
        if s:
            ci = s["full_resistance_ci"]
            out.append(f"| {m} | {_pct(s['full_resistance'])} | [{ci[1]*100:.0f}–{ci[2]*100:.0f}%] | "
                       f"{s['avg_turns_survived']} | {s['top_killing_blow'] or '—'} | {s['n']} |")

    out.append("\n## Repeat (Mode 1) — persistence under rephrasing (per attack)\n")
    for m in models:
        rp = agg["repeat"].get(m) or {}
        if not rp:
            continue
        out.append(f"\n**{m}**\n")
        out.append("| Attack | Fold | Avg variants survived | n |")
        out.append("|---|---|---|---|")
        for a, b in sorted(rp.items(), key=lambda kv: -kv[1]["fold"]):
            out.append(f"| {a} | {_pct(b['fold'])} | {b['avg_variants_survived']} | {b['n']} |")

    out.append("\n## Gauntlet (Mode 2) — endurance ceiling (NOT a potency claim)\n")
    out.append("| Model | Survived all | 95% CI | Avg turns survived | n |")
    out.append("|---|---|---|---|---|")
    for m in models:
        g = agg["gauntlet"].get(m)
        if g:
            ci = g["full_resistance_ci"]
            out.append(f"| {m} | {_pct(g['full_resistance'])} | [{ci[1]*100:.0f}–{ci[2]*100:.0f}%] | "
                       f"{g['avg_turns_survived']} | {g['n']} |")

    out.append("\n## Pairwise McNemar (variant-averaged per item×attack)\n")
    if agg["mcnemar"]:
        out.append("| A vs B | n pairs | A>B | B>A | χ²/stat | p | sig |")
        out.append("|---|---|---|---|---|---|---|")
        for c in agg["mcnemar"]:
            stat = "—" if c["statistic"] is None else f"{c['statistic']:.3f}"
            p = "—" if c["pvalue"] is None else f"{c['pvalue']:.4f}"
            out.append(f"| {c['model_a']} vs {c['model_b']} | {c['n_pairs']} | {c['a_better']} | "
                       f"{c['b_better']} | {stat} | {p} | {c['stars']} |")
    else:
        out.append("_(needs ≥2 models with overlapping items)_")

    out.append("\n## Standard benchmark — accuracy\n")
    out.append("| Model | Dataset | Accuracy / score | 95% CI | n |")
    out.append("|---|---|---|---|---|")
    for m in models:
        ba = agg["benchmark_accuracy"].get(m) or {}
        for ds in ("mmlu", "hellaswag", "truthfulqa_mc"):
            blk = ba.get(ds)
            if blk:
                ci = blk["ci"]
                out.append(f"| {m} | {ds} | {_pct(blk['accuracy'])} | [{ci[1]*100:.0f}–{ci[2]*100:.0f}%] | {blk['n']} |")
        gen = ba.get("truthfulqa_gen")
        if gen:
            out.append(f"| {m} | truthfulqa_gen | truthful {_pct(gen.get('truthful'))}, "
                       f"informative {_pct(gen.get('informative'))} | — | — |")

    out.append("\n## Moral / empathy — LLM-judge axis means (1–5)\n")
    out.append("| Model | Overall | Axes | By category |")
    out.append("|---|---|---|---|")
    for m in models:
        ma = agg["moral_axes"].get(m)
        if not ma:
            continue
        axes = ", ".join(f"{ax} {v}" for ax, v in ma["axes"].items())
        cats = ", ".join(f"{c} {v}" for c, v in ma["by_category"].items())
        out.append(f"| {m} | {ma['overall']} | {axes} | {cats} |")

    jr = agg["judge_reliability"]
    out.append("\n## Judge reliability (inter-judge agreement)\n")
    if jr:
        out.append(f"- judges: `{jr['judge_a']}` vs `{jr['judge_b']}` · n={jr['n']} matched ratings")
        out.append(f"- exact agreement: {_pct(jr['agreement'])} · Cohen's κ: {jr['kappa']}")
    else:
        out.append("_(needs ≥2 judges with overlapping ratings)_")

    out.append("\n## Cost & tokens\n")
    out.append("| Model | Calls | Cache hits | New spend $ | Total $ | Tokens |")
    out.append("|---|---|---|---|---|---|")
    for m, b in agg["cost"].items():
        out.append(f"| {m} | {b['calls']} | {b['cache_hits']} | {b['cost_new']:.4f} | "
                   f"{b['cost_total']:.4f} | {b['tokens']:,} |")

    return "\n".join(out)
