"""latest command-line entry point.

    python -m latest.cli run      [overrides]   # plan -> collect -> analyze (end to end)
    python -m latest.cli analyze  --run-id <id>  # re-analyze an existing run (no network)
    python -m latest.cli verify   --run-id <id>  # verify-ledger integrity + provenance

The `run` command exercises every enabled module (manipulation / benchmark /
moral). Overrides let you shrink it to a smoke test, e.g.:

    python -m latest.cli run --models anthropic:claude-haiku-4-5 openai:gpt-5.4-nano \\
        --subjects formal_logic college_physics --items-per-subject 1 --variants 2 \\
        --bench-n 2 --moral-per-cat 1
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

# Force UTF-8 stdout/stderr so report symbols never crash a redirected run on Windows.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

from latest import ledger as L  # noqa: E402
from latest import loaders  # noqa: E402
from latest.config.loader import load_config, models_from_config, validate  # noqa: E402
from latest.plan import design, freeze  # noqa: E402
from latest.provenance import make_run_id  # noqa: E402
from latest.env import load_env  # noqa: E402  (reuses the .env loader)

load_env()


def _build_trials(cfg, args):
    seed = (cfg.get("run") or {}).get("seed", 42)
    modules = args.modules or [m for m in ("manipulation", "benchmark", "moral")
                               if (cfg.get(m) or {}).get("enabled")]
    trials = []

    if "manipulation" in modules:
        mcfg = cfg["manipulation"]
        subjects = args.subjects or cfg["dataset"]["hard_mmlu"]
        ips = args.items_per_subject if args.items_per_subject is not None else mcfg.get("items_per_subject", 1)
        items = loaders.load_hard_mmlu_items(subjects, ips, seed)
        trials += design.build_manipulation_trials(
            items, loaders.load_attacks(), modes=args.modes or mcfg["modes"],
            include_drift=mcfg.get("include_drift", True), resamples=mcfg.get("resamples", 1),
            seed=seed, max_variants=args.variants if args.variants is not None else mcfg.get("max_variants"),
        )

    if "benchmark" in modules:
        bcfg = cfg["benchmark"]
        n = args.bench_n if args.bench_n is not None else bcfg.get("n", 10)
        items_by = {ds: loaders.load_benchmark_items(ds, n, seed) for ds in bcfg["datasets"]}
        trials += design.build_benchmark_trials(items_by)

    if "moral" in modules:
        scen = loaders.load_moral_scenarios()
        if args.moral_per_cat:
            bycat = defaultdict(list)
            for s in scen:
                bycat[s["category"]].append(s)
            scen = [s for c in bycat for s in bycat[c][: args.moral_per_cat]]
        trials += design.build_moral_trials(scen)

    return trials, modules


def cmd_run(args):
    from latest.analysis import report
    from latest.collect import engine, run as run_dispatch

    cfg = load_config()
    models = args.models or models_from_config(cfg)
    results_root = args.results_root or cfg["run"]["results_root"]
    concurrency = args.concurrency if args.concurrency is not None else cfg["run"]["concurrency"]
    run_id = args.run_id or make_run_id()

    for w in validate(cfg):
        print(f"  [config warning] {w}")

    trials, modules = _build_trials(cfg, args)
    by_mod = defaultdict(int)
    for t in trials:
        by_mod[t.module] += 1
    print(f"\nRun {run_id}: {len(trials)} trials across {modules} {dict(by_mod)}")
    print(f"Models: {', '.join(models)}  ·  judges: {', '.join(cfg.get('judges') or [])}")

    rd = L.ensure_run_dir(results_root, run_id)
    freeze.freeze(trials, rd, seed=(cfg.get('run') or {}).get('seed'))
    print(f"Froze design -> {L.trials_path(rd)}")

    engine.collect(trials, models, run_id=run_id, results_root=results_root, cfg=cfg,
                   concurrency=concurrency, run_trial_fn=run_dispatch.run_trial)

    print("\nAnalyzing ...")
    report.analyze(rd)
    problems = [m for s, m in L.verify(L.ledger_path(rd), L.read_manifest(rd)) if s == "error"]
    print(f"verify-ledger: {'clean' if not problems else f'{len(problems)} errors'}")
    print(f"\nDone. Report: {L.report_dir(rd) / 'report.md'}")
    return rd


def cmd_analyze(args):
    from latest.analysis import report

    cfg = load_config()
    results_root = args.results_root or cfg["run"]["results_root"]
    rd = L.run_dir(results_root, args.run_id)
    report.analyze(rd)
    print(f"Report: {L.report_dir(rd) / 'report.md'}")


def cmd_verify(args):
    cfg = load_config()
    results_root = args.results_root or cfg["run"]["results_root"]
    rd = L.run_dir(results_root, args.run_id)
    manifest = L.read_manifest(rd) if L.manifest_path(rd).exists() else None
    problems = L.verify(L.ledger_path(rd), manifest)
    if not problems:
        print("verify-ledger: clean")
    for sev, msg in problems:
        print(f"  [{sev}] {msg}")


def cmd_lock(args):
    """Pin dataset revisions + lock model snapshots for reproducibility."""
    from latest import lock
    from latest.config.loader import load_config, reload, validate

    if not args.no_datasets:
        print("Pinning dataset revisions (querying HuggingFace)...")
        for name, sha in lock.pin_dataset_revisions().items():
            print(f"  {name}: {sha}")
    print("Locking model snapshots (ledger + live probe for any unseen model)...")
    for pm, snap in lock.lock_model_snapshots(run_id=args.run_id, probe_missing=not args.no_probe).items():
        print(f"  {pm}: {snap}")

    reload()  # drop the cached config so validate() sees the new files
    warns = validate(load_config())
    print(f"\nremaining validate() warnings: {len(warns)}")
    for w in warns:
        print(f"  - {w}")


def main():
    p = argparse.ArgumentParser(prog="latest")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="plan -> collect -> analyze (end to end)")
    r.add_argument("--models", nargs="+", default=None)
    r.add_argument("--modules", nargs="+", default=None, choices=["manipulation", "benchmark", "moral"])
    r.add_argument("--modes", nargs="+", default=None)
    r.add_argument("--subjects", nargs="+", default=None)
    r.add_argument("--items-per-subject", type=int, default=None, dest="items_per_subject")
    r.add_argument("--variants", type=int, default=None)
    r.add_argument("--bench-n", type=int, default=None, dest="bench_n")
    r.add_argument("--moral-per-cat", type=int, default=None, dest="moral_per_cat")
    r.add_argument("--concurrency", type=int, default=None)
    r.add_argument("--run-id", default=None, dest="run_id")
    r.add_argument("--results-root", default=None, dest="results_root")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("analyze", help="re-analyze an existing run (no network)")
    a.add_argument("--run-id", required=True, dest="run_id")
    a.add_argument("--results-root", default=None, dest="results_root")
    a.set_defaults(func=cmd_analyze)

    v = sub.add_parser("verify", help="verify-ledger integrity + provenance")
    v.add_argument("--run-id", required=True, dest="run_id")
    v.add_argument("--results-root", default=None, dest="results_root")
    v.set_defaults(func=cmd_verify)

    lk = sub.add_parser("lock-snapshots", help="pin dataset revisions + lock model snapshots")
    lk.add_argument("--run-id", default=None, dest="run_id", help="source snapshots from this run (default: most recent)")
    lk.add_argument("--no-datasets", action="store_true", help="skip dataset-revision pinning")
    lk.add_argument("--no-probe", action="store_true", help="don't make live calls for unseen models")
    lk.set_defaults(func=cmd_lock)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
