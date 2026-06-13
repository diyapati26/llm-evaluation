"""latest/main.py — production entry point.

Preflight, then run. Preflight (fail-fast) checks, in order:
  1. all required modules import
  2. all API keys for the configured models + judges are present
  3. the offline test suite passes (pytest)
Only if all three pass does it run the full pipeline (plan -> collect -> analyze)
with constant saves — every API call fsync'd to the ledger, every event fsync'd
to run.log.jsonl. On a budget/quota wall it pauses and asks whether to resume.

Usage (full configured run):
    python -m latest.main

Smoke (everything, tiny):
    python -m latest.main --models anthropic:claude-haiku-4-5 openai:gpt-5.4-nano \\
        --subjects formal_logic college_physics --items-per-subject 1 --variants 2 \\
        --bench-n 2 --moral-per-cat 1

    --skip-tests   skip the pytest preflight (not recommended)
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

_REPO_ROOT = Path(__file__).resolve().parents[1]

from latest.env import load_env  # noqa: E402

load_env()

from latest import ledger as L  # noqa: E402
from latest.cli import _build_trials  # noqa: E402  (reuse the trial builder)
from latest.config.loader import load_config, models_from_config  # noqa: E402
from latest.provenance import make_run_id  # noqa: E402
from latest.runlog import RunLog  # noqa: E402

_REQUIRED_MODULES = [
    "latest.records", "latest.provenance", "latest.ledger", "latest.cache", "latest.runlog",
    "latest.config.loader", "latest.loaders",
    "latest.providers", "latest.providers.router", "latest.providers.base",
    "latest.plan.design", "latest.plan.freeze",
    "latest.collect.engine", "latest.collect.run",
    "latest.analysis.score", "latest.analysis.aggregate", "latest.analysis.stats", "latest.analysis.report",
]

_ENV_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY",
}

_OFFLINE_TESTS = ["latest/tests/test_offline.py", "latest/tests/test_plan.py", "latest/tests/test_analysis.py"]


def _real_key(val: str) -> bool:
    return bool(val) and "your_" not in val and "_here" not in val


def check_modules(runlog) -> bool:
    missing = []
    for m in _REQUIRED_MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{m}: {e}")
    runlog.log("preflight_modules", level=("info" if not missing else "error"),
               ok=not missing, n=len(_REQUIRED_MODULES), missing=missing)
    return not missing


def check_env(runlog, models, judges) -> bool:
    missing = []
    for pm in list(models) + list(judges):
        provider = pm.split(":", 1)[0]
        env = _ENV_BY_PROVIDER.get(provider)
        if env is None:
            missing.append(f"{pm}: unknown provider")
        elif not _real_key(os.environ.get(env, "")):
            missing.append(f"{pm}: {env} not set")
    runlog.log("preflight_env", level=("info" if not missing else "error"), ok=not missing, missing=missing)
    return not missing


def run_tests(runlog) -> bool:
    cmd = [sys.executable, "-m", "pytest", *_OFFLINE_TESTS, "-q"]
    env = {**os.environ, "PYTHONUTF8": "1"}
    res = subprocess.run(cmd, cwd=_REPO_ROOT, env=env)
    runlog.log("preflight_tests", level=("info" if res.returncode == 0 else "error"),
               ok=res.returncode == 0, returncode=res.returncode)
    return res.returncode == 0


def main() -> int:
    p = argparse.ArgumentParser(prog="latest.main")
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--modules", nargs="+", default=None, choices=["manipulation", "benchmark", "moral"])
    p.add_argument("--modes", nargs="+", default=None)
    p.add_argument("--subjects", nargs="+", default=None)
    p.add_argument("--items-per-subject", type=int, default=None, dest="items_per_subject")
    p.add_argument("--variants", type=int, default=None)
    p.add_argument("--bench-n", type=int, default=None, dest="bench_n")
    p.add_argument("--moral-per-cat", type=int, default=None, dest="moral_per_cat")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--run-id", default=None, dest="run_id")
    p.add_argument("--results-root", default=None, dest="results_root")
    p.add_argument("--skip-tests", action="store_true", dest="skip_tests")
    args = p.parse_args()

    cfg = load_config()
    models = args.models or models_from_config(cfg)
    judges = cfg.get("judges") or []
    results_root = args.results_root or cfg["run"]["results_root"]
    concurrency = args.concurrency if args.concurrency is not None else cfg["run"]["concurrency"]
    run_id = args.run_id or make_run_id()
    rd = L.ensure_run_dir(results_root, run_id)

    with RunLog(rd / "run.log.jsonl") as runlog:
        runlog.log("run_begin", run_id=run_id, models=models, judges=judges, results_root=str(results_root))

        # ── Preflight (fail-fast) ──────────────────────────────────────────
        ok_mods = check_modules(runlog)
        ok_env = check_env(runlog, models, judges)
        ok_tests = True if args.skip_tests else run_tests(runlog)
        if not (ok_mods and ok_env and ok_tests):
            runlog.log("preflight_failed", level="error", modules=ok_mods, env=ok_env, tests=ok_tests)
            print("\nPREFLIGHT FAILED — not starting the run. See events above.")
            return 1
        runlog.log("preflight_ok")

        # ── Build + freeze ─────────────────────────────────────────────────
        ns = SimpleNamespace(models=models, modules=args.modules, modes=args.modes, subjects=args.subjects,
                             items_per_subject=args.items_per_subject, variants=args.variants,
                             bench_n=args.bench_n, moral_per_cat=args.moral_per_cat)
        trials, modules = _build_trials(cfg, ns)
        by_mod = defaultdict(int)
        for t in trials:
            by_mod[t.module] += 1
        runlog.log("design_built", n_trials=len(trials), modules=modules, by_module=dict(by_mod))

        from latest.plan import freeze
        freeze.freeze(trials, rd, seed=(cfg.get("run") or {}).get("seed"))
        runlog.log("design_frozen", path=str(L.trials_path(rd)))

        # ── Collect (constant saves; interactive budget gate) ──────────────
        from latest.collect import engine, run as run_dispatch
        engine.collect(trials, models, run_id=run_id, results_root=results_root, cfg=cfg,
                       concurrency=concurrency, run_trial_fn=run_dispatch.run_trial, runlog=runlog)

        # ── Analyze + verify ───────────────────────────────────────────────
        from latest.analysis import report
        report.analyze(rd)
        errors = [m for s, m in L.verify(L.ledger_path(rd), L.read_manifest(rd)) if s == "error"]
        runlog.log("verify_ledger", ok=not errors, n_errors=len(errors))
        runlog.log("run_complete", report=str(L.report_dir(rd) / "report.md"))

    print(f"\nDone. Report: {L.report_dir(rd) / 'report.md'}")
    print(f"Run log: {rd / 'run.log.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
