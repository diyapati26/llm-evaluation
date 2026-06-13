"""latest/main.py — production entry point (interactive launcher + headless flags).

Interactive (just `python -m latest.main`): asks, in order —
  1. Run the offline test suite?            (pytest, no spend)
  2. Run a SMOKE test? (config/smoke.yaml)   tiny run across ALL modules + datasets
  3. Run the COMPREHENSIVE run? (eval.yaml)  the full configured run
Each actual run does its own preflight (modules + API keys + a live connectivity
check) and then collects with constant saves — every API call fsync'd to the
ledger, every event fsync'd to run.log.jsonl. On a budget wall it pauses and asks
whether to resume (yes = topped up; no = stop and keep everything saved).

Headless (for automation / CI):
  python -m latest.main --smoke            # run smoke, no prompts
  python -m latest.main --comprehensive    # run full config, no prompts
  python -m latest.main --smoke --comprehensive   # smoke then full
  flags: --skip-tests  --skip-live  --yes  + any override (--models, --bench-n, ...)
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
from latest.config.loader import load_config, load_smoke, models_from_config  # noqa: E402
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

_OFFLINE_TESTS = ["latest/tests/test_offline.py", "latest/tests/test_plan.py",
                  "latest/tests/test_analysis.py", "latest/tests/test_infra.py"]

# Fields config/smoke.yaml may override (CLI flags still win over the file).
_SMOKE_FIELDS = ("models", "modules", "subjects", "items_per_subject", "variants",
                 "bench_n", "moral_per_cat", "concurrency")


def _emit(runlog, event, level="info", **fields):
    """Log to the run's RunLog if present, else print (used during pre-run prompts)."""
    if runlog is not None:
        return runlog.log(event, level=level, **fields)
    print(f"  [{level}] {event} " + " ".join(f"{k}={v}" for k, v in fields.items()).rstrip())
    return None


def _real_key(val: str) -> bool:
    return bool(val) and "your_" not in val and "_here" not in val


def _prompt(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"{question} {suffix} ").strip().lower()
    except EOFError:
        return default
    return default if not ans else ans.startswith("y")


# ───────────────────────────── preflight checks ────────────────────────────


def check_modules(runlog=None) -> bool:
    missing = []
    for m in _REQUIRED_MODULES:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            missing.append(f"{m}: {e}")
    _emit(runlog, "preflight_modules", level=("info" if not missing else "error"),
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
    _emit(runlog, "preflight_env", level=("info" if not missing else "error"), ok=not missing, missing=missing)
    return not missing


def run_tests(runlog=None) -> bool:
    cmd = [sys.executable, "-m", "pytest", *_OFFLINE_TESTS, "-q"]
    res = subprocess.run(cmd, cwd=_REPO_ROOT, env={**os.environ, "PYTHONUTF8": "1"})
    _emit(runlog, "preflight_tests", level=("info" if res.returncode == 0 else "error"),
          ok=res.returncode == 0, returncode=res.returncode)
    return res.returncode == 0


def check_live(runlog, models, judges) -> bool:
    """Minimal REAL calls confirming keys WORK and both code paths run: a structured
    multi-turn Conversation per subject model + a free-form chat per judge."""
    from latest.providers import chat, start_conversation
    from latest.records import ReasonedAnswer

    q = ("Connectivity check. What is 1+1?\n1. 2\n2. 3\n3. 4\n4. 5\n"
         "Answer with the option number, your confidence 1-5, and one short sentence.")
    failures = []

    for m in models:
        try:
            r = start_conversation(m).send(q, ReasonedAnswer, max_tokens=256)
            ok = bool(r.raw and r.raw.get("letter"))
            _emit(runlog, "preflight_live_conversation", level=("info" if ok else "error"), model=m,
                  ok=ok, letter=(r.raw or {}).get("letter"), model_version=r.model_version)
            if not ok:
                failures.append(f"{m}: empty structured answer")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{m} (conversation): {e}")
            _emit(runlog, "preflight_live_conversation", level="error", model=m, ok=False, error=str(e)[:200])

    for j in judges:
        try:
            r = chat(j, [{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=20)
            ok = bool((r.text or "").strip())
            _emit(runlog, "preflight_live_chat", level=("info" if ok else "error"), judge=j,
                  ok=ok, model_version=r.model_version)
            if not ok:
                failures.append(f"{j}: empty chat reply")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{j} (chat): {e}")
            _emit(runlog, "preflight_live_chat", level="error", judge=j, ok=False, error=str(e)[:200])

    _emit(runlog, "preflight_live", level=("info" if not failures else "error"), ok=not failures, failures=failures)
    return not failures


# ───────────────────────────── one run ─────────────────────────────────────


def _execute(args, cfg, smoke: bool):
    """Preflight (env + live) then plan -> collect -> analyze for one run."""
    ns = SimpleNamespace(models=args.models, modules=args.modules, modes=args.modes, subjects=args.subjects,
                         items_per_subject=args.items_per_subject, variants=args.variants,
                         bench_n=args.bench_n, moral_per_cat=args.moral_per_cat, concurrency=args.concurrency)
    if smoke:
        sm = load_smoke()
        for f in _SMOKE_FIELDS:
            if getattr(ns, f, None) is None and f in sm:
                setattr(ns, f, sm[f])

    models = ns.models or models_from_config(cfg)
    judges = cfg.get("judges") or []
    results_root = args.results_root or cfg["run"]["results_root"]
    concurrency = ns.concurrency or cfg["run"]["concurrency"]
    run_id = (args.run_id or make_run_id()) + ("-smoke" if smoke else "")
    rd = L.ensure_run_dir(results_root, run_id)

    with RunLog(rd / "run.log.jsonl") as runlog:
        mode = "smoke" if smoke else "comprehensive"
        runlog.log("run_begin", mode=mode, run_id=run_id, models=models, judges=judges)

        if not check_env(runlog, models, judges):
            runlog.log("run_aborted", level="error", reason="missing API keys")
            print("Missing API keys — aborting this run.")
            return None
        if not args.skip_live and not check_live(runlog, models, judges):
            runlog.log("run_aborted", level="error", reason="live connectivity failed")
            print("Live key/connectivity check failed — aborting this run.")
            return None

        trials, modules = _build_trials(cfg, ns)
        by_mod = defaultdict(int)
        for t in trials:
            by_mod[t.module] += 1
        runlog.log("design_built", n_trials=len(trials), modules=modules, by_module=dict(by_mod))

        from latest.plan import freeze
        freeze.freeze(trials, rd, seed=(cfg.get("run") or {}).get("seed"))
        runlog.log("design_frozen", path=str(L.trials_path(rd)))

        from latest.collect import engine, run as run_dispatch
        engine.collect(trials, models, run_id=run_id, results_root=results_root, cfg=cfg,
                       concurrency=concurrency, run_trial_fn=run_dispatch.run_trial, runlog=runlog)

        from latest.analysis import report
        report.analyze(rd)
        errors = [m for s, m in L.verify(L.ledger_path(rd), L.read_manifest(rd)) if s == "error"]
        runlog.log("verify_ledger", level=("info" if not errors else "error"), ok=not errors, n_errors=len(errors))
        runlog.log("run_complete", report=str(L.report_dir(rd) / "report.md"))

    print(f"\n[{mode}] done. Report: {L.report_dir(rd) / 'report.md'}  ·  log: {rd / 'run.log.jsonl'}")
    return rd


# ───────────────────────────── entry point ─────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(prog="latest.main")
    p.add_argument("--smoke", action="store_true", help="run the smoke test (config/smoke.yaml), no prompts")
    p.add_argument("--comprehensive", action="store_true", help="run the full config/eval.yaml, no prompts")
    p.add_argument("--yes", "-y", action="store_true", help="assume yes to prompts")
    p.add_argument("--skip-tests", action="store_true", dest="skip_tests")
    p.add_argument("--skip-live", action="store_true", dest="skip_live", help="skip the live connectivity preflight")
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
    args = p.parse_args()

    if not check_modules():
        print("PREFLIGHT FAILED — module import error. See above.")
        return 1

    # ── Headless / flag-driven ──────────────────────────────────────────────
    if args.smoke or args.comprehensive:
        if not args.skip_tests and not run_tests() and not args.yes:
            print("Offline tests FAILED — aborting (use --skip-tests or --yes to override).")
            return 1
        cfg = load_config()
        if args.smoke:
            _execute(args, cfg, smoke=True)
        if args.comprehensive:
            _execute(args, cfg, smoke=False)
        return 0

    if not sys.stdin.isatty():
        print("Non-interactive shell and no run mode given. "
              "Use:  python -m latest.main --smoke   (or --comprehensive).")
        return 0

    # ── Interactive ─────────────────────────────────────────────────────────
    cfg = load_config()
    print("\n=== latest — interactive launcher ===")
    if _prompt("Run the offline test suite first?", default=True):
        if not run_tests() and not _prompt("Tests FAILED. Continue anyway?", default=False):
            return 1
    if _prompt("Run a SMOKE test now? (config/smoke.yaml — all modules + datasets, tiny)", default=True):
        _execute(args, cfg, smoke=True)
        if _prompt("Smoke done. Run the COMPREHENSIVE run now? (full config/eval.yaml — costs more)", default=False):
            _execute(args, cfg, smoke=False)
    elif _prompt("Run the COMPREHENSIVE run now? (full config/eval.yaml — costs more)", default=False):
        _execute(args, cfg, smoke=False)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
