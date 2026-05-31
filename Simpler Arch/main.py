"""Run all three evals one-by-one in a single process.

Order: (1) standard benchmark  (2) manipulation resistance  (3) moral / empathy.
Each stage is config-driven (config/config.yaml) and isolated in try/except, so a
failure in one stage still lets the others run. Every stage writes its own JSON
(and the manipulation stage also a text report) into results/.

Usage:
    python main.py                      # full run, config defaults
    python main.py --n 4                # shrink sample size for both benchmark & manipulation
    python main.py --models groq:openai/gpt-oss-120b
    python main.py --skip benchmark moral   # run only manipulation
    python main.py --mode stateful          # manipulation mode override
"""
import argparse
import sys
import traceback

# Force UTF-8 stdout/stderr so Unicode in the reports (−, █, ✓, …) doesn't crash
# with a cp1252 UnicodeEncodeError when output is redirected to a file on Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

import run_manipulation
import run_moral
import runner
from load_config import load_config, models_from_config

load_dotenv()


def _banner(n, title):
    print("\n" + "#" * 96)
    print(f"#  STAGE {n}/3 — {title}")
    print("#" * 96)


def main():
    p = argparse.ArgumentParser(description="Run benchmark + manipulation + moral evals sequentially.")
    p.add_argument("--models", nargs="+", default=None, help="Override the model list from config.")
    p.add_argument("--n", type=int, default=None,
                   help="Override sample size for BOTH benchmark (per dataset/subject) and manipulation (total).")
    p.add_argument("--mode", choices=["stateless", "stateful", "both"], default=None,
                   help="Manipulation mode override.")
    p.add_argument("--skip", nargs="+", default=[],
                   choices=["benchmark", "manipulation", "moral"],
                   help="Stages to skip.")
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()

    cfg = load_config()
    models = args.models or models_from_config(cfg)
    results_dir = args.results_dir or cfg["results_dir"]
    print(f"Models: {', '.join(models)}")
    print(f"Results dir: {results_dir}")
    print(f"Skipping: {', '.join(args.skip) or '(none)'}")

    # ── Stage 1: standard benchmark (MMLU / HellaSwag / TruthfulQA) ──
    if "benchmark" not in args.skip:
        _banner(1, "STANDARD BENCHMARK (accuracy)")
        try:
            bench = cfg["benchmark"]
            n = args.n if args.n is not None else bench["n"]
            runner.run(bench["datasets"], models, n, results_dir,
                       concurrency=bench["concurrency"])
        except Exception:
            print("\n[benchmark stage FAILED — continuing]")
            traceback.print_exc()

    # ── Stage 2: manipulation resistance ────────────────────────────
    if "manipulation" not in args.skip:
        _banner(2, "MANIPULATION RESISTANCE")
        try:
            manip = cfg["manipulation"]
            n = args.n if args.n is not None else manip["n"]
            run_manipulation.run(
                models, n, results_dir,
                include_drift=manip["include_drift"],
                concurrency=manip["concurrency"],
                mode=args.mode or manip.get("mode", "both"),
            )
        except Exception:
            print("\n[manipulation stage FAILED — continuing]")
            traceback.print_exc()

    # ── Stage 3: moral / empathy ────────────────────────────────────
    if "moral" not in args.skip:
        _banner(3, "MORAL / EMPATHY")
        try:
            moral_cfg = cfg.get("moral") or {}
            judge = moral_cfg.get("judge_model") or cfg["judge_model"]
            run_moral.run(models, judge_model=judge, results_dir=results_dir)
        except Exception:
            print("\n[moral stage FAILED — continuing]")
            traceback.print_exc()

    print("\n" + "=" * 96)
    print("ALL STAGES COMPLETE — see results/ for JSON + reports")
    print("=" * 96)


if __name__ == "__main__":
    main()
