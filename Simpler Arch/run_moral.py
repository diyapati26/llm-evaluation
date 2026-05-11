import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import moral
from load_config import load_config, models_from_config

load_dotenv()


def aggregate(per_model_results):
    """For each model, average scores per category and per axis."""
    summary = {}
    for model, by_cat in per_model_results.items():
        cat_summary = {}
        all_scores_per_axis = defaultdict(list)
        for cat, results in by_cat.items():
            scores_per_axis = defaultdict(list)
            for r in results:
                if r.get("error"):
                    continue
                for axis, v in r["scores"].items():
                    scores_per_axis[axis].append(v)
                    all_scores_per_axis[axis].append(v)
            cat_summary[cat] = {
                axis: round(sum(vs) / len(vs), 2) if vs else 0
                for axis, vs in scores_per_axis.items()
            }
            cat_summary[cat]["n"] = len([r for r in results if not r.get("error")])
        cat_summary["overall"] = {
            axis: round(sum(vs) / len(vs), 2) if vs else 0
            for axis, vs in all_scores_per_axis.items()
        }
        summary[model] = cat_summary
    return summary


def run(models, judge_model="gpt-5.4-mini", results_dir="results"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    by_model = {}
    for model in models:
        print(f"\n--- {model} ---")
        results = moral.run_all(model, judge_model=judge_model)
        by_model[model] = results

        # Quick per-category print
        for cat, items in results.items():
            scored = [r for r in items if not r.get("error")]
            if scored:
                axes = scored[0]["scores"].keys()
                avgs = {a: round(sum(r["scores"][a] for r in scored) / len(scored), 2) for a in axes}
                print(f"  {cat:12s} n={len(scored)}  axes={avgs}")
            else:
                print(f"  {cat:12s} n=0  (errors or empty)")

    summary = aggregate(by_model)

    out_path = os.path.join(results_dir, f"moral_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump({"run_id": run_id, "summary": summary, "raw": by_model}, f, indent=2)
    print(f"\nWrote {out_path}")
    print("\nSUMMARY (avg score 1-5 per axis):")
    for model, cats in summary.items():
        print(f"\n{model}")
        for cat, axes in cats.items():
            print(f"  {cat:12s} {axes}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--results-dir", default=None)
    args = p.parse_args()

    cfg = load_config()
    moral_cfg = cfg.get("moral") or {}  # moral: section may be empty in YAML

    models = args.models or models_from_config(cfg)
    judge_model = args.judge_model or moral_cfg.get("judge_model") or cfg["judge_model"]
    results_dir = args.results_dir or cfg["results_dir"]

    run(models, judge_model=judge_model, results_dir=results_dir)


if __name__ == "__main__":
    main()
