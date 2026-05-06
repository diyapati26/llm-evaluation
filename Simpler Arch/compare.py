"""Read every results/*.json and print a flat comparison table.

Usage (from repo root):
    python "Simpler Arch/compare.py"
    python "Simpler Arch/compare.py" --results-dir results --filter mmlu
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def load_all_results(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "results_*.json"))):
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            entry["_source_file"] = os.path.basename(path)
            rows.append(entry)
    return rows


def print_table(rows, columns, widths):
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns)
        print(line)


def latest_per_combo(rows):
    """For each (model, dataset), keep only the most recent run."""
    key = lambda r: (r.get("model", ""), r.get("dataset", ""))
    by_combo = defaultdict(list)
    for r in rows:
        by_combo[key(r)].append(r)
    return [max(group, key=lambda r: r.get("timestamp", "")) for group in by_combo.values()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--filter", help="Only show rows whose dataset contains this substring")
    p.add_argument("--all-runs", action="store_true",
                   help="Show every run; default is latest per (model, dataset).")
    args = p.parse_args()

    rows = load_all_results(args.results_dir)
    if not rows:
        print(f"No results files in {args.results_dir}/")
        return

    if not args.all_runs:
        rows = latest_per_combo(rows)

    if args.filter:
        rows = [r for r in rows if args.filter in r.get("dataset", "")]

    rows.sort(key=lambda r: (r.get("dataset", ""), -r.get("accuracy", 0)))

    columns = ["dataset", "model", "n_scored", "accuracy", "total_cost_usd", "avg_latency_ms", "errors", "timestamp"]
    widths = {
        "dataset":          16,
        "model":            32,
        "n_scored":          8,
        "accuracy":          9,
        "total_cost_usd":   12,
        "avg_latency_ms":   12,
        "errors":            7,
        "timestamp":        24,
    }

    # Format accuracy as %
    for r in rows:
        if isinstance(r.get("accuracy"), (int, float)):
            r["accuracy"] = f"{r['accuracy']*100:.2f}%"
        if isinstance(r.get("total_cost_usd"), (int, float)):
            r["total_cost_usd"] = f"${r['total_cost_usd']:.4f}"

    print_table(rows, columns, widths)
    print(f"\n{len(rows)} row(s) — source dir: {args.results_dir}")


if __name__ == "__main__":
    main()
