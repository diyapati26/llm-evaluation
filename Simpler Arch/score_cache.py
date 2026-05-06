"""Score Harsha's existing AI_Cache.json against MMLU ground truth.

The notebook ran gpt-5.4-mini against the hard-MMLU subjects and cached every
(subject, model, question) → response. This script joins those cached
responses against MMLU's correct answers and writes a results JSON in the
same shape as runner.py — so compare.py picks it up alongside fresh runs.

No API key required (uses cached responses + open HF dataset).

Usage:
    python "Simpler Arch/score_cache.py"
    python "Simpler Arch/score_cache.py" --cache AI_Cache.json --results-dir results
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from datasets import load_dataset
from providers import openai_provider
import scorers


def load_cache(path):
    with open(path) as f:
        return json.load(f)


def build_truth_table(subjects):
    """Return dict mapping (subject, question) -> correct answer (1-indexed string)."""
    truth = {}
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"   skip {subject}: {e}")
            continue
        for row in ds:
            # MMLU stores answer as int 0-3; we use "1","2","3","4" to match Harsha's notebook
            truth[(subject, row["question"])] = str(int(row["answer"]) + 1)
    return truth


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="AI_Cache.json", help="Path to cache file")
    p.add_argument("--results-dir", default="results", help="Where to write results JSON")
    p.add_argument("--model", default="gpt-5.4-mini", help="Only score entries for this model (default: gpt-5.4-mini)")
    args = p.parse_args()

    cache = load_cache(args.cache)
    print(f"Cache: {len(cache)} entries")

    # Group entries by subject, restricted to the requested model
    entries_by_subject = defaultdict(list)
    for key, val in cache.items():
        if not isinstance(val, dict):
            continue
        if val.get("model") != args.model:
            continue
        subject = val.get("subject")
        if not subject:
            continue
        entries_by_subject[subject].append(val)

    if not entries_by_subject:
        print(f"No entries found for model={args.model}")
        return

    print(f"\nLoaded entries for {args.model} across {len(entries_by_subject)} subjects:")
    for s, items in entries_by_subject.items():
        print(f"   {s:32s} {len(items):4d}")

    print("\nLoading MMLU ground truth from HF...")
    truth = build_truth_table(list(entries_by_subject.keys()))
    print(f"   {len(truth)} canonical (subject, question) pairs loaded")

    # Score
    per_subject_scores = defaultdict(list)
    total_input = 0
    total_output = 0
    total_latency_sec = 0.0
    misses = 0

    for subject, items in entries_by_subject.items():
        for entry in items:
            q = entry["question"]
            pred = str(entry.get("ai_response", "")).strip()
            correct = truth.get((subject, q))
            if correct is None:
                misses += 1
                continue
            score = scorers.exact_match(pred, correct)
            per_subject_scores[subject].append(score)
            total_input += entry.get("input_tokens", 0)
            total_output += entry.get("output_tokens", 0)
            total_latency_sec += entry.get("latency_sec", 0) or 0

    if misses:
        print(f"\n   ⚠ {misses} cached entries had no ground truth match (probably MMLU dataset version drift)")

    # Per-subject + overall
    print("\nPer-subject accuracy:")
    flat_scores = []
    for subject, sc in per_subject_scores.items():
        avg = sum(sc) / len(sc) if sc else 0
        flat_scores.extend(sc)
        print(f"   {subject:32s} n={len(sc):3d}  acc={avg*100:5.1f}%")
    overall = sum(flat_scores) / len(flat_scores) if flat_scores else 0
    print(f"\nOVERALL accuracy: {overall*100:.2f}% ({len(flat_scores)} samples)")

    # Cost estimate using our pricing dict
    cost = openai_provider.estimate_cost(args.model, total_input, total_output)
    avg_latency_ms = (total_latency_sec / len(flat_scores) * 1000) if flat_scores else 0

    # Write a results row in the same shape runner.py emits
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = [{
        "run_id":          run_id,
        "model":           args.model,
        "dataset":         "mmlu",
        "n_samples":       len(flat_scores),
        "n_scored":        len(flat_scores),
        "errors":          misses,
        "accuracy":        round(overall, 4),
        "total_cost_usd":  round(cost, 6),
        "avg_latency_ms":  round(avg_latency_ms, 2),
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "source":          "score_cache.py (scored from AI_Cache.json, no fresh API calls)",
    }]
    out_path = os.path.join(args.results_dir, f"results_cache_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Total tokens used originally: {total_input + total_output:,} (~${cost:.4f})")


if __name__ == "__main__":
    main()
