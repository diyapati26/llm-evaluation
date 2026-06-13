"""loaders.py — stimulus loaders.

Local JSONL templates (attacks / drift / moral) + HuggingFace benchmark datasets
(MMLU / HellaSwag / TruthfulQA), normalized into a uniform `item` dict:

    {item_id, dataset, subject, question, choices, correct_answer, refs}

  - question       : raw stem (NOT numbered); render_mc_prompt() adds choices.
  - choices        : list[str] for MC datasets; None for generation.
  - correct_answer : 1-indexed string for MC; None for generation.
  - refs           : {best_answer, correct_answers, incorrect_answers} for truthfulqa_gen.

Sampling is seeded (random.Random over a fixed dataset order) so the same seed +
pinned revision yields the same items. HF loads pass the pinned `revision` from
datasets.yaml when set. Local files live in latest/data/.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from latest.config.loader import dataset_spec, load_config

_DATA = Path(__file__).resolve().parent / "data"


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ───────────────────────────── local templates ────────────────────────────


def load_attacks() -> dict[str, list[str]]:
    """{attack_name: [variant, ...]} from manipulation_attacks.jsonl."""
    return {r["name"]: r["variants"] for r in _read_jsonl(_DATA / "manipulation_attacks.jsonl")}


def load_drift_turns() -> dict[str, str]:
    """{turn_name: template} from manipulation_drift.jsonl (insertion order preserved)."""
    return {r["name"]: r["template"] for r in _read_jsonl(_DATA / "manipulation_drift.jsonl")}


def load_moral_scenarios() -> list[dict]:
    return _read_jsonl(_DATA / "moral_scenarios.jsonl")


# ───────────────────────────── prompt rendering ────────────────────────────


def render_mc_prompt(question: str, choices: list[str]) -> str:
    """'question\\n1. choice0\\n2. choice1 ...' — works for any choice count."""
    return "\n".join([question, *[f"{i + 1}. {c}" for i, c in enumerate(choices)]])


# ───────────────────────────── HF datasets ─────────────────────────────────


def _revision(name: str) -> str | None:
    return (dataset_spec(name) or {}).get("revision")


def _sample_idx(n_total: int, n_want: int, seed_key: str) -> list[int]:
    idxs = list(range(n_total))
    random.Random(seed_key).shuffle(idxs)
    return [int(i) for i in idxs[:n_want]]


def load_hard_mmlu_items(subjects: list[str], items_per_subject: int, seed: int) -> list[dict]:
    """Hard-MMLU item pool for the manipulation module (one config per subject)."""
    from datasets import load_dataset

    rev = _revision("mmlu")
    items: list[dict] = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test", revision=rev)
        for i in _sample_idx(len(ds), items_per_subject, f"{seed}:mmlu:{subj}"):
            ex = ds[i]
            items.append({
                "item_id": f"mmlu/{subj}/{i}",
                "dataset": "mmlu",
                "subject": subj,
                "question": ex["question"],
                "choices": list(ex["choices"]),
                "correct_answer": str(int(ex["answer"]) + 1),
                "refs": None,
            })
    return items


def load_benchmark_items(name: str, n: int, seed: int) -> list[dict]:
    """Normalized items for a benchmark dataset."""
    from datasets import load_dataset

    rev = _revision(name)

    if name == "mmlu":
        # Spread across the configured hard subjects (lighter than the 'all' config).
        subjects = (load_config().get("dataset") or {}).get("hard_mmlu") or ["abstract_algebra"]
        per = max(1, math.ceil(n / len(subjects)))
        items = load_hard_mmlu_items(subjects, per, seed)
        return items[:n]

    if name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation", revision=rev)
        out = []
        for i in _sample_idx(len(ds), n, f"{seed}:hellaswag"):
            ex = ds[i]
            out.append({
                "item_id": f"hellaswag/{i}", "dataset": "hellaswag", "subject": "hellaswag",
                "question": ex["ctx"], "choices": list(ex["endings"]),
                "correct_answer": str(int(ex["label"]) + 1), "refs": None,
            })
        return out

    if name == "truthfulqa_mc":
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation", revision=rev)
        out = []
        for i in _sample_idx(len(ds), n, f"{seed}:tqa_mc"):
            ex = ds[i]
            choices = list(ex["mc1_targets"]["choices"])
            labels = list(ex["mc1_targets"]["labels"])
            correct = str(labels.index(1) + 1) if 1 in labels else None
            out.append({
                "item_id": f"truthfulqa_mc/{i}", "dataset": "truthfulqa_mc", "subject": "truthfulqa_mc",
                "question": ex["question"], "choices": choices, "correct_answer": correct, "refs": None,
            })
        return out

    if name == "truthfulqa_gen":
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", revision=rev)
        out = []
        for i in _sample_idx(len(ds), n, f"{seed}:tqa_gen"):
            ex = ds[i]
            out.append({
                "item_id": f"truthfulqa_gen/{i}", "dataset": "truthfulqa_gen", "subject": "truthfulqa_gen",
                "question": ex["question"], "choices": None, "correct_answer": None,
                "refs": {
                    "best_answer": ex.get("best_answer"),
                    "correct_answers": list(ex.get("correct_answers") or []),
                    "incorrect_answers": list(ex.get("incorrect_answers") or []),
                },
            })
        return out

    raise ValueError(f"unknown benchmark dataset '{name}'")
