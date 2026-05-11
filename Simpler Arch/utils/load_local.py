"""Local JSONL dataset loaders (manipulation attacks/drift, moral scenarios)."""
import json


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_attacks(path="data/manipulation_attacks.jsonl"):
    """{attack_name: template_string} for single-shot manipulation attacks."""
    return {row["name"]: row["template"] for row in _load_jsonl(path)}


def load_drift_turns(path="data/manipulation_drift.jsonl"):
    """{turn_name: template_string} for the 6-turn drift sequence (turns 2-6)."""
    return {row["name"]: row["template"] for row in _load_jsonl(path)}


def load_moral_scenarios(path="data/moral_scenarios.jsonl"):
    """Returns {category: [scenario_dict, ...]} grouped by 'category' field."""
    out = {}
    for row in _load_jsonl(path):
        out.setdefault(row["category"], []).append(row)
    return out
