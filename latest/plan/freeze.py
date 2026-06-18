"""freeze.py — materialize Trials to trials.parquet + trials.manifest.json.

Once frozen, the design is immutable input to collection. Object-valued columns
(choices, stateful_order, metadata) are JSON-encoded to keep the Parquet flat
and portable (no nested-type surprises); read_trials() decodes them back.
A design_hash over the sorted trial_ids lets a reviewer confirm two runs share
the exact same design.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from latest import ledger as L
from latest.records import Trial, canonical_json, sha256_hex

_OBJECT_COLS = ("choices", "stateful_order", "metadata")
_INT_COLS = ("variant_idx", "replicate_idx")


def _to_dataframe(trials: list[Trial]) -> pd.DataFrame:
    rows = []
    for t in trials:
        d = t.model_dump()
        for col in _OBJECT_COLS:
            if d.get(col) is not None:
                d[col] = json.dumps(d[col], ensure_ascii=True)
        rows.append(d)
    return pd.DataFrame(rows)


def freeze(trials: list[Trial], rd: Path, seed: int | None = None):
    """Write trials.parquet + trials.manifest.json into run dir `rd`. Returns (path, manifest)."""
    rd.mkdir(parents=True, exist_ok=True)
    df = _to_dataframe(trials)
    path = L.trials_path(rd)
    df.to_parquet(path, index=False)

    manifest = {
        "n_trials": len(trials),
        "seed": seed,
        "design_hash": sha256_hex(canonical_json(sorted(t.trial_id for t in trials))),
        "by_module": dict(Counter(t.module for t in trials)),
        "by_mode": dict(Counter(t.mode for t in trials if t.mode)),
        "by_arm": dict(Counter(t.arm for t in trials if t.arm)),
        "by_attack": dict(Counter(t.attack for t in trials if t.attack)),
    }
    L.trials_manifest_path(rd).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path, manifest


def read_trials(rd_or_path: Path | str) -> list[Trial]:
    """Load trials.parquet back into Trial objects (decoding the JSON columns)."""
    p = Path(rd_or_path)
    if p.is_dir():
        p = L.trials_path(p)
    df = pd.read_parquet(p)

    trials: list[Trial] = []
    for d in df.to_dict("records"):
        for col in _OBJECT_COLS:
            if isinstance(d.get(col), str):
                d[col] = json.loads(d[col])
        # Parquet stores optional ints as float w/ NaN; normalize NaN->None, float->int.
        clean = {}
        for k, v in d.items():
            if isinstance(v, float) and pd.isna(v):
                clean[k] = None
            elif k in _INT_COLS and isinstance(v, float):
                clean[k] = int(v)
            else:
                clean[k] = v
        if clean.get("metadata") is None:
            clean["metadata"] = {}
        trials.append(Trial(**clean))
    return trials
