"""lock.py — pin dataset revisions + model snapshots for reproducibility.

Clears the two reproducibility warnings from config.loader.validate():
  - datasets.yaml   : writes each dataset's current HuggingFace commit SHA as its
                      `revision`, so the same seed samples the same questions forever.
  - snapshots.lock.yaml : writes each configured model/judge's dated snapshot
                      (alias -> e.g. claude-haiku-4-5-20251001), sourced from a run's
                      ledger (most recent by default) plus a tiny live probe for any
                      model/judge not seen in a run yet.

Invoked via `python -m latest.cli lock-snapshots`. Idempotent; re-run anytime.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from latest import ledger as L
from latest.config import loader
from latest.config.loader import (
    DATASETS_PATH,
    SNAPSHOTS_PATH,
    judges_from_config,
    load_config,
    models_from_config,
)
from latest.providers.router import resolve


# ───────────────────────────── datasets ────────────────────────────────────


def pin_dataset_revisions() -> dict:
    """Query HF for each dataset's current commit SHA and write it as `revision`."""
    from huggingface_hub import HfApi

    api = HfApi()
    datasets = loader._read_yaml(DATASETS_PATH)
    pinned = {}
    for name, spec in datasets.items():
        hf_id = (spec or {}).get("hf_id")
        try:
            sha = api.dataset_info(hf_id).sha
        except Exception as e:  # noqa: BLE001 - keep any existing pin on failure
            sha = (spec or {}).get("revision")
            print(f"  [warn] could not fetch revision for {name} ({hf_id}): {e}")
        spec["revision"] = sha
        pinned[name] = sha
    _write_datasets(datasets)
    return pinned


def _write_datasets(datasets: dict) -> None:
    header = (
        "# Dataset sources + pinned revisions.\n"
        "# `revision` is the exact HF commit SHA so the same seed samples the same\n"
        "# questions across machines and time. Auto-written by `lock-snapshots`.\n\n"
    )
    DATASETS_PATH.write_text(header + yaml.safe_dump(datasets, sort_keys=False), encoding="utf-8")


# ───────────────────────────── snapshots ───────────────────────────────────


def _versions_from_runs(run_id: str | None, results_root: str) -> dict:
    """{bare_model_alias: model_version} gathered from run ledger(s)."""
    root = Path(results_root)
    if run_id:
        run_dirs = [root / run_id]
    else:
        run_dirs = sorted((d for d in root.glob("*") if d.is_dir() and d.name != "cache"), reverse=True)
    by_alias: dict = {}
    for rd in run_dirs:
        lp = L.ledger_path(rd)
        if not lp.exists():
            continue
        try:
            records = L.read(lp)
        except ValueError:
            continue
        for rec in records:
            if rec.model_version and rec.model_alias and rec.model_alias not in by_alias:
                by_alias[rec.model_alias] = rec.model_version
    return by_alias


def _probe_version(provider_model: str) -> str | None:
    """One tiny live call to capture a model's dated snapshot."""
    from latest.providers import chat

    try:
        r = chat(provider_model, [{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=8)
        return r.model_version
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not probe snapshot for {provider_model}: {e}")
        return None


def lock_model_snapshots(run_id: str | None = None, probe_missing: bool = True) -> dict:
    cfg = load_config()
    wanted = list(dict.fromkeys(models_from_config(cfg) + judges_from_config(cfg)))
    by_alias = _versions_from_runs(run_id, cfg["run"]["results_root"])

    out: dict = {}
    for pm in wanted:
        bare = resolve(pm)[1]
        snap = by_alias.get(bare)
        if snap is None and probe_missing:
            snap = _probe_version(pm)
        out[pm] = snap
    _write_snapshots(out)
    return out


def _write_snapshots(out: dict) -> None:
    lines = [
        "# Locked model snapshots — 'provider:model' -> dated snapshot.",
        "# Auto-written by `python -m latest.cli lock-snapshots`. Commit with results so",
        "# the paper cites exact snapshots.",
        "",
    ]
    for pm, snap in out.items():
        lines.append(f"{pm}: {snap if snap is not None else 'null'}")
    SNAPSHOTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
