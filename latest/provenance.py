"""provenance.py — capture everything needed to reproduce and audit a run.

Fills a RunManifest (records.RunManifest) with:
  - git SHA + dirty flag (was the working tree clean when this ran?)
  - config_hash / pricing_hash (sha256 of eval.yaml / pricing.yaml file bytes)
  - library versions (pydantic, openai, anthropic, datasets, statsmodels, ...)
  - dataset revisions + locked model snapshots (from the config files)
  - seed, models, judges, enabled modules

The manifest is written FIRST, once per run. Every CallRecord is stamped with
the same git_sha + config_hash, so any single ledger line ties back to the
manifest that produced it. Also exposes the shared time helpers used by the
ledger (utc_now_iso, make_run_id).

Depends only on records + config.loader (and stdlib).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

from latest.config.loader import (
    EVAL_PATH,
    PRICING_PATH,
    judges_from_config,
    load_config,
    models_from_config,
)
from latest.records import RunManifest

# latest/provenance.py -> parents[1] == repo root (where .git lives)
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Distribution names to record versions for (best-effort; missing ones skipped).
_TRACKED_LIBS = [
    "pydantic",
    "openai",
    "anthropic",
    "google-genai",
    "groq",
    "datasets",
    "huggingface-hub",
    "scipy",
    "statsmodels",
    "numpy",
    "pandas",
    "rouge-score",
    "bert-score",
    "tenacity",
    "PyYAML",
]


# ───────────────────────────── time helpers ────────────────────────────────


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp, ISO-8601 (used on every CallRecord)."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def make_run_id() -> str:
    """Compact UTC run id, e.g. 20260613T194501Z."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ───────────────────────────── git + hashes ────────────────────────────────


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_info() -> tuple[str | None, bool | None]:
    """Return (commit_sha, dirty). (None, None) if not a git repo / git absent."""
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return None, None
    status = _git("status", "--porcelain")
    dirty = bool(status) if status is not None else None
    return sha, dirty


def file_hash(path: Path | str) -> str | None:
    """sha256 of a file's bytes (None if missing)."""
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def config_hash() -> str | None:
    """Canonical config hash == hash of eval.yaml (stamped on every CallRecord)."""
    return file_hash(EVAL_PATH)


def pricing_hash() -> str | None:
    return file_hash(PRICING_PATH)


def lib_versions() -> dict[str, str]:
    """Best-effort installed versions for the tracked libraries + Python."""
    versions = {"python": platform.python_version()}
    for name in _TRACKED_LIBS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


# ───────────────────────────── manifest build ──────────────────────────────


def build_manifest(cfg: dict | None = None, run_id: str | None = None, notes: str | None = None) -> RunManifest:
    """Assemble the RunManifest for a run. Pure except for git/version probes."""
    cfg = cfg if cfg is not None else load_config()
    run_id = run_id or make_run_id()
    sha, dirty = git_info()

    datasets = cfg.get("_datasets") or {}
    return RunManifest(
        run_id=run_id,
        created_at=utc_now_iso(),
        git_sha=sha,
        git_dirty=dirty,
        config_hash=config_hash(),
        pricing_hash=pricing_hash(),
        seed=(cfg.get("run") or {}).get("seed"),
        models=models_from_config(cfg),
        judges=judges_from_config(cfg),
        modules=[m for m in ("benchmark", "manipulation", "moral") if (cfg.get(m) or {}).get("enabled")],
        dataset_revisions={name: (spec or {}).get("revision") for name, spec in datasets.items()},
        snapshots=dict(cfg.get("_snapshots") or {}),
        lib_versions=lib_versions(),
        notes=notes,
    )


def _main() -> None:
    """`python -m latest.provenance` — print a sample manifest."""
    import json

    m = build_manifest(notes="provenance self-check")
    print(json.dumps(m.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    _main()
