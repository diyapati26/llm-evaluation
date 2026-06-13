"""Config loader for latest.

All paths resolve relative to THIS file, so any runner works regardless of the
current working directory — fixing the cwd-coupling footgun in the old Simpler
Arch `load_config` (which did a bare `open("config/config.yaml")`).

Four YAML files in this directory are loaded and memoized:
  eval.yaml            -> the run config (what to run)
  pricing.yaml         -> authoritative per-1M-token prices
  datasets.yaml        -> HF dataset ids + pinned revisions
  snapshots.lock.yaml  -> alias -> dated model snapshot

`load_config()` returns a fresh, mutable copy each call (so CLI overrides never
poison the cache), with the read-only companion files attached under the
_pricing / _datasets / _snapshots keys.

`validate()` returns human-readable warnings for the reproducibility hazards the
architecture is designed to prevent: missing pricing, unpinned dataset
revisions, and unlocked model snapshots.
"""
from __future__ import annotations

import copy
import threading
import warnings
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
EVAL_PATH = _CONFIG_DIR / "eval.yaml"
PRICING_PATH = _CONFIG_DIR / "pricing.yaml"
DATASETS_PATH = _CONFIG_DIR / "datasets.yaml"
SNAPSHOTS_PATH = _CONFIG_DIR / "snapshots.lock.yaml"
SMOKE_PATH = _CONFIG_DIR / "smoke.yaml"

_lock = threading.Lock()
_cache: dict = {}


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_smoke() -> dict:
    """Smoke-test overrides (config/smoke.yaml). Empty dict if the file is absent."""
    return _read_yaml(SMOKE_PATH)


def _files() -> dict:
    """Load and memoize the four config files (thread-safe, read once)."""
    with _lock:
        if not _cache:
            _cache["eval"] = _read_yaml(EVAL_PATH)
            _cache["pricing"] = _read_yaml(PRICING_PATH)
            _cache["datasets"] = _read_yaml(DATASETS_PATH)
            _cache["snapshots"] = _read_yaml(SNAPSHOTS_PATH)
        return _cache


def reload() -> None:
    """Drop the cache — call after editing a YAML in a long-running process."""
    with _lock:
        _cache.clear()


def load_config() -> dict:
    """Return a fresh, mutable copy of the run config.

    Companion files are attached under _pricing / _datasets / _snapshots.
    Mutating the returned dict (e.g. applying CLI overrides) never affects the
    cached originals — the run config is deep-copied; the read-only companions
    are shared by reference.
    """
    f = _files()
    cfg = copy.deepcopy(f["eval"])
    cfg["_pricing"] = f["pricing"]
    cfg["_datasets"] = f["datasets"]
    cfg["_snapshots"] = f["snapshots"]
    return cfg


def models_from_config(cfg: dict | None = None) -> list[str]:
    """Flatten the per-provider model lists into 'provider:model' strings."""
    cfg = cfg if cfg is not None else load_config()
    out = []
    for provider, models in (cfg.get("models") or {}).items():
        for m in models or []:
            out.append(f"{provider}:{m}")
    return out


def judges_from_config(cfg: dict | None = None) -> list[str]:
    cfg = cfg if cfg is not None else load_config()
    return list(cfg.get("judges") or [])


def get_price(provider: str, model: str, default: dict | None = None) -> dict:
    """Per-1M-token price {input, output} for a model.

    pricing.yaml is authoritative. If the model is absent and no `default` is
    given, warn LOUDLY and return zero — a silent $0 must never slip into a
    reported cost unnoticed (the old code's failure mode).
    """
    prov = _files()["pricing"].get(provider, {})
    if model in prov:
        return prov[model]
    if default is not None:
        return default
    warnings.warn(f"[pricing] no entry for {provider}:{model} — cost computed as $0", stacklevel=2)
    return {"input": 0.0, "output": 0.0}


def dataset_spec(name: str) -> dict:
    """Source spec {hf_id, config, split, revision} for a dataset key."""
    return _files()["datasets"].get(name, {})


def snapshot_for(provider_model: str) -> str | None:
    """Locked dated snapshot for a 'provider:model' alias, or None if unlocked."""
    return _files()["snapshots"].get(provider_model)


def validate(cfg: dict | None = None) -> list[str]:
    """Return reproducibility/cost warnings. Empty list == clean."""
    cfg = cfg if cfg is not None else load_config()
    warns: list[str] = []

    pricing = cfg.get("_pricing", {})
    for pm in models_from_config(cfg) + judges_from_config(cfg):
        provider, model = pm.split(":", 1)
        if model not in pricing.get(provider, {}):
            warns.append(f"pricing: no entry for {pm} (cost will be $0)")

    for name, spec in (cfg.get("_datasets") or {}).items():
        if not (spec or {}).get("revision"):
            warns.append(f"datasets: '{name}' revision is unpinned (not reproducible across time)")

    snaps = cfg.get("_snapshots") or {}
    # Check models AND judges (judge snapshots matter for reproducibility too);
    # dict.fromkeys dedupes a model that's also a judge.
    for pm in dict.fromkeys(models_from_config(cfg) + judges_from_config(cfg)):
        if not snaps.get(pm):
            warns.append(f"snapshots: '{pm}' not locked (run `latest lock-snapshots` after a run)")

    return warns


def _main() -> None:
    """`python -m latest.config.loader` — print the resolved config + warnings."""
    cfg = load_config()
    print("latest config")
    print(f"  config dir : {_CONFIG_DIR}")
    print(f"  seed       : {cfg.get('run', {}).get('seed')}")
    print(f"  results    : {cfg.get('run', {}).get('results_root')}")
    print(f"  models ({len(models_from_config(cfg))}): " + ", ".join(models_from_config(cfg)))
    print(f"  judges ({len(judges_from_config(cfg))}): " + ", ".join(judges_from_config(cfg)))
    enabled = [m for m in ("benchmark", "manipulation", "moral") if (cfg.get(m) or {}).get("enabled")]
    print(f"  modules    : {', '.join(enabled) or '(none enabled)'}")
    manip = cfg.get("manipulation", {})
    print(f"  manip arms : {', '.join(manip.get('arms', []))}")
    print(f"  manip modes: {', '.join(manip.get('modes', []))}  "
          f"(items/subject={manip.get('items_per_subject')}, k={manip.get('resamples')})")

    warns = validate(cfg)
    if warns:
        print(f"\n  {len(warns)} warning(s):")
        for w in warns:
            print(f"    - {w}")
    else:
        print("\n  config clean (no warnings)")


if __name__ == "__main__":
    _main()
