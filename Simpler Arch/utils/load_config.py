"""YAML config loader for the Simpler Arch runners.

Each runner accepts --config <path>. Values from the YAML are used as defaults;
explicit CLI args still override. The grouped `models` block is flattened into
a single list of 'provider:model' strings the router understands.
"""
import os
from pathlib import Path

import yaml


# Path resolution: by default look at Simpler Arch/configs/eval_config.yaml
# regardless of where the runner was invoked from.
_HERE = Path(__file__).resolve().parent          # Simpler Arch/utils
_DEFAULT_PATH = _HERE.parent / "configs" / "eval_config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load YAML into a dict. Returns {} if path is None or the file is missing.

    Callers should treat the returned dict as "config-provided defaults" and
    overlay CLI args on top.
    """
    if path is None:
        path = _DEFAULT_PATH
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def flatten_models(cfg_models: dict | None) -> list[str]:
    """Convert {provider: [model, ...]} into a flat 'provider:model' list.

    Always prefixes with provider name for unambiguous routing — eliminates the
    cross-host name collision risk (e.g. meta-llama/llama-3.3-70b on Groq vs
    OpenRouter would otherwise be indistinguishable in the cache and results).
    """
    if not cfg_models:
        return []
    flat = []
    for provider, models in cfg_models.items():
        if not models:
            continue
        for m in models:
            flat.append(f"{provider}:{m}")
    return flat


def models_from_config(cfg: dict) -> list[str]:
    """Convenience: load config (already loaded), pull 'models', flatten."""
    return flatten_models(cfg.get("models"))
