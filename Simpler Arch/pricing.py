"""Centralized model pricing — single source of truth in config/pricing.yaml.

Per-1M-token USD {input, output} rates. Each provider's estimate_cost() calls
get_price(provider, model, default=...) instead of carrying its own table, so
prices live in one auditable, web-verifiable file.

The YAML is loaded once and cached. If the file is missing or a model isn't
listed, get_price() returns the caller's `default` — so providers keep working
(and keep their old hardcoded dict as that fallback) even without the YAML.
"""
import os
import threading

import yaml

_PATH = os.path.join(os.path.dirname(__file__), "config", "pricing.yaml")
_cache = None
_lock = threading.Lock()


def _load():
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:  # double-checked: another thread may have loaded
                try:
                    with open(_PATH, encoding="utf-8") as f:
                        _cache = yaml.safe_load(f) or {}
                except FileNotFoundError:
                    _cache = {}
    return _cache


def get_price(provider, model, default=None):
    """Return {'input': $/1M, 'output': $/1M} for (provider, model).

    Falls back to `default` when the YAML is absent or the model is unlisted.
    """
    return _load().get(provider, {}).get(model, default)
