"""File-backed JSON cache for LLM responses.

Cache key format: f"{dataset}::{model_version}::{question}"
- model_version is the dated snapshot returned by the API (e.g.
  "gpt-5.4-mini-2026-01-15"), NOT the alias. Two runs against the same alias
  resolve to the same snapshot and share the cache; if the provider rotates
  the snapshot, the cache invalidates automatically (different key).

Since lookups happen BEFORE the API call (when we only know the alias), we
keep a `_versions` sub-dict in the cache mapping alias -> last-known version.
Pre-call lookup uses this mapping to construct the version key.

First-time use of an alias misses (no mapping yet); subsequent calls hit.
"""
import json
import os

DEFAULT_PATH = "AI_Cache.json"
VERSIONS_KEY = "_versions"  # special key inside the cache dict: {alias: model_version}


def load_cache(path=DEFAULT_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(cache, path=DEFAULT_PATH):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def make_key(dataset, model_version, question):
    return f"{dataset}::{model_version}::{question}"


def _known_version(cache, model_alias):
    """Return the last-seen model_version for an alias (or None)."""
    return cache.get(VERSIONS_KEY, {}).get(model_alias)


def get(cache, dataset, model_alias, question, model_version=None):
    """Look up a cached response.

    If model_version is provided explicitly, use it.
    Otherwise check the alias->version mapping populated by prior puts.
    If neither is available, fall back to keying by alias (first-ever lookup).
    """
    version = model_version or _known_version(cache, model_alias) or model_alias
    return cache.get(make_key(dataset, version, question))


def put(cache, dataset, model_alias, question, value, model_version=None):
    """Store under the version-keyed slot and remember alias->version.

    If model_version is missing (some providers may not return one), key by
    the alias instead — preserves cacheability at the cost of reproducibility.
    """
    version = model_version or model_alias
    cache[make_key(dataset, version, question)] = value
    if model_version:
        versions = cache.setdefault(VERSIONS_KEY, {})
        versions[model_alias] = model_version
    return cache
