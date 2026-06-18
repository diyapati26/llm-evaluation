"""cache.py — content-addressed response cache, shared across runs.

A directory holding one JSON file per `call_id` (the content-address of the
request). The key includes the FULL prior message history, so the multi-turn
manipulation path caches too — the single biggest cost gap in the old
framework, where every manipulation run re-billed 100% of its calls.

Design choices:
  - One file per key (not one giant JSON): O(1) writes (no whole-file rewrite),
    and distinct keys are distinct files, so concurrent threads writing
    different keys need no global lock.
  - Atomic writes via temp-file + os.replace, so a crash can't leave a
    half-written cache entry.
  - Shared across runs (lives at <results_root>/cache), so the FIRST run pays
    and every identical re-run is free.

Stored value is a ProviderResponse-as-dict (provider, model, model_version,
temperature, tokens, cost_usd, latency_ms, answer/raw/text). On a hit the
provider layer reconstructs the response and logs a CallRecord with
cache_hit=True; "new spend" for a run is then sum(cost where not cache_hit).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from latest.records import make_call_id


def call_key(
    provider: str,
    model_id: str,
    temperature: float,
    schema_name: str,
    messages: list,
    max_tokens: int | None = None,
    seed: int | None = None,
) -> str:
    """Content-address a request -> call_id (== cache key).

    Everything that can change the response goes into the key. `model_id` is the
    pinned dated snapshot when one is locked (so a provider snapshot rotation
    invalidates the cache); otherwise the alias (the documented tradeoff:
    cacheable now, may serve stale across an un-pinned rotation). `messages` is
    the full history sent THIS turn, which is what makes multi-turn cacheable.
    """
    return make_call_id(
        {
            "provider": provider,
            "model": model_id,
            "temperature": temperature,
            "schema": schema_name,
            "max_tokens": max_tokens,
            "seed": seed,
            "messages": messages,
        }
    )


class Cache:
    """Directory-backed, content-addressed response cache."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, call_id: str) -> Path:
        return self.root / f"{call_id}.json"

    def has(self, call_id: str) -> bool:
        return self._path(call_id).exists()

    def get(self, call_id: str) -> dict | None:
        p = self._path(call_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # treat a corrupt entry as a miss; it will be overwritten

    def put(self, call_id: str, value: dict) -> None:
        """Atomic write: temp file in the same dir, then os.replace."""
        data = json.dumps(value, default=str, ensure_ascii=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, self._path(call_id))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def __contains__(self, call_id: str) -> bool:
        return self.has(call_id)

    def __len__(self) -> int:
        return sum(1 for p in self.root.glob("*.json"))
