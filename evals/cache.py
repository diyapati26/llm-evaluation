import json
import hashlib
import os
from pathlib import Path
from evals.schemas import LLMResponse
from datetime import datetime

class ResponseCache:
    """
    Saves every LLM response to disk as JSON.
    Key = hash of (model + prompt).
    If the same model gets the same prompt twice, 
    second call loads from disk — zero API cost.
    """

    def __init__(self, cache_dir: str = "results/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _make_key(self, model: str, prompt: str) -> str:
        """
        Create a unique filename from model + prompt.
        We hash them so filenames stay short and safe
        regardless of how long the prompt is.
        """
        raw = f"{model}::{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, model: str, prompt: str) -> LLMResponse | None:
        """
        Try to load a cached response.
        Returns None if not cached yet.
        """
        key = self._make_key(model, prompt)
        path = self.cache_dir / f"{key}.json"

        if path.exists():
            self._hits += 1
            data = json.loads(path.read_text())
            # Convert timestamp string back to datetime
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            return LLMResponse(**data)

        self._misses += 1
        return None

    def save(self, response: LLMResponse, prompt: str) -> None:
        """
        Save a response to disk.
        Called immediately after every real API call.
        """
        key = self._make_key(response.model, prompt)
        path = self.cache_dir / f"{key}.json"

        data = response.model_dump()
        # Convert datetime to string for JSON serialization
        data["timestamp"] = data["timestamp"].isoformat()

        path.write_text(json.dumps(data, indent=2))

    def stats(self) -> dict:
        """How many cache hits vs misses this session."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "saved":    f"~${self._hits * 0.000165:.4f}"
        }