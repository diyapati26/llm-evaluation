"""File-backed JSON cache for LLM responses.

Cache key format: f"{dataset}::{model}::{question}"
Same shape as Harsha's notebook AI_Cache.json — interoperable.
"""
import json
import os

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "AI_Cache.json",
)


def load_cache(path=DEFAULT_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache, path=DEFAULT_PATH):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def make_key(dataset, model, question):
    # Match Harsha's notebook key shape: f"{subject}::{model}::{question}".
    # `dataset` here can be a per-row "subject" for MMLU or a flat name like "hellaswag".
    return f"{dataset}::{model}::{question}"


def get(cache, dataset, model, question):
    return cache.get(make_key(dataset, model, question))


def put(cache, dataset, model, question, value):
    cache[make_key(dataset, model, question)] = value
    return cache
