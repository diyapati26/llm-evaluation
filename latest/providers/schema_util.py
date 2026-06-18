"""schema_util.py — the ONE _make_strict.

In the old Simpler Arch this helper was triplicated. Centralized here: recursively
pin additionalProperties=False on every object node of a JSON schema, which Groq
and (often) OpenRouter require for response_format=json_schema strict mode.
Pydantic v2's model_json_schema() doesn't emit it.
"""
from __future__ import annotations

from typing import Any


def make_strict(node: Any) -> Any:
    """Recursively set additionalProperties=False on every object node (in place)."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "additionalProperties" not in node:
            node["additionalProperties"] = False
        for v in node.values():
            make_strict(v)
    elif isinstance(node, list):
        for v in node:
            make_strict(v)
    return node
