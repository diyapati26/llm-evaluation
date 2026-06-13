"""Offline tests — no API keys, no network. The framework's correctness core.

Covers: deterministic IDs, the revived ReasonedAnswer, content-addressed +
multi-turn caching, ledger append/read/resume/verify, router resolution, and the
base Conversation template (multi-turn transcript + cache hit on replay).
"""
from __future__ import annotations

import latest.providers as P
from latest import ledger as L
from latest.cache import Cache, call_key
from latest.config import loader
from latest.providers.base import CallContext, Conversation
from latest.records import (
    CallRecord,
    ProviderResponse,
    ReasonedAnswer,
    make_call_id,
    make_trial_id,
)


# ───────────────────────────── records ─────────────────────────────────────


def test_reasoned_answer_has_four_axes():
    ra = ReasonedAnswer(letter="2", confidence="4", acknowledged_counterargument=False,
                        reasoning="Option 2 follows from the premise.")
    assert set(ra.model_dump()) == {"letter", "confidence", "acknowledged_counterargument", "reasoning"}
    # the schema sent to providers must carry all four (so the axes aren't dead)
    assert set(ReasonedAnswer.model_json_schema()["properties"]) == {
        "letter", "confidence", "acknowledged_counterargument", "reasoning"
    }


def test_trial_id_deterministic_and_order_independent():
    c = {"module": "manipulation", "dataset": "mmlu", "item_id": "q1", "attack": "x", "variant_idx": 0}
    assert make_trial_id(c) == make_trial_id(dict(reversed(list(c.items()))))
    assert make_trial_id(c) != make_trial_id({**c, "variant_idx": 1})


def test_call_id_includes_history():
    base = {"provider": "openai", "model": "m", "temperature": 0.0, "schema": "S"}
    k1 = make_call_id({**base, "messages": [{"role": "user", "content": "Q"}]})
    k2 = make_call_id({**base, "messages": [{"role": "user", "content": "Q"},
                                            {"role": "assistant", "content": "2"},
                                            {"role": "user", "content": "no, 3"}]})
    assert k1 != k2  # turn 2 must address differently than turn 1


# ───────────────────────────── cache ───────────────────────────────────────


def test_cache_roundtrip_and_multiturn_keys(run_root):
    c = Cache(L.cache_dir(run_root))
    msgs1 = [{"role": "user", "content": "Q"}]
    msgs2 = msgs1 + [{"role": "assistant", "content": "2"}, {"role": "user", "content": "no"}]
    k1 = call_key("openai", "m", 0.0, "ReasonedAnswer", msgs1)
    k2 = call_key("openai", "m", 0.0, "ReasonedAnswer", msgs2)
    assert k1 != k2
    assert c.get(k1) is None
    c.put(k1, {"cost_usd": 0.002, "text": "hi"})
    assert c.get(k1)["cost_usd"] == 0.002
    assert k1 in c and len(c) == 1


# ───────────────────────────── ledger ──────────────────────────────────────


def _rec(cid, cfg_hash, error=None):
    return CallRecord(call_id=cid, run_id="r", timestamp="2026-06-13T00:00:00+00:00",
                      provider="openai", model_alias="m", model_version="m-2026",
                      temperature=0.0, git_sha="sha", config_hash=cfg_hash,
                      input_tokens=10, output_tokens=5, cost_usd=0.001, error=error)


def test_ledger_append_read_resume(run_root):
    rd = L.ensure_run_dir(run_root, "run1")
    with L.Ledger(L.ledger_path(rd)) as lg:
        lg.append(_rec("c_a", "cfg"))
        lg.append(_rec("c_b", "cfg"))
        lg.append(_rec("c_err", "cfg", error="rate limit"))
    assert len(L.read(L.ledger_path(rd))) == 3
    done = L.completed_call_ids(L.ledger_path(rd))
    assert done == {"c_a", "c_b"}  # errored call is NOT skipped on resume


def test_ledger_verify_catches_problems(run_root):
    from latest.provenance import build_manifest

    rd = L.ensure_run_dir(run_root, "run2")
    man = build_manifest(run_id="run2")
    with L.Ledger(L.ledger_path(rd)) as lg:
        lg.append(_rec("c_a", man.config_hash))
        lg.append(_rec("c_a", man.config_hash))  # duplicate successful call_id
        lg.append(CallRecord(call_id="c_x", run_id="r", timestamp="t"))  # missing provenance
    probs = L.verify(L.ledger_path(rd), manifest=man)
    errors = [m for s, m in probs if s == "error"]
    warns = [m for s, m in probs if s == "warn"]
    assert any("duplicate" in m for m in errors)
    assert any("provenance" in m for m in warns)


# ───────────────────────────── router ──────────────────────────────────────


def test_router_resolves_families():
    assert P.resolve("gpt-5.4-nano") == ("openai", "gpt-5.4-nano")
    assert P.resolve("claude-haiku-4-5") == ("anthropic", "claude-haiku-4-5")
    assert P.resolve("gemini-2.5-pro") == ("gemini", "gemini-2.5-pro")
    assert P.resolve("openai/gpt-oss-120b") == ("groq", "openai/gpt-oss-120b")
    assert P.resolve("openrouter:qwen/q") == ("openrouter", "qwen/q")


def test_router_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        P.resolve("mystery-model")


def test_all_providers_registered():
    assert set(P.registered_providers()) == {"openai", "anthropic", "gemini", "groq", "openrouter"}


def test_estimate_cost_from_pricing_yaml():
    # nano: 1000*0.20/1e6 + 500*1.25/1e6 = 0.000825
    assert abs(P.estimate_cost("openai:gpt-5.4-nano", 1000, 500) - 0.000825) < 1e-9


# ───────────────────── base Conversation (fake provider) ───────────────────


class _FakeConversation(Conversation):
    provider = "fake"
    calls = 0

    def _raw_send(self, transcript, schema, max_tokens, temperature):
        type(self).calls += 1
        ans = ReasonedAnswer(letter="2", confidence="4",
                             acknowledged_counterargument=(len(transcript) > 1),
                             reasoning="fake reasoning for this turn")
        return ProviderResponse(provider="fake", model=self.model, model_version="fake-v1",
                                temperature=temperature, input_tokens=10, output_tokens=5,
                                cost_usd=0.001, latency_ms=1.0, answer=ans, raw=ans.model_dump())


def test_conversation_multiturn_then_cache_replay(run_root):
    """Ask a question, then a follow-up; replay in a fresh run hits the cache."""
    _FakeConversation.calls = 0
    cache = Cache(L.cache_dir(run_root))
    rd = L.ensure_run_dir(run_root, "run1")
    ctx = CallContext(run_id="run1", trial_id="t1", module="manipulation",
                      git_sha="sha", config_hash="cfg")

    with L.Ledger(L.ledger_path(rd)) as lg:
        conv = _FakeConversation("fake-model", ctx=ctx, cache=cache, ledger=lg)
        r1 = conv.send("What is the answer?", ReasonedAnswer)
        r2 = conv.send("I disagree, it's 3.", ReasonedAnswer)

    assert _FakeConversation.calls == 2          # two real calls
    assert len(conv.transcript) == 4             # user, assistant, user, assistant
    assert r1.raw["letter"] == "2" and r2.raw["letter"] == "2"
    assert len(L.read(L.ledger_path(rd))) == 2   # two ledger lines

    # Replay the SAME two turns in a fresh run -> all cache hits, no new calls.
    rd2 = L.ensure_run_dir(run_root, "run2")
    with L.Ledger(L.ledger_path(rd2)) as lg2:
        conv2 = _FakeConversation("fake-model",
                                  ctx=CallContext(run_id="run2", git_sha="sha", config_hash="cfg"),
                                  cache=cache, ledger=lg2)
        conv2.send("What is the answer?", ReasonedAnswer)
        conv2.send("I disagree, it's 3.", ReasonedAnswer)

    assert _FakeConversation.calls == 2          # STILL 2 -> replay was fully cached
    recs2 = L.read(L.ledger_path(rd2))
    assert all(r.cache_hit for r in recs2)


# ───────────────────────────── config ──────────────────────────────────────


def test_config_loads_and_flattens():
    cfg = loader.load_config()
    models = loader.models_from_config(cfg)
    assert "openai:gpt-5.4-nano" in models and len(models) == 6
    assert loader.judges_from_config(cfg)  # at least one judge
    # validate() runs and returns a list (warnings expected for unpinned datasets)
    assert isinstance(loader.validate(cfg), list)
