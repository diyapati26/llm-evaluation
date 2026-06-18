"""collect/_calls.py — a cached, ledger-logged single-turn free-form call.

The Conversation handles cached multi-turn STRUCTURED calls; this is its
free-form sibling for moral/gen answers and all LLM-judge calls. Same contract:
content-address the request, hit the cache if present, else call the provider
and populate the cache, and append exactly one CallRecord either way.
"""
from __future__ import annotations

from latest.cache import call_key
from latest.providers import router
from latest.provenance import utc_now_iso
from latest.records import CallRecord, ProviderResponse


def cached_chat(model, messages, *, ctx, cache, ledger, role="subject", condition=None,
                judged_model=None, attack=None, schema_name="freeform",
                max_tokens=None, temperature=0.0, turn_index=0) -> ProviderResponse:
    provider, real = router.resolve(model)
    # Judge calls pass their own model; never reuse the subject's locked snapshot as the key id.
    model_key = real if role == "judge" else ((ctx.model_id_for_key if ctx else None) or real)
    key = call_key(provider, model_key, temperature, schema_name, messages, max_tokens,
                   getattr(ctx, "seed", None))

    cached = cache.get(key) if cache is not None else None
    if cached is not None:
        resp = ProviderResponse(
            provider=cached.get("provider", provider), model=cached.get("model", real),
            model_version=cached.get("model_version"), temperature=cached.get("temperature", temperature),
            input_tokens=cached.get("input_tokens", 0), output_tokens=cached.get("output_tokens", 0),
            cost_usd=cached.get("cost_usd", 0.0), latency_ms=cached.get("latency_ms", 0.0),
            text=cached.get("text"), raw=cached.get("raw"),
        )
        cache_hit = True
    else:
        resp = router.chat(model, messages, max_tokens=max_tokens, temperature=temperature)
        cache_hit = False
        if cache is not None:
            cache.put(key, {
                "provider": resp.provider, "model": resp.model, "model_version": resp.model_version,
                "temperature": resp.temperature, "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens, "cost_usd": resp.cost_usd,
                "latency_ms": resp.latency_ms, "text": resp.text, "raw": resp.raw,
            })

    rec = CallRecord(
        call_id=key, run_id=getattr(ctx, "run_id", "") or "", timestamp=utc_now_iso(),
        trial_id=getattr(ctx, "trial_id", None), module=getattr(ctx, "module", None),
        item_id=getattr(ctx, "item_id", None), subject=getattr(ctx, "subject", None),
        dataset=getattr(ctx, "dataset", None), provider=resp.provider, model_alias=real,
        model_version=resp.model_version, temperature=resp.temperature, seed=getattr(ctx, "seed", None),
        max_tokens=max_tokens, role=role, condition=condition, attack=attack, judged_model=judged_model,
        turn_index=turn_index, prompt=(messages[-1]["content"] if messages else None),
        text=resp.text, answer_raw=resp.raw, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
        cost_usd=resp.cost_usd, latency_ms=resp.latency_ms, cache_hit=cache_hit,
        git_sha=getattr(ctx, "git_sha", None), config_hash=getattr(ctx, "config_hash", None),
    )
    if ledger is not None:
        ledger.append(rec)
    return resp
