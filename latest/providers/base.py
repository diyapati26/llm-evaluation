"""base.py — the Conversation ABC with cache + ledger wired into .send().

Design decision: ALL providers use client-side history (the full transcript is
re-sent each turn), even OpenAI. The old Simpler Arch used OpenAI's server-side
conversation state as a special case, but that breaks content-addressed caching
(a cache hit on turn 2 would leave the server's conversation state un-advanced
for a turn-3 miss). A uniform client-side transcript makes the cache key — and
therefore caching and resume — correct and identical for every provider, at the
cost of re-sending history (which is free on a cache hit anyway).

send() is a template method:
    1. append the user turn to the transcript
    2. content-address the request (full transcript) -> call_id (== cache key)
    3. cache hit  -> reconstruct the response, no network call
       cache miss -> subclass._raw_send() does the real API call, then cache.put
    4. append the assistant turn (stable JSON) to the transcript
    5. accumulate cost/tokens, build + append a CallRecord to the ledger
    6. return the ProviderResponse

Subclasses implement ._raw_send(transcript, schema, max_tokens, temperature)
and need to know nothing about caching, the ledger, or provenance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from latest.cache import call_key
from latest.provenance import utc_now_iso
from latest.records import (
    CallRecord,
    ProviderResponse,
    canonical_json,
    sha256_hex,
)


@dataclass
class CallContext:
    """Per-conversation context the base needs to write rich CallRecords.

    The collector builds one of these per trial. None-friendly: an unset context
    still produces a valid (if sparsely-linked) ledger line.
    """

    run_id: str | None = None
    trial_id: str | None = None
    module: str | None = None
    item_id: str | None = None
    subject: str | None = None
    dataset: str | None = None
    arm: str | None = None
    attack: str | None = None
    variant_idx: int | None = None
    replicate_idx: int | None = None
    condition: str | None = None
    role: str = "subject"
    seed: int | None = None
    max_tokens: int | None = None
    model_id_for_key: str | None = None  # locked snapshot if available, else alias
    git_sha: str | None = None
    config_hash: str | None = None


class Conversation:
    """Base multi-turn conversation. Tracks transcript + cumulative accounting.

    Subclasses MUST implement `_raw_send`. They MUST NOT touch the cache or
    ledger — the base owns both so behavior is identical across providers.
    """

    provider = "base"

    def __init__(
        self,
        model: str,
        ctx: CallContext | None = None,
        cache: Any = None,
        ledger: Any = None,
    ):
        self.model = model
        self.ctx = ctx or CallContext()
        self.cache = cache
        self.ledger = ledger

        self.transcript: list[dict] = []
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency_ms = 0.0
        self.turn_count = 0
        self.records: list[CallRecord] = []
        self.last_record: CallRecord | None = None

    # ── subclass hook ──────────────────────────────────────────────────────
    def _raw_send(self, transcript: list[dict], schema, max_tokens, temperature) -> ProviderResponse:
        raise NotImplementedError

    # ── accounting ───────────────────────────────────────────────────────--
    def _accumulate(self, resp: ProviderResponse) -> None:
        self.total_cost += resp.cost_usd
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_latency_ms += resp.latency_ms

    # ── the template method ─────────────────────────────────────────────────
    def send(self, message: str, schema, max_tokens: int | None = None, temperature: float = 0.0) -> ProviderResponse:
        self.transcript.append({"role": "user", "content": message})
        turn_index = self.turn_count
        self.turn_count += 1

        model_key = self.ctx.model_id_for_key or self.model
        key = call_key(
            self.provider, model_key, temperature, schema.__name__,
            self.transcript, max_tokens, self.ctx.seed,
        )

        cached = self.cache.get(key) if self.cache is not None else None
        if cached is not None:
            resp = ProviderResponse(
                provider=cached.get("provider", self.provider),
                model=cached.get("model", self.model),
                model_version=cached.get("model_version"),
                temperature=cached.get("temperature", temperature),
                input_tokens=cached.get("input_tokens", 0),
                output_tokens=cached.get("output_tokens", 0),
                cost_usd=cached.get("cost_usd", 0.0),
                latency_ms=cached.get("latency_ms", 0.0),
                raw=cached.get("raw"),
                text=cached.get("text"),
                answer=None,
            )
            cache_hit = True
        else:
            resp = self._raw_send(self.transcript, schema, max_tokens, temperature)
            cache_hit = False
            if self.cache is not None:
                self.cache.put(key, {
                    "provider": resp.provider, "model": resp.model,
                    "model_version": resp.model_version, "temperature": resp.temperature,
                    "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd, "latency_ms": resp.latency_ms,
                    "raw": resp.raw, "text": resp.text,
                })

        # Feed the parsed answer back as the assistant turn (stable JSON) so the
        # next turn's transcript — and therefore its cache key — is deterministic.
        assistant_content = (
            json.dumps(resp.raw, sort_keys=True) if resp.raw is not None else (resp.text or "")
        )
        self.transcript.append({"role": "assistant", "content": assistant_content})

        self._accumulate(resp)
        record = self._build_record(key, message, resp, turn_index, cache_hit)
        self.records.append(record)
        self.last_record = record
        if self.ledger is not None:
            self.ledger.append(record)
        return resp

    # ── ledger line builder ──────────────────────────────────────────────--
    def _build_record(self, call_id, prompt, resp: ProviderResponse, turn_index, cache_hit) -> CallRecord:
        ctx = self.ctx
        return CallRecord(
            call_id=call_id,
            run_id=ctx.run_id or "",
            timestamp=utc_now_iso(),
            trial_id=ctx.trial_id,
            module=ctx.module,
            item_id=ctx.item_id,
            subject=ctx.subject,
            dataset=ctx.dataset,
            provider=resp.provider,
            model_alias=self.model,
            model_version=resp.model_version,
            temperature=resp.temperature,
            seed=ctx.seed,
            max_tokens=ctx.max_tokens,
            role=ctx.role,
            condition=ctx.condition or ctx.arm,
            attack=ctx.attack,
            variant_idx=ctx.variant_idx,
            replicate_idx=ctx.replicate_idx,
            turn_index=turn_index,
            messages_hash=sha256_hex(canonical_json(self.transcript)),
            prompt=prompt,
            answer_raw=resp.raw,
            text=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            latency_ms=resp.latency_ms,
            cache_hit=cache_hit,
            error=None,
            git_sha=ctx.git_sha,
            config_hash=ctx.config_hash,
        )
