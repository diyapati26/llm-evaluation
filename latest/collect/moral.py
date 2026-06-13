"""collect/moral.py — execute one moral/empathy Trial.

The subject model answers the scenario free-form (one call), then each configured
judge scores that answer on the category's axes (LLM-as-judge calls). All logged
to the ledger; the integer ratings are parsed in analysis.
"""
from __future__ import annotations

from latest.collect import judge
from latest.collect._calls import cached_chat
from latest.providers.base import CallContext
from latest.providers.router import resolve


def run_trial(model, trial, cache, ledger, manifest, snapshot) -> None:
    alias = resolve(model)[1]  # bare model id; matches CallRecord.model_alias for the join
    ctx = CallContext(
        run_id=manifest.run_id, trial_id=trial.trial_id, module="moral",
        item_id=trial.item_id, subject=trial.subject, dataset="moral",
        seed=manifest.seed, model_id_for_key=snapshot, git_sha=manifest.git_sha,
        config_hash=manifest.config_hash, role="subject",
    )
    resp = cached_chat(model, [{"role": "user", "content": trial.question}], ctx=ctx, cache=cache,
                       ledger=ledger, role="subject", condition="moral_answer",
                       schema_name="moral_answer", max_tokens=600, turn_index=0)
    judge.run_moral_judges(trial, resp.text or "", alias, manifest.judges, ctx, cache, ledger)
