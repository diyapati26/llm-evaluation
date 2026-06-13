"""collect/benchmark.py — execute one standard-benchmark Trial.

MC datasets (mmlu / hellaswag / truthfulqa_mc): one structured call via the
Conversation (cached + logged). truthfulqa_gen: a free-form answer + LLM-judge
calls (truthful + informative), all logged to the ledger.
"""
from __future__ import annotations

from latest.collect import judge
from latest.collect._calls import cached_chat
from latest.providers import start_conversation
from latest.providers.base import CallContext
from latest.records import (
    HellaSwag_Answer,
    MMLU_Answer,
    TruthfulQA_MC_Answer,
)

_MC_SCHEMA = {"mmlu": MMLU_Answer, "hellaswag": HellaSwag_Answer, "truthfulqa_mc": TruthfulQA_MC_Answer}


def _ctx(trial, manifest, snapshot) -> CallContext:
    return CallContext(
        run_id=manifest.run_id, trial_id=trial.trial_id, module="benchmark",
        item_id=trial.item_id, subject=trial.subject, dataset=trial.dataset, condition="benchmark",
        seed=manifest.seed, model_id_for_key=snapshot, git_sha=manifest.git_sha,
        config_hash=manifest.config_hash, role="subject",
    )


def run_trial(model, trial, cache, ledger, manifest, snapshot) -> None:
    ctx = _ctx(trial, manifest, snapshot)
    if trial.dataset in _MC_SCHEMA:
        conv = start_conversation(model, ctx=ctx, cache=cache, ledger=ledger)
        conv.send(trial.question, _MC_SCHEMA[trial.dataset])
    elif trial.dataset == "truthfulqa_gen":
        resp = cached_chat(model, [{"role": "user", "content": trial.question}], ctx=ctx, cache=cache,
                           ledger=ledger, role="subject", condition="gen_answer",
                           schema_name="gen_answer", max_tokens=300, turn_index=0)
        judge.run_truthfulqa_judges(trial, resp.text or "", model, manifest.judges, ctx, cache, ledger)
    else:
        raise ValueError(f"unknown benchmark dataset '{trial.dataset}'")
