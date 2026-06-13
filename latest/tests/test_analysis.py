"""Offline analysis tests on a synthetic ledger (no network).

Covers scoring + aggregation across all modules and guards the multi-model
grouping bug (manipulation trials run once per model share a trial_id).
"""
from __future__ import annotations

import pytest

from latest.analysis import aggregate as A
from latest.analysis.report import _markdown
from latest.analysis.score import score_all
from latest.records import CallRecord, RunManifest, Trial, make_trial_id

MODELS = ["claude-haiku-4-5", "gpt-5.4-nano"]
JUDGES = ["gpt-5.4-mini", "claude-sonnet-4-6"]


def _build():
    trials, records = [], []

    def cr(**k):
        base = dict(call_id="c" + str(len(records)), run_id="r", timestamp="t", temperature=0.0,
                    git_sha="sha", config_hash="cfg", input_tokens=5, output_tokens=3, cost_usd=0.001)
        base.update(k)
        records.append(CallRecord(**base))

    def manip(attack, arm="pressure_wrong"):
        c = dict(module="manipulation", item_id="q1", mode="stateless", attack=attack, variant_idx=0, replicate_idx=0)
        t = Trial(trial_id=make_trial_id(c), module="manipulation", dataset="mmlu", item_id="q1",
                  subject="formal_logic", question="Q", choices=["a", "b", "c", "d"], correct_answer="2",
                  arm=arm, attack=attack, variant_idx=0, mode="stateless", offered_answer="3")
        trials.append(t)
        return t

    for attack, haiku_post, nano_post in [("confident_assertion", "2", "3"), ("authority_pressure", "3", "3")]:
        t = manip(attack)
        for model, post in [("claude-haiku-4-5", haiku_post), ("gpt-5.4-nano", nano_post)]:
            cr(trial_id=t.trial_id, module="manipulation", model_alias=model, role="subject", turn_index=0,
               condition="pressure_wrong", answer_raw={"letter": "2"})
            cr(trial_id=t.trial_id, module="manipulation", model_alias=model, role="subject", turn_index=1,
               condition="pressure_wrong", attack=attack, answer_raw={"letter": post})
    ct = manip("neutral_control", arm="control")
    for model in MODELS:
        cr(trial_id=ct.trial_id, module="manipulation", model_alias=model, role="subject", turn_index=0,
           condition="control", answer_raw={"letter": "2"})
        cr(trial_id=ct.trial_id, module="manipulation", model_alias=model, role="subject", turn_index=1,
           condition="control", attack="neutral_control", answer_raw={"letter": "2"})

    bt = Trial(trial_id=make_trial_id({"module": "benchmark", "dataset": "mmlu", "item_id": "m1"}),
               module="benchmark", dataset="mmlu", item_id="m1", question="Q",
               choices=["a", "b", "c", "d"], correct_answer="2")
    trials.append(bt)
    cr(trial_id=bt.trial_id, module="benchmark", dataset="mmlu", model_alias="claude-haiku-4-5",
       role="subject", condition="benchmark", answer_raw={"answer": "2"})
    cr(trial_id=bt.trial_id, module="benchmark", dataset="mmlu", model_alias="gpt-5.4-nano",
       role="subject", condition="benchmark", answer_raw={"answer": "4"})

    mt = Trial(trial_id=make_trial_id({"module": "moral", "dataset": "moral", "item_id": "crisis_01"}),
               module="moral", dataset="moral", item_id="crisis_01", subject="crisis", question="help")
    trials.append(mt)
    for model in MODELS:
        cr(trial_id=mt.trial_id, module="moral", model_alias=model, role="subject", condition="moral_answer", text="reply")
        for judge in JUDGES:
            for axis, rating in [("empathy", "5"), ("safety", "4"), ("helpfulness", "5")]:
                cr(trial_id=mt.trial_id, module="moral", model_alias=judge, role="judge",
                   condition=axis, judged_model=model, text=rating)

    return trials, records


def test_multimodel_manipulation_scoring():
    trials, records = _build()
    scores = score_all(trials, records)
    overall = A.stateless_overall(scores)
    assert set(overall) == set(MODELS)  # BOTH models scored (the grouping-bug guard)
    assert overall["claude-haiku-4-5"]["resist"] == 0.5  # resisted confident, folded authority
    assert overall["gpt-5.4-nano"]["resist"] == 0.0      # folded both


def test_mcnemar_fires_with_two_models():
    trials, records = _build()
    scores = score_all(trials, records)
    mc = A.mcnemar_pairwise(scores)
    assert mc and mc[0]["n_pairs"] == 2
    assert mc[0]["a_better"] + mc[0]["b_better"] >= 1  # at least one discordant pair


def test_benchmark_and_moral_and_judges():
    trials, records = _build()
    scores = score_all(trials, records)
    acc = A.benchmark_accuracy(scores)
    assert acc["claude-haiku-4-5"]["mmlu"]["accuracy"] == 1.0
    assert acc["gpt-5.4-nano"]["mmlu"]["accuracy"] == 0.0
    moral = A.moral_axes(scores)
    assert moral["claude-haiku-4-5"]["axes"]["empathy"] == 5.0
    jr = A.judge_reliability(records)
    assert jr and jr["agreement"] == 1.0 and jr["kappa"] == 1.0


def test_report_renders():
    trials, records = _build()
    scores = score_all(trials, records)
    man = RunManifest(run_id="t", created_at="t", models=MODELS, seed=42)
    agg = {
        "stateless_overall": A.stateless_overall(scores), "stateless_by_attack": A.stateless_by_attack(scores),
        "stateless_adjusted": A.stateless_adjusted(scores), "natural_drift": A.natural_drift(scores),
        "drift": A.drift_summary(scores), "stateful": A.stateful_summary(scores),
        "repeat": A.repeat_summary(scores), "gauntlet": A.gauntlet_summary(scores),
        "mcnemar": A.mcnemar_pairwise(scores), "benchmark_accuracy": A.benchmark_accuracy(scores),
        "moral_axes": A.moral_axes(scores), "judge_reliability": A.judge_reliability(records),
        "cost": A.cost_summary(records),
    }
    md = _markdown(man, agg)
    assert "latest manipulation report" in md and "Judge reliability" in md
