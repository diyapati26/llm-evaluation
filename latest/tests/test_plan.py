"""Offline tests for the design-matrix layer (no network)."""
from __future__ import annotations

from collections import Counter

from latest import loaders
from latest.plan import design, freeze
from latest.plan.arms import PRESSURE_ATTACKS

_ITEMS = [
    {"item_id": f"mmlu/formal_logic/{i}", "dataset": "mmlu", "subject": "formal_logic",
     "question": f"Question {i}?", "choices": ["alpha", "beta", "gamma", "delta"],
     "correct_answer": "2", "refs": None}
    for i in range(2)
]


def _manip():
    return design.build_manipulation_trials(
        _ITEMS, loaders.load_attacks(),
        modes=["stateless", "stateful"], include_drift=True, resamples=1, seed=42,
    )


def test_manipulation_expansion_counts():
    mt = _manip()
    # per item: (7 pressure + 1 control) x 4 variants = 32 stateless + 1 drift + 1 stateful = 34; x2 = 68
    assert len(mt) == 68
    by_mode = Counter(t.mode for t in mt)
    assert by_mode == {"stateless": 64, "drift": 2, "stateful": 2}


def test_arms_and_offered_answers():
    mt = _manip()
    control = [t for t in mt if t.arm == "control"]
    pressure = [t for t in mt if t.arm == "pressure_wrong"]
    assert control and all(t.offered_answer is None for t in control)
    assert all(t.offered_answer is not None for t in pressure)
    assert all(t.offered_answer != t.correct_answer for t in pressure)


def test_stateful_order_counterbalanced():
    sf = [t for t in _manip() if t.mode == "stateful"]
    assert all(sorted(t.stateful_order) == sorted(PRESSURE_ATTACKS) for t in sf)
    assert sf[0].stateful_order != sf[1].stateful_order  # different per item


def test_design_is_deterministic():
    assert [t.trial_id for t in _manip()] == [t.trial_id for t in _manip()]


def test_freeze_roundtrip(run_root):
    from latest import ledger as L

    trials = _manip() + design.build_moral_trials(loaders.load_moral_scenarios())
    rd = L.ensure_run_dir(run_root, "plan")
    _, manifest = freeze.freeze(trials, rd, seed=42)
    assert manifest["n_trials"] == len(trials)
    reloaded = freeze.read_trials(rd)
    assert sorted(t.trial_id for t in reloaded) == sorted(t.trial_id for t in trials)
    # nested columns survive the round trip
    rsf = next(t for t in reloaded if t.mode == "stateful")
    assert sorted(rsf.stateful_order) == sorted(PRESSURE_ATTACKS)
    rmoral = next(t for t in reloaded if t.module == "moral")
    assert isinstance(rmoral.metadata, dict) and rmoral.metadata
