"""collect/run.py — per-trial dispatcher by module (the engine's run_trial_fn)."""
from __future__ import annotations

from latest.collect import benchmark, manipulation, moral

_DISPATCH = {
    "manipulation": manipulation.run_trial,
    "benchmark": benchmark.run_trial,
    "moral": moral.run_trial,
}


def run_trial(model, trial, cache, ledger, manifest, snapshot) -> None:
    fn = _DISPATCH.get(trial.module)
    if fn is None:
        raise ValueError(f"no collector for module '{trial.module}'")
    fn(model, trial, cache, ledger, manifest, snapshot)
