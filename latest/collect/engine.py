"""collect/engine.py — fan a per-trial executor across models into the ledger.

Writes the manifest FIRST, opens the shared cache + the append-only ledger, then
runs each pending (model, trial) with bounded concurrency. Resume is per
(model, trial_id): a `progress.jsonl` records completed pairs so a re-run skips
them (and the content-addressed cache makes any re-issued call free anyway).
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from latest import ledger as L
from latest.cache import Cache
from latest.config.loader import snapshot_for
from latest.provenance import build_manifest
from latest.providers.retry import _HARD_FAIL_MARKERS


def _is_budget_error(exc: BaseException) -> bool:
    """A spend cap / billing / quota wall — retrying or continuing is pointless."""
    return any(m in str(exc).lower() for m in _HARD_FAIL_MARKERS)


class Progress:
    """Crash-safe set of completed (model, trial_id) pairs, backed by progress.jsonl."""

    def __init__(self, rd: Path):
        self.path = rd / "progress.jsonl"
        self._lock = threading.Lock()
        self._done: set[tuple[str, str]] = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        self._done.add((d["model"], d["trial_id"]))

    def is_done(self, model: str, trial_id: str) -> bool:
        return (model, trial_id) in self._done

    def mark(self, model: str, trial_id: str) -> None:
        with self._lock:
            self._done.add((model, trial_id))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"model": model, "trial_id": trial_id}) + "\n")
                f.flush()


def _prompt_budget(model: str) -> bool:
    """Interactive budget gate. Returns True to resume (user topped up), False to stop+save.

    Falls back to stop+save when there is no interactive terminal (EOFError), so an
    unattended run never hangs on the prompt.
    """
    try:
        ans = input(
            f"\n*** Budget/quota exhausted for '{model}'.\n"
            f"    Everything so far is saved. Add more budget, then type 'y' to RESUME "
            f"(re-runs only the unfinished trials, cached ones are free), or 'n' to STOP and save as-is: [y/N] "
        )
        return ans.strip().lower().startswith("y")
    except EOFError:
        return False


def collect(trials, models, *, run_id, results_root, cfg, concurrency, run_trial_fn,
            runlog=None, on_budget=None):
    """Execute `trials` for each model into the ledger, with constant saves.

    run_trial_fn(model, trial, cache, ledger, manifest, snapshot).
    on_budget(model, error) -> bool: resume (True) or stop+save (False). Defaults
    to an interactive prompt (stop+save when non-interactive).
    """
    rd = L.ensure_run_dir(results_root, run_id)
    manifest = build_manifest(cfg, run_id)
    manifest.models = list(models)  # reflect the actual (possibly overridden) model set
    L.write_manifest(manifest, rd)

    cache = Cache(L.cache_dir(results_root))
    progress = Progress(rd)
    on_budget = on_budget or _prompt_budget

    own_log = runlog is None
    if own_log:
        from latest.runlog import RunLog
        runlog = RunLog(L.run_dir(results_root, run_id) / "run.log.jsonl")

    halted = False
    try:
        with L.Ledger(L.ledger_path(rd)) as lg:
            for model in models:
                snapshot = snapshot_for(model)
                # Retry loop: on a budget wall we pause, ask, and (on resume) re-run
                # only the still-unfinished trials — completed ones are skipped via progress.
                while True:
                    pending = [t for t in trials if not progress.is_done(model, t.trial_id)]
                    runlog.log("model_start", model=model, pending=len(pending), total=len(trials))
                    if not pending:
                        break
                    done = failed = 0
                    budget_hit = False
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futs = {pool.submit(run_trial_fn, model, t, cache, lg, manifest, snapshot): t
                                for t in pending}
                        for f in tqdm(as_completed(futs), total=len(futs), desc=model):
                            trial = futs[f]
                            try:
                                f.result()
                                progress.mark(model, trial.trial_id)
                                done += 1
                            except Exception as e:  # noqa: BLE001
                                failed += 1
                                if _is_budget_error(e):
                                    runlog.log("budget_exhausted", level="error", model=model, error=str(e)[:200])
                                    budget_hit = True
                                    for pf in futs:
                                        pf.cancel()
                                    break
                                runlog.log("trial_error", level="warn", model=model, trial_id=trial.trial_id,
                                           mode=trial.mode, attack=trial.attack, error=str(e)[:200])
                    runlog.log("model_progress", model=model, done=done, failed=failed)
                    if not budget_hit:
                        break
                    resume = bool(on_budget(model, "budget/quota exhausted"))
                    runlog.log("budget_decision", model=model, resume=resume)
                    if not resume:
                        halted = True
                        break
                if halted:
                    runlog.log("run_halted", level="error",
                               note="stopped on budget; all completed work saved")
                    break
    finally:
        runlog.log("collect_done", halted=halted)
        if own_log:
            runlog.close()

    return rd, manifest
