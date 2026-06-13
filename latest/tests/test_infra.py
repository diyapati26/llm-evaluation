"""Offline tests for operational infra: budget classification + run-log."""
from __future__ import annotations

import json

from latest.collect.engine import _is_budget_error
from latest.runlog import RunLog


def test_budget_error_classifier():
    assert _is_budget_error(Exception("exceeded its monthly spending cap"))
    assert _is_budget_error(Exception("billing hard limit reached"))
    assert not _is_budget_error(Exception("rate limit; please retry"))
    assert not _is_budget_error(Exception("500 internal server error"))


def test_runlog_appends_and_fsyncs(tmp_path):
    p = tmp_path / "run.log.jsonl"
    with RunLog(p, echo=False) as log:
        log.log("run_begin", run_id="x", models=["a", "b"])
        log.log("trial_error", level="warn", trial_id="t1", error="boom")
        log.log("run_complete")
    lines = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["event"] for r in lines] == ["run_begin", "trial_error", "run_complete"]
    assert all("ts" in r and "level" in r for r in lines)
    assert lines[1]["level"] == "warn" and lines[1]["error"] == "boom"
