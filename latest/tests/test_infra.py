"""Offline tests for operational infra: budget classification + run-log."""
from __future__ import annotations

import json

from latest.collect.engine import _is_budget_error
from latest.providers.retry import _is_retryable
from latest.runlog import RunLog


def test_budget_error_classifier():
    # Genuine quota/billing walls -> budget (and therefore not retried).
    assert _is_budget_error(Exception("Error code: insufficient_quota"))
    assert _is_budget_error(Exception("billing hard limit reached"))
    assert _is_budget_error(Exception("Your credit balance is too low"))
    assert _is_budget_error(Exception("You exceeded your current quota"))
    # Transient rate limits are NOT budget, even when they mention 'monthly'.
    monthly_429 = Exception("Rate limit reached. You've hit your monthly request limit, retry shortly.")
    assert not _is_budget_error(monthly_429)
    assert _is_retryable(monthly_429)  # and it stays retryable
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
