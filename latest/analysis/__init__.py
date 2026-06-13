"""analysis/ — PURE, network-free interpretation of the ledger.

Reads trials.parquet (design) + ledger.jsonl (facts) and produces ScoreRecords,
aggregate tables (per-attack rates, natural-drift correction, stateful
killing-blow, repeat/gauntlet endurance), confidence intervals, McNemar tests,
and a Markdown report. No API keys, no network — fully reproducible from the
ledger and unit-testable on fixtures.
"""
