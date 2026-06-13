"""collect/ — execute a frozen design across models into the append-only ledger.

engine.py owns the fan-out (bounded concurrency), the manifest write, the shared
cache, and resume (skip already-completed (model, trial)). manipulation.py turns
a manipulation Trial into the right send() sequence per mode. Collection produces
CallRecords ONLY — never scores.
"""
