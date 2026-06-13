"""runlog.py — structured, append-only JSONL operational log.

Separate from the data ledger: the ledger records every API CALL (the facts);
the run-log records EVENTS (preflight results, stage transitions, per-model
progress, trial errors, budget stops, completion). Every event is appended and
fsync'd immediately and echoed to stdout, so nothing is lost on a crash and a
run is auditable after the fact. One file per run: runs/<run_id>/run.log.jsonl.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from latest.provenance import utc_now_iso


class RunLog:
    def __init__(self, path: str | Path, echo: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8", newline="")

    def log(self, event: str, level: str = "info", **fields) -> dict:
        rec = {"ts": utc_now_iso(), "level": level, "event": event, **fields}
        line = json.dumps(rec, default=str, ensure_ascii=True) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            os.fsync(self._fh.fileno())
        if self.echo:
            extra = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"  [{level}] {event} {extra}".rstrip())
        return rec

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
