"""ledger.py — the append-only run record + run-directory layout.

The ledger (`runs/<run_id>/ledger.jsonl`) is the single source of truth for a
run: one CallRecord per line, one line per API call that actually happened.
Append-only and fsync'd, so a crash/kill/hang loses at most the one call
currently in flight — never the whole run.

Responsibilities:
  - Run-directory layout (manifest / trials / ledger / scored / report) +
    the SHARED cross-run cache directory.
  - Manifest read/write (the manifest is written FIRST).
  - Ledger append (thread-safe, fsync'd) and read.
  - Resume: completed_call_ids() -> the set of call_ids already done.
  - verify(): integrity + provenance gate for the analysis layer.

Depends on records (+ provenance for read_manifest/write_manifest timestamps).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from latest.records import CallRecord, RunManifest


def _turn_key(d: dict) -> tuple:
    """Logical-turn identity of a CallRecord (as a dict). call_id is a CACHE address
    shared across trials with identical requests, so identity is the turn's
    coordinates, not call_id."""
    return (d.get("model_alias"), d.get("trial_id"), d.get("turn_index"),
            d.get("judged_model"), d.get("role"), d.get("condition"))


# ───────────────────────── run-directory layout ────────────────────────────


def run_dir(results_root: str | Path, run_id: str) -> Path:
    return Path(results_root) / run_id


def manifest_path(rd: Path) -> Path:
    return rd / "manifest.json"


def ledger_path(rd: Path) -> Path:
    return rd / "ledger.jsonl"


def trials_path(rd: Path) -> Path:
    return rd / "trials.parquet"


def trials_manifest_path(rd: Path) -> Path:
    return rd / "trials.manifest.json"


def scored_path(rd: Path) -> Path:
    return rd / "scored.parquet"


def report_dir(rd: Path) -> Path:
    return rd / "report"


def cache_dir(results_root: str | Path) -> Path:
    """SHARED cache, one level above run dirs, so repeats across runs are free."""
    return Path(results_root) / "cache"


def ensure_run_dir(results_root: str | Path, run_id: str) -> Path:
    rd = run_dir(results_root, run_id)
    rd.mkdir(parents=True, exist_ok=True)
    return rd


# ───────────────────────────── manifest io ─────────────────────────────────


def write_manifest(manifest: RunManifest, rd: Path) -> Path:
    rd.mkdir(parents=True, exist_ok=True)
    p = manifest_path(rd)
    p.write_text(json.dumps(manifest.model_dump(), indent=2, default=str), encoding="utf-8")
    return p


def read_manifest(rd_or_path: Path) -> RunManifest:
    p = Path(rd_or_path)
    if p.is_dir():
        p = manifest_path(p)
    return RunManifest(**json.loads(p.read_text(encoding="utf-8")))


# ───────────────────────────── ledger writer ───────────────────────────────


class Ledger:
    """Append-only, fsync'd CallRecord writer. Thread-safe.

    Use as a context manager so the file handle is always closed:
        with Ledger(ledger_path(rd)) as lg:
            lg.append(record)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Idempotency set: logical-turn keys of successful rows already on disk, so a
        # resume that re-runs a (model, trial) can't append duplicate turn rows.
        self._seen: set[tuple] = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a torn final line
                    if not d.get("error"):
                        self._seen.add(_turn_key(d))
        self._fh = open(self.path, "a", encoding="utf-8", newline="")

    def append(self, record: CallRecord) -> bool:
        """Append a CallRecord. Idempotent: a successful logical turn already on disk
        is skipped (returns False) so resumes don't duplicate rows. Error rows always
        write (they're informational and don't claim the turn succeeded)."""
        key = _turn_key(record.model_dump())
        line = json.dumps(record.model_dump(), default=str, ensure_ascii=True) + "\n"
        with self._lock:
            if not record.error and key in self._seen:
                return False
            self._fh.write(line)
            self._fh.flush()
            os.fsync(self._fh.fileno())
            if not record.error:
                self._seen.add(key)
            return True

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ───────────────────────────── ledger reader ───────────────────────────────


def read(path: str | Path) -> list[CallRecord]:
    """Parse every line into a CallRecord. Raises ValueError on a corrupt line."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[CallRecord] = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(CallRecord(**json.loads(line)))
            except Exception as e:  # noqa: BLE001 - surface the offending line
                raise ValueError(f"{p}:{i}: corrupt ledger line: {e}") from e
    return out


def completed_call_ids(path: str | Path) -> set[str]:
    """call_ids that completed WITHOUT error — used to skip work on resume.

    Reads tolerantly (a half-written final line from a crash is ignored) so a
    resume never aborts on the very corruption it exists to recover from.
    """
    p = Path(path)
    done: set[str] = set()
    if not p.exists():
        return done
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line
            if d.get("call_id") and not d.get("error"):
                done.add(d["call_id"])
    return done


# ───────────────────────────── verify-ledger ───────────────────────────────

_REQUIRED_PROVENANCE = ("temperature", "git_sha", "config_hash")


def verify(path: str | Path, manifest: RunManifest | None = None) -> list[tuple[str, str]]:
    """Integrity + provenance check. Returns [(severity, message), ...].

    severity in {"error", "warn"}. Empty list == clean. Errors mean the ledger
    should not be used to produce paper numbers; warns are advisories.
    """
    p = Path(path)
    problems: list[tuple[str, str]] = []
    if not p.exists():
        return [("error", f"ledger not found: {p}")]

    # call_id is the CACHE address and is intentionally SHARED across trials that
    # issue an identical request (e.g. every stateless attack on an item re-asks
    # the same turn-0 question). So uniqueness is enforced on the logical turn
    # identity (model, trial_id, turn_index), not on call_id.
    seen_turns: set[tuple] = set()
    n = 0
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(("error", f"line {i}: invalid JSON ({e})"))
                continue
            try:
                rec = CallRecord(**d)
            except Exception as e:  # noqa: BLE001
                problems.append(("error", f"line {i}: not a valid CallRecord ({e})"))
                continue

            cid = rec.call_id
            for field in _REQUIRED_PROVENANCE:
                if getattr(rec, field, None) in (None, ""):
                    problems.append(("warn", f"line {i} {cid}: missing provenance '{field}'"))
            if rec.model_version in (None, "") and not rec.error:
                problems.append(("warn", f"line {i} {cid}: model_version not recorded"))

            if not rec.error:
                # Logical-turn identity (incl. role + condition/axis + judged_model),
                # so distinct judge axes/roles are never conflated.
                turn_key = _turn_key(rec.model_dump())
                if turn_key in seen_turns:
                    problems.append(("error", f"line {i}: duplicate turn {turn_key}"))
                seen_turns.add(turn_key)

            if manifest is not None:
                if rec.config_hash and manifest.config_hash and rec.config_hash != manifest.config_hash:
                    problems.append(("error", f"line {i} {cid}: config_hash != manifest"))
                if rec.git_sha and manifest.git_sha and rec.git_sha != manifest.git_sha:
                    problems.append(("warn", f"line {i} {cid}: git_sha != manifest"))

    if n == 0:
        problems.append(("warn", "ledger is empty"))
    return problems
