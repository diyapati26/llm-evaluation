"""pytest configuration: load .env at collection time + shared fixtures."""
from __future__ import annotations

import pytest

from latest.env import load_env

load_env()  # make API keys available before any live test collects


@pytest.fixture
def run_root(tmp_path):
    """A throwaway results root for ledger/cache tests."""
    return tmp_path
