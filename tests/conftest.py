import json
import os
from pathlib import Path

os.environ["MLFLOW_TRACING_ENABLED"] = "false"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import pytest

from memory_poc.core import DuckMemory, Ledger, MemoryService, Principal
from memory_poc.ingest import ingest


@pytest.fixture
def service(tmp_path):
    ledger = Ledger(tmp_path / "sources.duckdb")
    svc = MemoryService(ledger, DuckMemory(ledger))
    root = Path(__file__).parents[1] / "data"
    admin = Principal(user="curator", **json.loads((root / "principals.json").read_text())["curator"])
    ingest(svc, root, admin)
    svc.sync()
    yield svc
    ledger.close()


@pytest.fixture
def alice():
    return Principal("relief-demo", "alice", ("riverside",))
