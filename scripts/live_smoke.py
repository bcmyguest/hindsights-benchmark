"""Opt-in real Hindsight smoke test. Keeps banks under a unique prefix for inspection."""
import json
import os
import uuid
from pathlib import Path

os.environ["MEMORY_BACKEND"] = "hindsight"
os.environ["BANK_PREFIX"] = "smoke-" + uuid.uuid4().hex[:10]

from memory_poc.core import Ledger, MemoryService, Principal, backend
from memory_poc.graph import build_graph, invoke
from memory_poc.ingest import ingest

state = Path(os.getenv("POC_STATE", ".state/live")) / os.environ["BANK_PREFIX"]
state.mkdir(parents=True, exist_ok=True)
ledger = Ledger(state / "sources.duckdb")
service = MemoryService(ledger, backend(ledger))
principals = json.loads(Path("data/principals.json").read_text())
ingest(service, Path("data"), Principal(user="curator", **principals["curator"]))
sync = service.sync()
assert not sync["errors"], sync
alice = Principal(user="alice", **principals["alice"])
graph = build_graph(service, alice, event="riverside", read_only=True)
r = invoke(graph, service, alice, "Riverside shelter occupancy and capacity", "live", "1", event="riverside")
assert "riverside-v3" in {s["id"] for s in r["sources"]}, "Current pointer not recalled; inspect extraction/model performance"
assert "16 people" in r["answer"] and "99 people" not in r["answer"]
print(json.dumps({"status": "passed", "bank_prefix": os.environ["BANK_PREFIX"], "answer": r["answer"]}, indent=2))
ledger.close()
