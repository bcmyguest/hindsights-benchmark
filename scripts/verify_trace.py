"""Run once before evaluation; verify one captured trace under a 60s shell timeout."""
import json
import time
from pathlib import Path

import mlflow

from memory_poc.core import Ledger, MemoryService, Principal, backend
from memory_poc.evaluate import configure_tracking
from memory_poc.graph import build_graph, invoke

configure_tracking()
ledger = Ledger(".trace-check.duckdb")
service = MemoryService(ledger, backend(ledger))
p = Principal("trace-test", "trace-test")
start = int(time.time() * 1000)
result = invoke(build_graph(service, p, read_only=True), service, p, "unknown chess result", "trace-test", "trace-test")
trace_id = mlflow.get_last_active_trace_id()
print(json.dumps({"trace_id": trace_id}), flush=True)
print("Starting trace readback", flush=True)
mlflow.flush_trace_async_logging()
trace = mlflow.get_trace(trace_id)
assert trace and trace.info.request_time >= start and trace.data.spans
print(json.dumps({"trace_id": trace_id, "spans": [s.name for s in trace.data.spans]}, indent=2))
ledger.close()
