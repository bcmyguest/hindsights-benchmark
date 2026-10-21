import argparse
import json
import os
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .core import Ledger, MemoryService, Principal, backend
from .evaluate import configure_tracking, evaluate
from .graph import build_graph, invoke
from .ingest import ingest


def main():
    parser = argparse.ArgumentParser(description="Synthetic three-scope memory POC")
    parser.add_argument("--state", default=os.getenv("POC_STATE", ".state/" + os.getenv("MEMORY_BACKEND", "duckdb")))
    parser.add_argument("--data", default="data")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest")
    sub.add_parser("sync")
    chat = sub.add_parser("chat")
    chat.add_argument("query")
    chat.add_argument("--user", default="alice")
    chat.add_argument("--event")
    chat.add_argument("--thread", default="demo")
    chat.add_argument("--turn", default=None, help="Reuse for idempotent replay")
    chat.add_argument("--mode", choices=["none", "personal", "federated", "gated"], default="gated")
    chat.add_argument("--as-of")
    chat.add_argument("--known-at")
    chat.add_argument("--history", action="store_true")
    promote = sub.add_parser("promote")
    promote.add_argument("id")
    promote.add_argument("--user", default="curator")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--modes", nargs="+", default=["none", "personal", "federated", "gated"])
    ev.add_argument("--output", default="results")
    ev.add_argument("--cases", type=Path, help="Evaluation JSONL; defaults to data/eval.jsonl")
    args = parser.parse_args()
    state, data = Path(args.state), Path(args.data)
    state.mkdir(parents=True, exist_ok=True)
    principals = json.loads((data / "principals.json").read_text())
    ledger = Ledger(state / "sources.duckdb")
    service = MemoryService(ledger, backend(ledger), os.getenv("RETRIEVAL_MODE", "hybrid"))
    try:
        if args.command == "ingest":
            p = Principal(user="curator", **principals["curator"])
            count = ingest(service, data, p)
            result = {"ingested": count, **service.sync()}
        elif args.command == "sync":
            result = service.sync()
        elif args.command == "promote":
            p = Principal(user=args.user, **principals[args.user])
            r = ledger.promote(p.bank("org"), args.id, p)
            result = {"promoted": r.id, **service.sync()}
        else:
            configure_tracking()
            if args.command == "evaluate":
                result = evaluate(service, data, principals, Path(args.output), args.modes, args.cases)
            else:
                p = Principal(user=args.user, **principals[args.user])
                # The CLI's allowlisted identity is a test harness, not authentication.
                with SqliteSaver.from_conn_string(str(state / "checkpoints.sqlite")) as cp:
                    graph = build_graph(service, p, cp, args.event, args.as_of, args.known_at, args.history)
                    r = invoke(graph, service, p, args.query, args.thread, args.turn or uuid.uuid4().hex, args.mode, args.event)
                result = {k: r[k] for k in ("answer", "routes", "decisions", "writes", "sync")}
        print(json.dumps(result, indent=2))
        errors = result.get("errors", result.get("sync", {}).get("errors", []))
        if errors:
            raise SystemExit("Pending memory writes remain; inspect configuration and run sync")
    finally:
        ledger.close()


if __name__ == "__main__":
    main()
