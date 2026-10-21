from __future__ import annotations

import json
import os
import statistics
import time
import uuid
from pathlib import Path

import mlflow

from .core import MemoryService, Principal, digest
from .graph import build_graph, invoke


def configure_tracking():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    if not os.getenv("MLFLOW_EXPERIMENT_ID"):
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "three-scope-memory"))
    mlflow.langchain.autolog()


def evaluate(service: MemoryService, data: Path, principals: dict, output: Path, modes: list[str],
             cases_file: Path | None = None):
    cases_file = cases_file or data / "eval.jsonl"
    cases = [json.loads(line) for line in cases_file.read_text().splitlines() if line.strip()]
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    with mlflow.start_run(run_name="memory-comparison") as parent:
        mlflow.log_params({"backend": os.getenv("MEMORY_BACKEND", "duckdb"),
                           "gate_model": os.getenv("GATE_MODEL", "rules"),
                           "write_model": os.getenv("WRITE_MODEL", "rules"),
                           "answer_model": os.getenv("ANSWER_MODEL", "extractive"),
                           "model": os.getenv("MODEL_NAME", "qwen2.5:7b"),
                           "read_only": True, "dataset_sha256": digest(cases_file.read_text()),
                           "dataset": cases_file.name})
        mlflow.log_artifact(str(cases_file), artifact_path="dataset")
        for mode in modes:
            with mlflow.start_run(run_name=mode, nested=True):
                mlflow.log_params({"mode": mode, "backend": os.getenv("MEMORY_BACKEND", "duckdb")})
                rows = []
                for c in cases:
                    p = Principal(user=c["user"], **principals[c["user"]])
                    start = time.perf_counter()
                    graph = build_graph(service, p, event=c.get("event"), as_of=c.get("as_of"),
                                        known_at=c.get("known_at"), history=c.get("history", False), read_only=True)
                    result = invoke(graph, service, p, c["query"], "eval-" + uuid.uuid4().hex, c["id"], mode, c.get("event"))
                    got = {s["id"] for s in result["sources"]}
                    expected = set(c["expected_sources"])
                    # Precision for an empty result is 1; recall exposes missing evidence.
                    precision = len(got & expected) / len(got) if got else 1.0
                    recall = len(got & expected) / len(expected) if expected else float(not got)
                    contains = all(x.lower() in result["answer"].lower() for x in c.get("must_contain", []))
                    avoids = all(x.lower() not in result["answer"].lower() for x in c.get("must_not_contain", []))
                    scoped = all(s["org"] == p.org and (s["scope"] != "personal" or s["user"] == p.user)
                                 and (s["scope"] != "event" or s["event"] == c.get("event")) for s in result["sources"])
                    # Entire evidence passages are exact source text, preserving number/relationship pairs.
                    numeric_grounded = all(service.ledger.get(
                        p.bank(s["scope"], c.get("event")), s["id"]).text == s["text"] for s in result["sources"])
                    rows.append({"case": c["id"], "query": c["query"], "expected_sources": sorted(expected),
                                 "actual_sources": sorted(got), "answer": result["answer"],
                                 "decisions": result["decisions"], "precision": precision, "recall": recall,
                                 "answer_pass": float(contains and avoids), "scope_pass": float(scoped),
                                 "numeric_grounding_pass": float(numeric_grounded),
                                 "latency_ms": 1000 * (time.perf_counter() - start)})
                metrics = {k: statistics.mean(r[k] for r in rows) for k in (
                    "precision", "recall", "answer_pass", "scope_pass", "numeric_grounding_pass", "latency_ms")}
                positives = [r for r in rows if r["expected_sources"]]
                negatives = [r for r in rows if not r["expected_sources"]]
                metrics["positive_precision"] = statistics.mean(
                    r["precision"] if r["actual_sources"] else 0.0 for r in positives) if positives else 0.0
                metrics["positive_recall"] = statistics.mean(r["recall"] for r in positives) if positives else 0.0
                metrics["abstention_accuracy"] = statistics.mean(
                    float(not r["actual_sources"]) for r in negatives) if negatives else 0.0
                summary[mode] = metrics
                mlflow.log_metrics(metrics)
                mlflow.log_dict(rows, "cases.json")
                (output / f"{mode}.json").write_text(json.dumps(rows, indent=2))
        mlflow.log_dict(summary, "comparison.json")
        (output / "comparison.json").write_text(json.dumps(summary, indent=2))
        return {"run_id": parent.info.run_id, "metrics": summary}
