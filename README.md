# Hindsight + LangGraph: three memory scopes

A small experiment for personal, organizational, and event memory, with source-backed figures and MLflow traces/evaluation. All included people, policies, incidents, and numbers are synthetic.

**Start with the no-model harness, then run the same graph against real Hindsight.** DuckDB is a test double and source snapshot registry. It does not replace Hindsight's PostgreSQL engine or reproduce its semantic/graph retrieval.

## Design decisions

| Scope | Bank boundary | Capture | Read path |
|---|---|---|---|
| Personal | Organization + user | Selective end-of-turn node, human text only | Automatic recall before answering |
| Organizational | Organization, published knowledge only | OKF document ingestion; chat learnings enter a candidate queue; curator promotion | Scope-bound search tool, pertinence gate, source-opening tool |
| Event | Organization + active authorized event | Report ingestion; automatic end-of-turn extraction; explicit remember tool | Scope-bound search tool, pertinence gate, source-opening tool |

The graph is `route → recall tools → pertinence → open sources → answer → remember`. In the live Compose setup, `RETRIEVAL_MODE=hybrid` combines Hindsight recall with a line-level search of the synchronized source registry. Both paths use the same authorized bank and source-time filters; the combined list has at most five candidates per bank. Hindsight errors remain visible. Set `RETRIEVAL_MODE=hindsight` to measure Hindsight-only retrieval. The local source search is a POC-scale supplement, not a claim about Hindsight's native accuracy.

**A numeric claim is supported by a source passage, not by a remembered number.** Hindsight receives narrative hints and source IDs. The mock answers quote immutable source passages returned by `read_source`. The passage includes its revision, source location, effective time, recorded time, and checksum. This preserves the relationship between a number, its unit, and what it measures. A number occurring somewhere in memory is not enough evidence.

The answer node is intentionally extractive so the first evaluation measures memory behavior without adding a synthesis-model confound. It does not implement a fluent conversational agent. Your existing LangGraph model can replace that node after retrieval and numeric attribution work.

Set `ANSWER_MODEL=llm` to try an Ollama-generated answer over the opened passages. This is an experimental integration path: inspect the resulting text and citations in MLflow. The default extractive answer is the source-grounded baseline; the model prompt alone does not guarantee accurate numeric claims.

Read [docs/design.md](docs/design.md) for the reasoning and integration approach, [docs/testing.md](docs/testing.md) for the evaluation protocol and limitations, and [docs/benchmarks.md](docs/benchmarks.md) for existing datasets that exercise scoped memory.

A paired diagnostic on the published memory-bench organizational dataset can be run with [these instructions](docs/benchmarks.md#published-memory-bench-diagnostic). Start with the [visual report index](index.html) or [results guide](RESULTS.md). The [hybrid retrieval report](results/memory-bench-hybrid-retrieval/report.html) checks whether the target event was opened on each side; the [hybrid 7B answer report](results/memory-bench-hybrid-7b/report.html) checks the generated outputs. For comparison, the [Hindsight-only retrieval report](results/memory-bench-retrieval/report.html) and [Hindsight-only 7B answer report](results/memory-bench-7b/report.html) used the same previously retained banks. These are small diagnostics, not official benchmark scores. The published memory-bench excerpts in the reports are credited to [notmehul/memory-bench](https://huggingface.co/datasets/notmehul/memory-bench) under CC BY 4.0.

## Fastest run: Docker Compose, no model

Requires Docker with Compose v2. Run from this directory.

```bash
cp .env.example .env
docker compose up -d --build mlflow
docker compose run --rm --build app ingest
docker compose run --rm --entrypoint python app scripts/verify_trace.py
docker compose run --rm app evaluate
docker compose run --rm --entrypoint python app scripts/render_report.py
docker compose run --rm app chat "What is the current Riverside shelter occupancy and capacity?" --event riverside
```

Open [MLflow](http://localhost:5000). Runs include the fixture dataset, per-case outputs, metrics, gate decisions, and graph/tool traces. Local comparison files appear in `results/`.
Run `docker compose up -d reports`, then open the [mock report](http://localhost:8765/report.html) or [live Hindsight report](http://localhost:8765/live/report.html) in a browser. Run the report command again after each evaluation.

The nine-case baseline is a wiring check and has a ceiling on the included fixtures. Run the separate challenge cases to probe paraphrases, unsupported questions, and missing user/event context:

```bash
docker compose run --rm app evaluate --cases data/eval_challenge.jsonl --output results/challenge
docker compose run --rm --entrypoint python app scripts/render_report.py --results results/challenge --output results/challenge/report.html --dataset "Challenge set"
```

Open the [mock challenge report](http://localhost:8765/challenge/report.html) or [live Hindsight challenge report](http://localhost:8765/live-challenge/report.html). With `MEMORY_BACKEND=hindsight`, use a distinct output directory such as `results/live-challenge` and render with `--backend hindsight`.

The fixture correction says occupancy was **16 people**, correcting a previous report of **18**, with capacity of **30**. The answer quotes the correction document, including its context. Morning and historical questions deliberately return different revisions.

```bash
docker compose run --rm app chat "Riverside shelter occupancy and capacity" --event riverside --thread morning --as-of 2026-02-01T10:00:00Z
docker compose run --rm app chat "Riverside shelter occupancy and capacity" --event riverside --thread as-known --as-of 2026-02-01T15:00:00Z --known-at 2026-02-01T16:00:00Z
docker compose run --rm app chat "How did Riverside occupancy change?" --event riverside --thread history --history
```

`--as-of` is the effective-time cutoff. `--known-at` is the reporting/knowledge cutoff. With `--as-of` alone, later corrections to that historical state are allowed. To reproduce what was known at that time, set both. These are strict application-side filters; Hindsight's temporal retrieval and query timestamp alone do not enforce them.

## Real Hindsight + local model

The `live` profile uses Hindsight's **slim** image, embedded PostgreSQL, Ollama `qwen2.5:7b`, Ollama `all-minilm` embeddings through its OpenAI-compatible endpoint, and RRF in place of a neural reranker. It downloads models on first start. No provider API key is needed.

```bash
docker compose --profile live up -d --wait
```

After services are healthy, change `MEMORY_BACKEND=hindsight` in `.env`, then:

```bash
docker compose run --rm app ingest
docker compose run --rm --entrypoint python app scripts/verify_trace.py
docker compose run --rm app evaluate --output results/live
docker compose run --rm --entrypoint python app scripts/render_report.py --results results/live --output results/live/report.html --backend hindsight
docker compose run --rm --entrypoint python app scripts/live_smoke.py
```

Open the [Hindsight UI](http://localhost:19999) to inspect the banks. Live and mock state use separate paths. The Hindsight adapter never falls back to local lexical retrieval. A live failure or missing result stays visible.

Initially, Hindsight uses the configured local model for its own fact extraction. The graph's pertinence and write-selection nodes use rules. To try semantic classification, set `GATE_MODEL=llm` and/or `WRITE_MODEL=llm` in `.env`. The same model will then select relevant source IDs and/or verbatim user spans to retain. Fabricated extracted spans are discarded. Source opening and numeric grounding remain deterministic.

To exercise the answer model as well, set `ANSWER_MODEL=llm`. Run `evaluate` again and compare its answer assertions and traces against `ANSWER_MODEL=extractive`. The `numeric_grounding_pass` metric checks the opened passages, not claims in generated text; inspect generated numeric statements against their cited passages.

A 7B model is still an experiment, not a known-good extraction benchmark. Strict structured output is enabled; if retains fail or miss key concepts, inspect the pending queue and Hindsight logs. Automatic observations and reflection are disabled in this experiment to reduce work and isolate the raw-memory/source-pointer path.

In the earlier 1.5B fixture run, the generated answer gave the right Riverside figures but omitted the requested source IDs. The [validation results](validation/STATUS.md) include that check. Use the extractive baseline when the answer itself must preserve exact source passages.

Upstream service image tags are mutable. After the first successful run, pin `HINDSIGHT_IMAGE` and the Ollama image to the tested digests. Python package pins and a complete resolved app dependency list are included.

## Try the three write paths

In rules mode, these phrases make the mock's scope classification explicit:

```bash
docker compose run --rm app chat "I prefer concise maps." --thread prefs --turn preference-1
docker compose run --rm app chat "Event update: the west road reopened." --event riverside --thread incident --turn update-1
docker compose run --rm app chat "Remember for this event: use the west entrance for accessible transfers." --event riverside --thread incident
docker compose run --rm app chat "Org learning: ask the duty officer to verify route maps." --thread learning
```

The last command prints a candidate ID. It is not searchable as published knowledge until:

```bash
docker compose run --rm app promote REPLACE_WITH_PRINTED_ID --user curator
```

The candidate ID already starts with `candidate-`; pass the full printed ID once. Repeating a turn with the same payload and `--turn` does not create a second personal record. Event notes and org candidates use stable content identities. Changing active event requires a new thread. Tool-based writes queue a record; `sync` publishes pending records to Hindsight. On failure, the queued record remains durable and the CLI exits nonzero.

With `WRITE_MODEL=llm`, the end-of-turn node also handles ordinary unprefixed event reports and preferences. This is experimental semantic selection. In rules mode, unmarked reports are deliberately not inferred.

## Local Python alternative

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
memory-poc ingest
timeout 60 python scripts/verify_trace.py
memory-poc evaluate
pytest -q
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

On platforms without `timeout`, run the trace check under an equivalent 60-second process deadline. Local runs default to SQLite-backed MLflow. If using the Compose tracking server from your host, set `MLFLOW_TRACKING_URI=http://localhost:5000`.

For host-side live experiments, set `MEMORY_BACKEND=hindsight`, `HINDSIGHT_URL=http://localhost:18888` (or `HINDSIGHT_HOST_PORT`), and, for LLM gates or answers, `MODEL_BASE_URL=http://localhost:11435/v1` (or `OLLAMA_HOST_PORT`). The CLI automatically chooses a separate `.state/hindsight` directory.

The app includes the full MLflow package because the pinned LangGraph autologging integration imports its model dependencies. This avoids a skinny-client import failure and keeps local SQLite tracking available as well.

## Files to change

| File | Purpose |
|---|---|
| `memory_poc/graph.py` | Nodes, scope-bound LangChain tools, rule/LLM selection, extractive answer |
| `memory_poc/core.py` | Bank mapping, source registry/outbox, DuckDB double, real HTTP adapter, temporal filters |
| `memory_poc/ingest.py` | Minimal OKF concept ingestion and source-pointer validation |
| `data/knowledge/` | Linked OKF organizational documents |
| `data/sources/` | Human-readable synthetic source documents |
| `data/records.jsonl` | Source passage snapshots, ownership, revision links, timestamps |
| `data/eval.jsonl` | Nine labeled retrieval/answer cases |
| `data/principals.json` | Test identities and allowed events |
| `scripts/live_smoke.py` | Opt-in real Hindsight round trip |
| `tests/` | Isolation, temporal, write, grounding, checkpoint and HTTP-contract checks |

The CLI is a trusted local test harness. Its `--user` option is not authentication. The bundled cases are synthetic; no de-identification pipeline is claimed. Before using real case files, replace fixture identity selection with your authenticated principal and supply reviewed, appropriately de-identified source passages.

## Shutdown

`docker compose --profile live down` stops the suite and preserves named volumes. Removing volumes erases the POC state, so that is intentionally not part of the normal shutdown command.

## References checked for this POC

* [Hindsight LangGraph integration](https://hindsight.vectorize.io/sdks/integrations/langgraph)
* [Hindsight retain API](https://hindsight.vectorize.io/developer/api/retain)
* [Hindsight recall API](https://hindsight.vectorize.io/developer/api/recall)
* [Hindsight HTTP reference](https://hindsight.vectorize.io/api-reference)
* [Hindsight installation](https://hindsight.vectorize.io/developer/installation) and [configuration](https://hindsight.vectorize.io/developer/configuration)
* [Hindsight vs RAG](https://hindsight.vectorize.io/guides/2026/04/21/comparison-hindsight-vs-rag-for-ai-agents)
* [Staleness-Aware Memory](https://hindsight.vectorize.io/blog/2026/06/17/freshness-aware-memory)
* [From Documents to Decisions: architxt and Hindsight](https://hindsight.vectorize.io/blog/2026/07/03/architxt-hindsight-temporal-mosaic)
* [Canonical Open Knowledge Format specification](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md)
* [LangGraph memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
* [MLflow LangGraph tracing](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph/)
* [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
