# Design review and integration notes

The three scopes are useful because ownership, authority, and lifetime differ. They should share an interface, not a single unqualified bank.

## Enhancements to the proposed design

**Keep session state and durable memory separate.** LangGraph checkpoints preserve the execution state of a thread. A memory bank carries selected context across threads. Existing Postgres checkpoints can remain where they are. Hindsight can use the same Postgres cluster, with its own database/schema and migrations. Sharing a cluster does not create a shared transaction between the graph and memory API. For this POC, checkpoints use SQLite, source snapshots/outbox use DuckDB, and live Hindsight owns its embedded PostgreSQL. DuckDB cannot serve as Hindsight's database.

**Use personal banks per user, organizational banks per organization, event banks per event and organization.** Keep IDs opaque and derive them from an authenticated principal. Strict tags provide additional scope checks, while the gateway authorizes access before making any Hindsight request. The example tools bind scope in closures: the model never supplies a tenant ID, user ID, bank ID, or event ID. Bank naming alone does not secure a directly exposed Hindsight API.

**Automatically evaluate retention every turn, rather than retain every turn verbatim.** Prefer durable preferences, explicit commitments, stable personal context, and event reports. Ignore greetings, hypothetical scenarios, and assistant-generated claims. Keep human attribution. The cheap default extractor recognizes a few explicit phrases; optional structured LLM selection handles ordinary phrasing. It selects verbatim spans and cannot invent source text. This still needs quality evaluation: substring validation prevents invented wording, not wrong scope classification.

**Separate read pertinence from write pertinence.** On reads, filter authorized candidates by temporal validity and pertinence to the question, then allocate a small quota per scope. On writes, decide whether a new claim is worth retaining and whether it is personal, event-specific, or organizational. Neither node is a substitute for authorization.

**Do not use multi-bank search to decide publication.** Search combines evidence already published in authorized banks. It cannot safely decide that one user's experience should become organizational knowledge. Chat-derived org learnings enter a review queue with their source and proposer. A curator can promote them. Later you can automate promotion for narrowly trusted sources or repeatedly corroborated patterns, and evaluate false promotions separately.

**Use OKF as a curated knowledge interchange layer.** The organization bundle contains typed Markdown concepts, links, producer-defined provenance, status and timestamp fields. The importer preserves frontmatter in metadata and retains searchable hints. It is deliberately not a full OKF validator or exporter. Hindsight's internal entity graph supports its memory retrieval; it does not become your authoritative knowledge graph schema. If you need audited entities, typed edges and stable IDs across systems, keep a canonical concept/edge catalog and make Hindsight a derived index over it. Reviewed updates should become versioned OKF changes before a production org bank is refreshed. The POC promotion command publishes a candidate to the bank; it does not write a reviewed concept back into the OKF bundle.

## Source-backed numeric facts

Treat a source pointer as `{scope, source_id, revision, passage, checksum, effective_time, recorded_time}`. The POC represents revision with an immutable source ID and `supersedes`, passage with the URI fragment, and exact source text with the snapshot stored in DuckDB. Source fixtures are included as Markdown, and ingestion checks that the snapshot occurs in its referenced file. At answer time, `read_source` resolves the registered immutable snapshot, rechecks authorization/time/revision, and computes its checksum. It does not fetch arbitrary URLs or read model-specified filesystem paths.

Search returns a narrative hint and pointer. The next node opens only selected source pointers. Exact numbers and units come from those opened passages. A stale pointer is rejected; an unregistered pointer is not silently replaced by an unrelated local search hit. The live adapter requests raw facts, avoiding a consolidated observation's many-source attribution problem in the first experiment.

The live Compose default is now `RETRIEVAL_MODE=hybrid`. It unions Hindsight recall with a line-level inverse-document-frequency ranking over the synchronized source snapshots, limited to five candidate pointers per bank. The local candidate pool is built only after organization, witness, scope, effective-time, knowledge-time, and supersession filtering. Hindsight still runs first and errors still fail the search. This is an application-side hybrid retriever, not Hindsight-only retrieval; set `RETRIEVAL_MODE=hindsight` for the original adapter behavior. The source registry is small and single-process here, so a production implementation should use an indexed search service over the same authorized source records.

The POC masks digit quantities in default hints and provides hand-authored nonnumeric summaries for the incident reports. That masking is a convenience, not semantic redaction or a guarantee that every written-out number disappears. The stronger invariant is that the answer is an exact source quotation. Hindsight-extracted facts are retrieval hints, regardless of whether they contain a number.

Production numeric synthesis needs claim-level citation checking, table/cell or passage locators, units, geography, denominator, observation time, and conflict handling. If calculations are needed, use a deterministic computation tool over opened sources and retain its input references. The POC performs no calculations and does not pretend a numeric-token whitelist establishes factual correctness.

An event report may supply an attributed observation without being verified truth. User-supplied event memories are labeled as unverified when displayed. Organizational procedures have a different authority than incident reports. Preserve those distinctions when your agent synthesizes across the banks.

## Time and changing narratives

There are two separate questions:

* What state was effective at the requested time?
* What information had been reported by the requested knowledge cutoff?

The sample starts with a morning shelter report, adds an afternoon report, then a next-day correction to the afternoon figure. `valid_from` and `recorded_at` allow both questions. Revision links retire an earlier passage for current-state retrieval while retaining history.

Hindsight's `query_timestamp` anchors temporal reasoning, and its temporal retrieval can influence ranking. The application applies hard cutoffs itself. Otherwise, a high semantic match could still return a future or stale source. The POC's read-time temporal guard remains enabled in every experimental arm.

Supersession is passage-level: the new passage fully replaces the old passage. Split a long situation report into independently versioned facets (shelter, access, logistics, assessment) before using this approach on real documents. Replacing the whole report because one figure changed could incorrectly retire still-valid facts about another facet. An open-ended report is not automatically expired by elapsed time; add `valid_to` or explicit expiration policies for transient facts when needed.

The sample does not resolve conflicts among independent sources or automatically assign personal-preference slots. Two contradictory chat preferences can coexist until an explicit revision links them. Implement source authority, preference slots, and confidence policies after establishing the basic measurements.

## Delivery and retry behavior

The remember node first writes a source snapshot to a durable outbox. Synchronous Hindsight retains make test visibility predictable. Hindsight document IDs are stable: retrying the same source revision replaces its document instead of creating duplicates. New revisions use new IDs so historical evidence is retained. The app ledger rejects changing an existing source revision in place.

The POC drains pending writes at the end of a normal chat turn or through `sync`. Failed retains stay queued and cause a nonzero CLI exit. The answer can already exist when memory synchronization fails. In a deployed service, put draining behind a worker so the response stream does not wait on extraction. Poll async Hindsight operations to terminal completion before an evaluation reads the bank; do not equate a queued retain with a visible memory.

DuckDB is appropriate for this single-process harness. It is not a concurrent server job queue. Use the existing Postgres deployment for source metadata and an outbox if multiple application workers will write simultaneously.

## Integrate with an existing LangGraph model

Reuse `make_tools(service, principal, event, ...)` to obtain scope-bound LangChain `StructuredTool` objects. Their public methods are `search_personal`, `search_organization`, `search_event`, `read_source`, `remember_event`, and `propose_organization_learning`. They can be bound to a tool-capable model and run through a normal LangGraph ToolNode loop. Only provide the event tools when an authorized active event exists.

This POC invokes the tools deterministically from graph nodes. That makes it usable with a tiny model that may not reliably choose tools. It does not claim to implement or evaluate an autonomous model/tool loop.

The useful integration boundary is:

```python
tools = make_tools(service, principal, active_event)

# Before your existing model node:
# 1. Recall personal context and candidate org/event pointers.
# 2. Run the pertinence node.
# 3. Call read_source for accepted pointers.
# 4. Supply opened sources as evidence, with scope, authority and dates.

# After your existing model node:
# Select durable facts from the new human turn only and enqueue retains.
```

Keep source evidence in transient turn state, not as another persistent system message copied into every later checkpoint. This example resets all derived retrieval fields on each invocation. Checkpoint threads are namespaced by organization/user, and an existing thread cannot silently switch event.

For your existing Postgres stack, retain your checkpointer and compile with it. Replace only this sample's SQLite context manager. Keep Hindsight writes behind `MemoryService` so you can compare mock/live modes and preserve the outbox.

## Why these Hindsight posts matter

[Hindsight vs RAG](https://hindsight.vectorize.io/guides/2026/04/21/comparison-hindsight-vs-rag-for-ai-agents) describes using memory for ongoing context and retrieval for source documents. This informed the two-stage memory-pointer/source-opening design; requiring source passages for every numeric claim is our application policy.

[Staleness-Aware Memory](https://hindsight.vectorize.io/blog/2026/06/17/freshness-aware-memory) explains how observations and mental models may lag recent raw facts. If you later enable reflection or narrative summaries, use freshness signals and verify important facts against current evidence. This POC avoids the derived layer while measuring the basic path.

[From Documents to Decisions](https://hindsight.vectorize.io/blog/2026/07/03/architxt-hindsight-temporal-mosaic) describes composing current knowledge across partial, dated documents. Its emphasis on source/entity metadata and freshness per facet is relevant to evolving situation reports. The POC's revision model is intentionally simpler than that system.
