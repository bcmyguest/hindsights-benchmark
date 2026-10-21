# Evaluation protocol

The first experiment asks whether each scope supplies useful, authorized evidence, whether exact figures resolve to sources, and whether retention stays selective and replay-safe.

## Comparison arms

| Arm | Reads | Pertinence |
|---|---|---|
| `none` | No memory | None |
| `personal` | Current user's bank | No extra topical gate |
| `federated` | Personal, organizational, active event | No extra topical gate |
| `gated` | Same authorized banks as federated | Rule-based or optional LLM selection |

Authorization, temporal cutoffs, revision handling, per-bank quotas and source grounding stay enabled in all arms. Evaluation is read-only and uses fresh thread IDs, so an answer cannot teach the next test case its expected result. The fixture data is ingested before the comparison. Nine cases across four arms produce 36 graph calls.

`data/eval_challenge.jsonl` is a separate eight-case probe. It tests paraphrased requests and negative questions where a passage shares the topic but does not contain the requested fact. It is not mixed into the baseline score. Pass it with `evaluate --cases data/eval_challenge.jsonl` and use a separate `--output` directory.

For the initial experiment, both federated arms recall all authorized scopes. This isolates the pertinence gate. A production router can avoid an org/event call when its likelihood of helping is low; evaluate that as a later cost/recall ablation rather than silently changing both routing and gating at once.

## Metrics logged to MLflow

| Metric | Meaning | Limitation |
|---|---|---|
| `precision` | Fraction of selected source IDs in the labeled expected set | Expected source labels are fixture-specific; empty selection scores 1 |
| `recall` | Fraction of expected source IDs selected | Empty-expected cases score 1 only when nothing is selected |
| `positive_precision` | Mean source precision on cases that require evidence | A positive case with no selected source scores 0 |
| `positive_recall` | Mean source recall on cases that require evidence | Separates missed evidence from correct abstention |
| `abstention_accuracy` | Fraction of no-evidence cases that opened no source | A topical but non-answering passage counts as a false positive |
| `answer_pass` | Required phrases present and forbidden phrases absent | Deterministic fixture assertion, not semantic answer grading |
| `scope_pass` | All selected evidence belongs to the permitted tenant/user/event | Unit tests separately exercise denied tool calls |
| `numeric_grounding_pass` | Opened passages exactly match registered source snapshots | Integrity check; it does not measure real-world truth or whether a source is trustworthy |
| `latency_ms` | Graph construction + execution duration per case | Mock timings are not Hindsight performance measurements |

No-source answers can pass evidence integrity while failing answer coverage. Read integrity together with recall and answer assertions. Do not use the integrity metric alone as a quality score.
The HTML reports emphasize positive-case evidence precision/recall and negative-case abstention accuracy. The older aggregate `precision` and `recall` remain in JSON and MLflow for comparison, but mix these two different tasks.

The deterministic lexical backend deliberately returns distractors, analogous to nearest-neighbor search without a score floor. The gate removes candidates lacking topical word overlap; an event name alone does not count as topical evidence. This is a transparent test double, not a Hindsight simulation. Mock/live metric differences cannot be attributed purely to the memory product because their retrieval algorithms differ.

The LLM gate receives only candidate hints and returns source IDs; IDs not in the candidate set are ignored. Its schema/parsing failures are visible errors, not silent rule fallbacks. It cannot bypass source authorization or temporal filtering. Evaluate paraphrases and irrelevant-but-similar material before trusting it in production.

`ANSWER_MODEL=llm` switches the final graph node to Ollama while leaving retrieval and source opening intact. Answer assertions then measure the model output too. `numeric_grounding_pass` still checks source passage integrity only; it does not verify that generated numeric claims are supported. Review source IDs and quoted passages in each case before drawing accuracy conclusions.

## Run commands

```bash
docker compose run --rm --entrypoint python app -m pytest -q
docker compose run --rm --entrypoint python app scripts/verify_trace.py
docker compose run --rm app evaluate
docker compose run --rm --entrypoint python app scripts/render_report.py
```

For a real Hindsight smoke check after the `live` services are healthy:

```bash
docker compose run --rm --entrypoint python app scripts/live_smoke.py
```

The live smoke uses a unique bank prefix and keeps its records available for inspection. Running it performs real ingestion and may take longer on CPU. A recall miss fails explicitly. Nothing is substituted from local lexical retrieval.

## What the unit tests cover

* Organization/user/event isolation, including a denied event request.
* Morning, afternoon-as-known, and later-corrected historical states.
* Historical revision preservation and stale-pointer rejection.
* Source-derived figures absent from memory hints.
* Unknown-topic abstention.
* Selective personal/event/org capture and same-turn replay.
* Organizational review/promotion permissions.
* Durable retry queue across a simulated Hindsight outage.
* Checkpoint isolation and clearing stale turn evidence.
* HTTP retain/recall shape, strict tags and source-ID handling.
* No silent local fallback and no grounding of unknown remote pointers.

HTTP transport tests validate our adapter against documented payloads with mocked responses. They do not prove compatibility with a running Hindsight image. The live smoke is the separate check for that.

## Add next, after the smoke test

Expand to paraphrased personal preferences, source deletion, conflicting reports, denied events, numeric units/denominators, no-active-event requests, many irrelevant docs, late corrections, repeated source ingestion, and attempts to put instructions in ingested documents. Use reviewed synthetic incident/case material first.

Evaluate write selection separately with labels for durable/non-durable and intended scope. For org promotion, measure false promotion rate before automating it. For retrieval, compare a fixed source set and record the exact Hindsight image digest, model digest, gate configuration, and fixture version. Consider judge-based fluency/correctness evaluation only after you replace the extractive answer with your own model node.

## Execution status

See `validation/STATUS.md` in this package for the checks actually run while preparing this POC. Docker and the live model path must be verified on a machine with Docker available.
