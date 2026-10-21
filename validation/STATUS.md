# Validation status

Checked on 2026-10-20 UTC in Docker Compose with Python 3.12, MLflow 3.17.1, Hindsight's `latest-slim` image, and Ollama `qwen2.5:1.5b` plus `all-minilm`. Image tags are mutable; these results do not establish a reproducible image digest.

| Check | Result |
|---|---|
| Container tests | 21 passed |
| Synthetic fixture ingestion | 9 records synchronized in each backend, no errors |
| MLflow trace readback | Passed; graph, recall tools, pertinence, source opening, answer, and remember spans present |
| DuckDB baseline | 36 graph calls; [HTML report](../results/report.html) |
| Real Hindsight baseline | 36 graph calls; [HTML report](../results/live/report.html) |
| DuckDB challenge | 32 graph calls on 8 separate cases; [HTML report](../results/challenge/report.html) |
| Real Hindsight challenge | 32 graph calls on 8 separate cases; [HTML report](../results/live-challenge/report.html) |
| Published memory-bench diagnostic | 4 selected cases, each run on original and counterfactual streams; [HTML report](../results/memory-bench/report.html) |
| Real Hindsight smoke | Passed; recalled the corrected Riverside passage and its 16 people / 30 places figures |
| Tiny-model answer | Completed over live evidence; stated the correct fixture figures but omitted source IDs despite the prompt requesting them |
| Tiny-model pertinence | Completed on a Riverside query; selected the current report plus two additional passages |
| Tiny-model end-of-turn write | Completed on an ordinary preference and event update; synchronized one personal and one event record |
| HTML report JavaScript | `node --check` passed on the embedded script |

## Measured retrieval results

| Backend and arm | Source precision | Source recall | Answer assertions | Scope integrity |
|---|---:|---:|---:|---:|
| DuckDB, no memory | 1.000 | 0.222 | 0.222 | 1.000 |
| DuckDB, personal only | 0.222 | 0.167 | 0.222 | 1.000 |
| DuckDB, all banks | 0.463 | 0.889 | 0.889 | 1.000 |
| DuckDB, all banks + pertinence | 0.926 | 1.000 | 1.000 | 1.000 |
| Hindsight, no memory | 1.000 | 0.222 | 0.222 | 1.000 |
| Hindsight, personal only | 1.000 | 0.278 | 0.333 | 1.000 |
| Hindsight, all banks | 0.926 | 0.889 | 0.889 | 1.000 |
| Hindsight, all banks + pertinence | 0.926 | 0.889 | 0.889 | 1.000 |

Source integrity was 1.000 in every arm. Empty selections have precision 1 by convention, so read that metric with recall and answer assertions. These are nine synthetic cases, not a general accuracy estimate. The live evaluation used the extractive answer baseline; the single generated-answer check is separate. The numeric grounding metric verifies opened passage integrity and does not verify model-generated claims.

The baseline was rerun after the challenge metric change. The live baseline missed one answer assertion on this rerun, showing that the nine-case score is not stable enough to present as general accuracy. Open the local [MLflow UI](http://localhost:5000) while Compose is running to inspect traces and case artifacts.

## Separate challenge set

The challenge set has three positive and five negative cases. Each backend used a fresh app state and fresh bank prefix with the same nine ingested records. These metrics are diagnostic counts over eight cases, not estimates of deployment accuracy.

| Backend, all banks + pertinence | Positive evidence precision | Positive evidence recall | Negative abstention accuracy | Answer assertions |
|---|---:|---:|---:|---:|
| DuckDB test double | 0.667 | 0.667 | 0.400 | 0.500 |
| Real Hindsight | 0.333 | 0.333 | 0.400 | 0.375 |

The challenge demonstrates remaining retrieval and pertinence failures that the baseline hides. See [existing dataset options](../docs/benchmarks.md) for a larger follow-up. No external benchmark was run here.

The tiny model's generated answer omitted source IDs even though its prompt requested them. Its pertinence selection also kept an unrelated case passage for the Riverside occupancy question. Treat `ANSWER_MODEL=llm` and `GATE_MODEL=llm` as quality experiments. For an application that must enforce citations, add an output validator or use the extractive answer path for numeric statements.

## Published memory-bench sample

The [paired report](../results/memory-bench/report.html) contains eight generated outputs for four published probes: A1 (cross-person decision), A2 (silent rule), A4 (scope conflict), and A7 (supersession). It used the `notmehul/memory-bench` release at Hugging Face revision `e6833b44e30b602c81f7087ebf682769fff297dc`, organization `org-00001`, its counterfactual twin, live Hindsight, and Ollama `qwen2.5:1.5b` with 512-token native API answers. Dataset checksums passed. The original and twin each ingested events through E-0069 into separate bank namespaces. All 23 opened passages were checked against the published witness rules; none was outside the probe principal's visibility.

Qualitative inspection found that A7 followed the changed three-service versus two-service cutoff on both sides. A2's original answer shifted the start of the 30-minute window and its twin invented Thursday times instead of the next-business-day rule. A1 missed the respective runbook review rules on both sides, and A4 missed the personal note-sharing time on both sides. The report shows the exact outputs, assertions, and retrieved passages. This is **not an official memory-bench score**: only four cases from one public organization were sampled, the worker differs from the released screening worker, and the upstream blind judge and full protocol were not run.

The OpenAI-compatible Ollama answer path timed out during the first attempt because this installed build continued generating beyond the requested output cap. The completed run used the native `/api/chat` path with `num_predict=512` and resumed from its saved source ledger. Run ID: `20261020T121934Z-5bfd678c`.
