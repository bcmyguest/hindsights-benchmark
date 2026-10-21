# Results guide

These are saved outputs from a small proof of concept, with source passages and individual verdicts in each HTML report. The external cases come from [notmehul/memory-bench](https://huggingface.co/datasets/notmehul/memory-bench), revision `e6833b44e30b602c81f7087ebf682769fff297dc`, licensed CC BY 4.0. The sample is one selected case from each of A1, A2, A4, and A7, run on original and counterfactual streams for eight probes total.

| Diagnostic | Hindsight only | Hybrid source search |
|---|---|---|
| Target event opened | [6/8](results/memory-bench-retrieval/report.html) | [8/8](results/memory-bench-hybrid-retrieval/report.html) |
| 7B answer met visible criterion | [4/8](results/memory-bench-7b/report.html) | [5/8](results/memory-bench-hybrid-7b/report.html) |

The retrieval measure checks whether the target event appears among opened passages. It does not score the generated answer. The answer measure is an assistant review against the visible criterion, not the benchmark's blind judge. The hybrid answer run met both sides of one of four pairs. Each report has a matching `results.json` and `review.json` where applicable.

The runs reused Hindsight banks previously populated with 1.5B extraction. The source-opening path applied each probe's simulated effective-time and knowledge-time cutoff, but future facts retained in those banks could still affect Hindsight ranking. The later 7B runs changed answer generation; they did not redo ingestion. A fresh 7B ingest was stopped before completion because CPU extraction was slow. These results are diagnostics, not official benchmark scores or a validated fresh-bank comparison.

The [mock report](results/report.html), [challenge report](results/challenge/report.html), and [live Hindsight report](results/live/report.html) use the repository's separate synthetic fixture set. See [testing protocol](docs/testing.md) and [benchmark instructions](docs/benchmarks.md) to interpret or rerun them. Local DuckDB state, MLflow state, and downloaded benchmark data are intentionally excluded from this repository.
