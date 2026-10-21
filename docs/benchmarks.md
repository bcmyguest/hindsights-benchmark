# Existing datasets for the three-scope experiment

The included nine-case baseline checks wiring. The eight-case challenge set exposes some retrieval and abstention failures, but neither is a published benchmark. For a reusable external dataset, these are the closest matches:

| Dataset | Useful part of this design | Practical limit |
|---|---|---|
| [memory-bench](https://github.com/notmehul/memory-bench) ([data](https://huggingface.co/datasets/notmehul/memory-bench)) | Best match for shared organizational memory: streams of meetings, chat, email, documents, and PRs; personal/team/org conflicts, updates, and decisions learned by other people. | Software organizations rather than disaster events. The authors report failed validity gates and no evaluated memory-system comparison beyond a memoryless floor; treat the released cases as diagnostic data, not a settled leaderboard. |
| [PrecisionMemBench](https://github.com/tenurehq/precisionmembench) ([data](https://huggingface.co/datasets/tenurehq/precisionmembench)) | Small, labeled retrieval set for scoped beliefs, updates, cross-user isolation, and distractor precision. A good first adapter for this POC's bank routing and pertinence gate. | Its universal/code/writing scopes are domain scopes, not personal/org/event ownership. It does not test document ingestion or event timelines. |
| [GateMem](https://github.com/rzhub/GateMem/blob/main/DATASET_CARD.md) | Multi-principal access, utility, and deletion scenarios; relevant to user and organization boundaries. | Synthetic episodes and LLM-based evaluation; no native event bank or source-grounded situation reports. |
| [EnterpriseMem-Bench](https://github.com/enterprisemem-bench/enterprisemem-bench) | Organizational fact introduction, update, retirement, and deterministic answer scoring. Its 10-question subset is inexpensive to try. | One fictional organization's finance context; no personal bank or cross-person access evaluation. |

**Recommended order:** Use PrecisionMemBench first for a fast retrieval-only comparison across the existing bank router, then use memory-bench to exercise org knowledge learned through several people's streams. Add GateMem if cross-user leakage and deletion are central to the application. Preserve each dataset's native labels and report scores separately; do not relabel code/writing domains as event banks or combine unrelated scores into one number.

For a disaster-event bank, the published sets above are partial analogues. A later event-specific evaluation still needs dated revisions, source pointers, incident boundaries, and questions about what was known at a given time. The included Riverside fixture tests that behavior locally, but is too small for an accuracy claim.

## Published memory-bench diagnostic

`scripts/run_memory_bench.py` replays a bounded paired sample from the released memory-bench data into the existing graph. Download the [dataset](https://huggingface.co/datasets/notmehul/memory-bench) at revision `e6833b44e30b602c81f7087ebf682769fff297dc` to `/tmp/memory-bench-data`, and verify its `CHECKSUMS.txt`. The runner expects `/tmp/memory-bench-data/data` with the original and twin organization directories. With the live Compose stack running:

```bash
docker compose build app
docker compose run --rm -v /tmp/memory-bench-data:/benchmark:ro --entrypoint python app \
  scripts/run_memory_bench.py --data /benchmark/data \
  --dataset-revision e6833b44e30b602c81f7087ebf682769fff297dc \
  --org org-00001 --per-archetype 1 --backend hindsight --mode gated \
  --answer-model extractive \
  --output results/memory-bench-hybrid-retrieval
```

The runner samples one published case from each of A1, A2, A4, and A7, then runs both original and counterfactual streams. It keeps streams in separate bank namespaces and limits each principal to events they witness under the published visibility metadata. Each probe enforces its simulated event timestamp as both effective and knowledge cutoff, including on saved-ledger replay. It searches with the task text and uses the full framed prompt only for answer generation. The Compose default `RETRIEVAL_MODE=hybrid` combines Hindsight recall with line-level ranking of synchronized, authorized source snapshots; set `RETRIEVAL_MODE=hindsight` for the original path. Extractive mode shows opened passages without asking the model to write a deliverable. The [hybrid retrieval report](../results/memory-bench-hybrid-retrieval/report.html) labels target-source coverage; the [hybrid 7B report](../results/memory-bench-hybrid-7b/report.html) labels assistant-reviewed answer criteria. The [Hindsight-only retrieval report](../results/memory-bench-retrieval/report.html) and [Hindsight-only 7B report](../results/memory-bench-7b/report.html) are comparison runs on saved banks populated with 1.5B. Saved-bank replay applies strict source-time filtering but does not reproduce a fresh-bank ingest at each probe. None is an official memory-bench score: the original validity screening used a different, now withdrawn worker, and this POC does not reproduce the benchmark's blind judge or full protocol.

`ollama_native` uses Ollama's `/api/chat` with `num_predict` to bound tiny-model answers. This run uses it because the installed Ollama build did not stop at the requested `max_tokens` through its OpenAI-compatible endpoint. If a replay fails after some events are ingested, pass the saved run ID with `--run-id` to reuse the same ledger and bank namespaces.
