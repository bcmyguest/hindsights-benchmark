"""Render evaluation JSON as a self-contained HTML report."""

import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def render(result_dir: Path, output: Path, backend: str, dataset: str) -> None:
    summary = json.loads((result_dir / "comparison.json").read_text())
    cases = {mode: json.loads((result_dir / f"{mode}.json").read_text()) for mode in summary}
    payload = json.dumps({"summary": summary, "cases": cases}).replace("<", "\\u003c")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    note = (
        "Real Hindsight retrieval on synthetic cases. These scores describe this fixture and configuration, "
        "not general accuracy. Source integrity checks opened passages, not generated-answer correctness."
        if backend == "hindsight" else
        "Deterministic DuckDB retrieval test double. These scores are not Hindsight accuracy results. "
        "Source integrity checks opened passages, not generated-answer correctness."
    )
    case_count = len(next(iter(cases.values()))) if cases else 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Three-scope memory evaluation</title>
<style>
:root{color-scheme:light;font-family:system-ui,sans-serif;background:#f5f7fb;color:#18243a}
body{max-width:1100px;margin:0 auto;padding:28px 20px 60px}h1{margin-bottom:6px}h2{margin-top:32px}
.muted{color:#526079}.note{background:#fff6df;border-left:4px solid #d99b18;padding:14px 18px;border-radius:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px}
.card,details{background:white;border:1px solid #dce3ef;border-radius:10px;padding:16px;box-shadow:0 1px 3px #16233d12}
.card h3{margin:0 0 12px}.metric{display:flex;justify-content:space-between;gap:12px;margin:8px 0}
.track{height:7px;background:#e9eef7;border-radius:9px;overflow:hidden}.fill{height:100%;background:#3868d9}
select{font:inherit;padding:7px;border:1px solid #b7c4d9;border-radius:6px;background:white}
details{margin:10px 0}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fb;padding:12px;border-radius:6px}
.pass{color:#196d46}.fail{color:#a12b36}.pill{font-size:.8em;padding:2px 7px;border-radius:12px;background:#e6edf9}
</style></head><body>
<h1>Three-scope memory evaluation</h1>
<p class="muted">""" + html.escape(dataset) + " · " + str(case_count) + " cases · Generated " + html.escape(generated) + " · Backend: " + html.escape(backend) + "</p>\n" + """
<p class="note">""" + html.escape(note) + """</p>
<h2>Comparison</h2><div id="summary" class="grid"></div>
<h2>Cases</h2><label for="mode">Evaluation arm: </label><select id="mode"></select><div id="cases"></div>
<script>const data=""" + payload + """;
const escapeHtml = value => String(value ?? '').replace(/[&<>\"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[ch]));
const names = {none:'No memory',personal:'Personal only',federated:'All authorized banks',gated:'All banks + pertinence'};
const labels = {positive_precision:'Evidence precision',positive_recall:'Evidence recall',abstention_accuracy:'Abstention accuracy',answer_pass:'Answer assertions',scope_pass:'Scope integrity',numeric_grounding_pass:'Source integrity'};
document.querySelector('#summary').innerHTML = Object.entries(data.summary).map(([mode, metrics]) =>
  `<section class="card"><h3>${escapeHtml(names[mode] || mode)}</h3>${Object.entries(labels).filter(([key]) => Number.isFinite(metrics[key])).map(([key,label]) =>
    `<div class="metric"><span>${label}</span><strong>${(100 * metrics[key]).toFixed(1)}%</strong></div><div class="track"><div class="fill" style="width:${100 * metrics[key]}%"></div></div>`).join('')}
    <p class="muted">Mean latency: ${metrics.latency_ms.toFixed(0)} ms</p></section>`).join('');
const select = document.querySelector('#mode');
select.innerHTML = Object.keys(data.summary).map(mode => `<option value="${escapeHtml(mode)}">${escapeHtml(names[mode] || mode)}</option>`).join('');
select.value = data.summary.gated ? 'gated' : Object.keys(data.summary)[0];
function showCases(){document.querySelector('#cases').innerHTML = data.cases[select.value].map(row =>
  `<details><summary>${escapeHtml(row.case)} · <span class="${row.answer_pass ? 'pass':'fail'}">${row.answer_pass ? 'PASS':'FAIL'}</span></summary>
  <p><strong>Question:</strong> ${escapeHtml(row.query)}</p>
  <p><strong>Expected sources:</strong> ${escapeHtml(row.expected_sources.join(', ') || 'none')}<br>
  <strong>Opened sources:</strong> ${escapeHtml(row.actual_sources.join(', ') || 'none')}</p>
  <p>${row.expected_sources.length ? `<span class="pill">evidence precision ${row.actual_sources.length ? (100 * row.precision).toFixed(0) : 0}%</span> <span class="pill">evidence recall ${(100 * row.recall).toFixed(0)}%</span>` : `<span class="pill">abstained ${row.actual_sources.length ? 'no' : 'yes'}</span>`}</p>
  <strong>Answer</strong><pre>${escapeHtml(row.answer)}</pre>
  <strong>Pertinence decisions</strong><pre>${escapeHtml(JSON.stringify(row.decisions, null, 2))}</pre></details>`).join('');}
select.addEventListener('change', showCases);showCases();
</script></body></html>""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/report.html"))
    parser.add_argument("--backend", default=os.getenv("MEMORY_BACKEND", "duckdb"))
    parser.add_argument("--dataset", default="Fixture baseline")
    args = parser.parse_args()
    render(args.results, args.output, args.backend, args.dataset)
    print(args.output)


if __name__ == "__main__":
    main()
