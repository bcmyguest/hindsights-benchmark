"""Run a small, paired diagnostic sample from notmehul/memory-bench."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memory_poc.core import Ledger, MemoryService, Principal, Record, backend
from memory_poc.graph import build_graph, invoke


DATASET_URL = "https://huggingface.co/datasets/notmehul/memory-bench"
LICENSE = "CC BY 4.0"
ARCHETYPES = ("A1", "A2", "A4", "A7")
FRAME_LINES = {
    "org_facing": "The artifact you produce is org-facing (visible to the whole company).",
    "team_internal": "The artifact you produce is internal to your team ({ref}).",
    "personal": "The artifact is for your own personal use.",
    "project": "The artifact belongs to the cross-team project ({ref}).",
    "external": "The artifact is client-facing (external).",
}
SCAFFOLD = (
    "Output only the deliverable itself — no meta-commentary and no notes "
    "about missing information; make reasonable choices and produce your "
    "best version. Apply only what fits this artifact's audience and scope "
    "— do not enumerate practices that belong to other scopes, teams, or "
    "individuals. Where the task calls for thresholds, cadences, or "
    "figures, state them concretely."
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection(org_dir: Path, per_archetype: int) -> tuple[list[dict], dict]:
    probes = _read_jsonl(org_dir / "probes.jsonl")
    report = json.loads((org_dir / "g3-report.json").read_text())
    counts = Counter(p["archetype"] for p in probes)
    candidates = defaultdict(list)
    for probe in probes:
        anchor = report["clusters"][probe["cluster_id"]]["instances"][probe["probe_id"]]
        if not anchor.get("valid", False):
            continue
        probe = {**probe, "sampling_anchor": {
            "valid": bool(anchor.get("valid")), "floor": anchor.get("floor"),
            "ceiling": anchor.get("ceiling"),
        }}
        event_n = int(probe["inject_after_event"].removeprefix("E-"))
        candidates[probe["archetype"]].append((
            float(anchor.get("floor", 1.0)), event_n, probe["probe_id"], probe
        ))
    chosen = []
    selected_counts = {}
    for archetype in ARCHETYPES:
        ranked = sorted(candidates[archetype])
        selected = [row[3] for row in ranked[:per_archetype]]
        if len(selected) < per_archetype:
            raise ValueError(f"Only {len(selected)} valid samples available for {archetype}")
        chosen.extend(selected)
        selected_counts[archetype] = len(selected)
    chosen.sort(key=lambda p: (int(p["inject_after_event"][2:]), p["probe_id"]))
    return chosen, {
        "available_probe_counts": dict(counts),
        "valid_instances_available": {k: len(v) for k, v in candidates.items()},
        "selected_instances": selected_counts,
        "rule": (
            f"For each archetype, selected {per_archetype} instances marked valid in the "
            "published g3-report, sorted by lowest published floor anchor, then earliest "
            "inject_after_event, then probe_id. Anchor status is a sampling hint only; "
            "no anchor score is reused."
        ),
    }


def _persona_line(org: dict, pid: str) -> str:
    persona = next(p for p in org["entities"]["personas"] if p["id"] == pid)
    name = pid.split(":", 1)[1].capitalize()
    teams = ", ".join(t.split(":", 1)[1] for t in persona.get("teams", [])) or "exec staff"
    return f"You are {name}, {persona['role']} ({teams}) at {org['org_id']}."


def _task_prompt(org: dict, probe: dict, task: str | None = None) -> str:
    frame = probe["context_frame"]
    line = FRAME_LINES[frame["kind"]].format(ref=frame.get("ref") or "")
    return "\n\n".join((
        _persona_line(org, probe["principal"]), line,
        f"Task: {task if task is not None else probe['task']}", SCAFFOLD,
    ))


def _witnesses(org: dict, event_meta: dict, event_index: int) -> list[str]:
    """Mirror memory-bench's event delivery visibility rule."""
    out = set(event_meta.get("participants", []))
    personas = org["entities"]["personas"]
    teams = {team["id"]: set(team.get("members", [])) for team in org["entities"]["teams"]}
    def active(persona: dict) -> bool:
        ids = {ev["event_id"]: i for i, ev in enumerate(org["events"])}
        joined = persona.get("joins_at")
        left = persona.get("leaves_at")
        return (joined is None or ids[joined] <= event_index) and (left is None or ids[left] > event_index)
    visibility = event_meta["visibility"]
    if visibility == "org_public":
        out.update(p["id"] for p in personas if active(p))
    elif visibility == "team_confidential":
        team = event_meta.get("team")
        if not team:
            raise ValueError(f"{event_meta['event_id']}: team_confidential event has no team")
        out.update(pid for pid in teams.get(team, set())
                   if active(next(p for p in personas if p["id"] == pid)))
    elif visibility != "private":
        raise ValueError(f"Unknown visibility: {visibility}")
    known = {p["id"] for p in personas}
    return sorted(out & known)


def _render_event(event: dict) -> str:
    participants = ", ".join(event.get("participants", []))
    when = f"[{event['sim_time']}] " if event.get("sim_time") else ""
    return (f"{when}{event['surface']} ({event['channel']}; participants: {participants})\n"
            f"{event['content']}")


def _assertions(probe: dict, twin: bool) -> list[dict]:
    if twin:
        return probe.get("counterfactual_probe", {}).get("assertions", [])
    return probe.get("assertions", [])


def _run_variant(org_dir: Path, probes: list[dict], output_dir: Path, variant: str,
                 backend_name: str, mode: str, run_id: str) -> dict[str, dict]:
    org = json.loads((org_dir / "org.json").read_text())
    stream = _read_jsonl(org_dir / "events.jsonl")
    event_index = {meta["event_id"]: i for i, meta in enumerate(org["events"])}
    if [e["event_id"] for e in stream] != [e["event_id"] for e in org["events"]]:
        raise ValueError(f"{org_dir}: events.jsonl does not align with org.json event index")
    by_event = defaultdict(list)
    for probe in probes:
        by_event[probe["inject_after_event"]].append(probe)
    last_event = max(event_index[p["inject_after_event"]] for p in probes)

    # Each paired stream gets a new org namespace, while event memories share
    # its org bank and carry recipient tags derived from the published witness index.
    org_namespace = f"memorybench-{org['org_id']}-{run_id}-{variant}"
    state = output_dir / "state" / run_id / f"{variant}.duckdb"
    state.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MEMORY_BACKEND"] = backend_name
    ledger = Ledger(state)
    service = MemoryService(ledger, backend(ledger), os.getenv("RETRIEVAL_MODE", "hybrid"))
    seen: set[str] = set()
    rows: dict[str, dict] = {}
    try:
        for i, event in enumerate(stream[:last_event + 1]):
            event_id = event["event_id"]
            meta = org["events"][i]
            recipients = _witnesses(org, meta, i)
            timestamp = (datetime(2026, 1, 1, tzinfo=timezone.utc)
                         + timedelta(minutes=i)).isoformat()
            if event_id not in seen:
                record = Record(
                    id=f"{org['org_id']}-{event_id}", scope="org", org=org_namespace,
                    text=_render_event(event),
                    source=f"{DATASET_URL}/tree/main/data/{org_dir.name}/events.jsonl#{event_id}",
                    valid_from=timestamp, recorded_at=timestamp,
                    metadata={"witnesses": recipients, "dataset_event_id": event_id,
                              "visibility": meta["visibility"], "team": meta.get("team")},
                )
                curator = Principal(org_namespace, "dataset-ingest", can_ingest=True)
                service.write(record, curator)
                seen.add(event_id)
            for probe in by_event.get(event_id, []):
                sync = service.sync()
                if sync["errors"]:
                    raise RuntimeError(f"Hindsight retain failed: {sync['errors']}")
                principal = Principal(org_namespace, probe["principal"])
                prompt = _task_prompt(org, probe)
                graph = build_graph(service, principal, as_of=timestamp,
                                    known_at=timestamp, read_only=True)
                result = invoke(graph, service, principal, prompt,
                                f"{variant}-{probe['probe_id']}",
                                f"{run_id}-{variant}-{probe['probe_id']}", mode=mode,
                                retrieval_query=probe["task"])
                rows[probe["probe_id"]] = {
                    "output": result["answer"],
                    "sources": [{"id": s["id"], "scope": s["scope"], "source": s["source"],
                                 "text": s["text"]} for s in result["sources"]],
                    "decisions": result["decisions"],
                }
        return rows
    finally:
        ledger.close()


def _write_report(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    review_path = path.parent / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else None
    if review and review.get("run_id") != payload["metadata"]["run_id"]:
        raise ValueError("Review run ID does not match result run ID")
    source_coverage = bool(review and review.get("metric") == "source_coverage")
    verdicts = {item["probe_id"]: item for item in review.get("verdicts", [])} if review else {}
    cases = payload["cases"]
    reviewed = [verdicts[case["probe_id"]] for case in cases if case["probe_id"] in verdicts]
    met_outputs = sum(item[side]["status"] == "met" for item in reviewed for side in ("original", "twin"))
    met_pairs = sum(all(item[side]["status"] == "met" for side in ("original", "twin")) for item in reviewed)
    output_total = 2 * len(reviewed)
    pair_total = len(reviewed)

    def status_badge(status: str) -> str:
        labels = ({"met": "✓ Opened", "missed": "✕ Missing"} if source_coverage
                  else {"met": "✓ Met", "missed": "✕ Missed"})
        label = labels.get(status, "? Not reviewed")
        css = status if status in ("met", "missed") else "unknown"
        return f'<span class="badge {css}">{label}</span>'

    cards = []
    sections = []
    for case in cases:
        probe_id = html.escape(case["probe_id"])
        verdict = verdicts.get(case["probe_id"], {})
        side_cards = []
        side_details = []
        for side, heading, assertion_key in (
            ("original", "Original", "assertions"),
            ("twin", "Twin", "counterfactual_assertions"),
        ):
            judgment = verdict.get(side, {})
            status = judgment.get("status", "unknown")
            reason = html.escape(judgment.get("reason", "No assistant review is available for this output."))
            criteria = "".join(
                f"<li>{html.escape(a['criterion'])}</li>" for a in case[assertion_key]
            )
            source_count = len(case[side]["sources"])
            sources = "".join(
                f"<details class=\"source\"><summary>{html.escape(s['id'])} · {html.escape(s['source'])}</summary>"
                f"<pre>{html.escape(s['text'])}</pre></details>" for s in case[side]["sources"]
            ) or "<p>No source passages were opened.</p>"
            side_cards.append(f"""<div class="side-summary"><div class="side-title"><strong>{heading}</strong>{status_badge(status)}</div>
<p>{reason}</p><small>{source_count} source passage{'s' if source_count != 1 else ''} opened</small></div>""")
            side_details.append(f"""<div class="side-evidence"><h3>{heading} {status_badge(status)}</h3>
<p><strong>Published criterion</strong></p><ul>{criteria}</ul>
<p><strong>Review</strong><br>{reason}</p>
<p><strong>Generated answer</strong></p><pre>{html.escape(case[side]['output'] or '')}</pre>
<details class="source-list"><summary>Opened source passages ({source_count})</summary>{sources}</details></div>""")
        cards.append(f"""<a class="result-card" href="#{probe_id}"><div class="card-heading">
<span class="archetype">{html.escape(case['archetype'])}</span><strong>{probe_id}</strong><span class="open-cue">View evidence ↗</span></div>
<div class="side-grid">{''.join(side_cards)}</div></a>""")
        sections.append(f"""<details class="case" id="{probe_id}"><summary><span class="archetype">{html.escape(case['archetype'])}</span>
{probe_id} · Evidence and generated answers</summary><div class="case-body">
<p class="muted">{html.escape(case['principal'])} · {html.escape(case['inject_after_event'])} · {html.escape(case['context_frame']['kind'])}</p>
<p><strong>Task</strong><br>{html.escape(case['task'])}</p>
<div class="evidence-grid">{''.join(side_details)}</div></div></details>""")
    meta = payload["metadata"]
    info = "".join(f"<dt>{html.escape(k.replace('_', ' ').title())}</dt><dd>{html.escape(str(v))}</dd>"
                   for k, v in meta.items() if k not in ("selection", "checksums"))
    selection = html.escape(json.dumps(meta["selection"], indent=2))
    checksums = html.escape(json.dumps(meta["checksums"], indent=2))
    score = (f"{met_pairs} of {pair_total}" if pair_total else "No")
    output_score = (f"{met_outputs} of {output_total}" if output_total else "No")
    coverage_note = (f"{len(cases) - pair_total} pair(s) have no review. "
                     if pair_total != len(cases) else "")
    headline = (f"{output_score} target passages opened" if source_coverage
                else f"{score} pairs met both criteria" if pair_total
                else "No answer review yet")
    summary = (f"{score} pairs opened the target on both sides. {coverage_note}"
               "This measures retrieval only; it does not score the generated deliverable."
               if source_coverage else
               f"{output_score} individual answers met their visible criterion. {coverage_note}"
               "Each pair asks the same task against an original and a changed memory stream."
               if pair_total else "Open the evidence sections to inspect outputs and source passages.")
    metric_label = "Target source coverage" if source_coverage else "Assistant review of published criteria"
    bar_met = met_outputs if source_coverage else met_pairs
    bar_total = output_total if source_coverage else pair_total
    bar_fraction = round(100 * bar_met / bar_total) if bar_total else 0
    legend_met = (f"{met_outputs} opened target" if source_coverage else f"{met_pairs} pair(s) met both")
    legend_miss = (f"{output_total - met_outputs} missed target" if source_coverage
                   else f"{pair_total - met_pairs} pair(s) missed at least one")
    caveat = ("Target event IDs were checked against opened passages. This says whether the needed "
              "source reached the graph; it does not establish answer correctness or an official benchmark score."
              if source_coverage else
              f"Green means this assistant review found the published criterion in the generated answer; red means it did not. This is a qualitative review of {output_total} answers, not the benchmark's official blind-judge score or a general accuracy estimate.")
    review_note = (f"<p class=\"caveat\">{html.escape(review['note'])}</p>"
                   if review and review.get("note") else "")
    title = f"memory-bench diagnostic · {html.escape(meta['org_id'])}"
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>
*{{box-sizing:border-box}}body{{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:36px 22px 70px;color:#172339;background:#f4f6fa}}
h1{{font-size:clamp(1.8rem,4vw,2.6rem);line-height:1.15;margin:10px 0 18px}}h2{{font-size:1.4rem;margin:0 0 12px}}h3{{font-size:1rem;margin:18px 0 8px}}
p{{margin:10px 0}}a{{color:#2457a8}}.eyebrow{{font-size:.78rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#4b6285}}
.hero,.result-card,.case,.technical{{background:#fff;border:1px solid #dce3ef;border-radius:16px;box-shadow:0 3px 14px #15243a0a}}
.hero{{padding:28px;margin-bottom:24px}}.hero-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;flex-wrap:wrap}}
.big-number{{font-size:clamp(2rem,6vw,4rem);font-weight:800;line-height:1.05;color:#8b2635;margin:8px 0}}.big-number.good{{color:#13724c}}
.hero-note{{max-width:680px;color:#4d5d70}}.mini-stat{{background:#f4f6fa;border-radius:12px;padding:13px 18px;min-width:180px}}
.mini-stat strong{{display:block;font-size:1.45rem}}.mini-stat span{{color:#53647b;font-size:.85rem}}
.bar{{height:22px;border-radius:99px;background:#eed4d8;overflow:hidden;margin:20px 0 8px}}
.bar-fill{{height:100%;background:#178256;border-radius:99px 0 0 99px}}.legend{{display:flex;gap:20px;flex-wrap:wrap;font-size:.9rem;color:#4d5d70}}
.legend b{{color:#172339}}.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;background:#178256}}
.dot.miss{{background:#ba4455}}.caveat{{font-size:.9rem;color:#53647b;border-top:1px solid #e2e7ef;padding-top:15px;margin-top:22px}}
.section-title{{margin:27px 0 12px}}.section-title p{{color:#53647b;margin:0}}
.card-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
.result-card{{display:block;padding:20px;text-decoration:none;color:inherit;transition:transform .15s,box-shadow .15s}}
.result-card:hover,.result-card:focus-visible{{transform:translateY(-2px);box-shadow:0 7px 22px #15243a20}}
.card-heading,.side-title{{display:flex;align-items:center;gap:10px;flex-wrap:wrap}}.card-heading{{margin-bottom:16px}}
.archetype{{display:inline-block;background:#eaf0ff;color:#275095;border-radius:7px;padding:3px 8px;font-size:.75rem;font-weight:800;letter-spacing:.05em}}
.open-cue{{margin-left:auto;font-size:.8rem;color:#4569a5}}.side-grid,.evidence-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
.side-summary{{border:1px solid #e4e9f0;border-radius:10px;padding:12px;background:#fafbfe}}
.side-summary p{{font-size:.9rem;line-height:1.45;margin:9px 0}}small,.muted,dt{{color:#5b6b80}}
.badge{{display:inline-block;border-radius:99px;padding:3px 9px;font-size:.76rem;font-weight:800;white-space:nowrap}}
.badge.met{{background:#daf4e7;color:#12633f}}.badge.missed{{background:#fbe2e6;color:#94283c}}.badge.unknown{{background:#e8edf3;color:#45556d}}
.case{{margin:12px 0;padding:0}}.case>summary{{cursor:pointer;padding:18px 22px;font-weight:700;list-style:none}}
.case>summary::-webkit-details-marker{{display:none}}.case>summary::after{{content:'＋';float:right;color:#577094}}
.case[open]>summary::after{{content:'−'}}.case-body{{padding:0 22px 22px;border-top:1px solid #e1e6ee}}
.side-evidence{{min-width:0}}.side-evidence h3{{display:flex;gap:8px;align-items:center}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5fa;padding:14px;border-radius:8px;max-height:360px;overflow:auto;font:13px/1.55 ui-monospace,monospace}}
.source-list,.source{{margin:8px 0;padding:10px;border:1px solid #e0e6ef;border-radius:8px}}summary{{cursor:pointer;overflow-wrap:anywhere}}
.technical{{margin-top:26px;padding:18px 22px}}.technical>summary{{font-weight:700}}dl{{display:grid;grid-template-columns:180px 1fr;gap:6px 18px}}
dd{{margin:0;overflow-wrap:anywhere}}code{{overflow-wrap:anywhere}}li{{margin:5px 0}}
@media(max-width:760px){{.card-grid,.evidence-grid{{grid-template-columns:1fr}}.hero{{padding:20px}}dl{{grid-template-columns:1fr;gap:2px}}dd{{margin-bottom:8px}}}}
</style></head><body>
<div class="eyebrow">Hindsight + LangGraph · memory-bench diagnostic</div><h1>Did the memory system use the right facts?</h1>
<section class="hero"><div class="hero-top"><div><div class="eyebrow">{metric_label}</div>
<div class="big-number{' good' if bar_total and bar_met == bar_total else ''}">{headline}</div>
<p class="hero-note">{summary}</p></div>
<div class="mini-stat"><strong>{len(cases)} pairs · {2 * len(cases)} outputs</strong><span>{html.escape(meta.get('retrieval_mode', 'hindsight'))} retrieval · {html.escape(meta['answer_model'])} answer</span></div></div>
<div class="bar" role="img" aria-label="{bar_met} of {bar_total} {html.escape(metric_label.lower())}"><div class="bar-fill" style="width:{bar_fraction}%"></div></div>
<div class="legend"><span><i class="dot"></i><b>{legend_met}</b></span><span><i class="dot miss"></i><b>{legend_miss}</b></span></div>
<p class="caveat"><strong>How to read this:</strong> {caveat}</p>{review_note}</section>
<div class="section-title"><h2>Pair comparison</h2><p>Select a card to inspect its task, published criteria, generated answers, and opened passages.</p></div>
<div class="card-grid">{''.join(cards)}</div>
<div class="section-title"><h2>Evidence</h2><p>The original and changed-memory twin are shown side by side.</p></div>
{''.join(sections)}
<details class="technical"><summary>Method, run details, and provenance</summary>
<p>This page reports a diagnostic run of published tasks. The released v1 anchors were produced with a withdrawn worker and are not transferable to this model. Published validity/floor values were used only to choose a small sample; no anchor or benchmark score was reused.</p><dl>{info}</dl>
<h3>Selection counts and rule</h3><pre>{selection}</pre><h3>Input checksums</h3><pre>{checksums}</pre>
<p>Dataset: <a href="{DATASET_URL}">{DATASET_URL}</a> · {LICENSE}</p>
<p>Bank mapping: one Hindsight organization bank per stream variant. Every published event is stored once with witness ACL tags calculated from that variant's <code>org.json</code>: private events go only to participants, team confidential events to active team members plus participants, and org public events to all active personas plus participants. Retrieval requires the principal's witness tag and local ledger check. The source stream contains no visibility field; visibility comes from the public witness index.</p>
<p>This small run uses the POC's current retrieval, pertinence gate, extractive or configured answer model, and answer path. It does not implement the upstream blind judge, official paired credit, or full-run protocol. Outputs and source passages are shown for inspection.</p>
</details><script>
document.querySelectorAll('.result-card').forEach(card => card.addEventListener('click', () => {{
  const detail = document.getElementById(card.getAttribute('href').slice(1));
  if (detail) detail.open = true;
}}));
</script></body></html>"""
    path.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="HF download's data/ directory (not copied into this repository)")
    parser.add_argument("--dataset-revision", default="unverified",
                        help="Hugging Face dataset commit SHA for the downloaded files")
    parser.add_argument("--org", default="org-00001")
    parser.add_argument("--per-archetype", type=int, default=1)
    parser.add_argument("--backend", choices=("hindsight", "duckdb"), default="hindsight")
    parser.add_argument("--mode", choices=("none", "personal", "federated", "gated"), default="gated")
    parser.add_argument("--answer-model", choices=("ollama_native", "llm", "extractive"),
                        default=os.getenv("ANSWER_MODEL", "ollama_native"))
    parser.add_argument("--model-max-tokens", type=int, default=512,
                        help="Limit generated answer length for the tiny local model")
    parser.add_argument("--run-id", help="Resume a prior run's source ledger and bank namespaces")
    parser.add_argument("--output", type=Path, default=Path("results/memory-bench"))
    args = parser.parse_args()
    if args.per_archetype < 1:
        parser.error("--per-archetype must be at least 1")
    if args.model_max_tokens < 1:
        parser.error("--model-max-tokens must be at least 1")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("--run-id must contain only letters, digits, hyphens, and underscores")
    os.environ["ANSWER_MODEL"] = args.answer_model
    os.environ["MODEL_MAX_TOKENS"] = str(args.model_max_tokens)
    org_dir = args.data / args.org
    twin_dir = args.data / f"{args.org}-twin"
    probes, selection = _selection(org_dir, args.per_archetype)
    run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                             + "-" + uuid.uuid4().hex[:8])
    original = _run_variant(org_dir, probes, args.output, "original", args.backend, args.mode, run_id)
    twin = _run_variant(twin_dir, probes, args.output, "twin", args.backend, args.mode, run_id)
    checksums = {}
    for folder in (org_dir, twin_dir):
        for name in ("events.jsonl", "org.json"):
            path = folder / name
            checksums[f"{folder.name}/{name}"] = _sha256(path)
    checksums[f"{org_dir.name}/probes.jsonl"] = _sha256(org_dir / "probes.jsonl")
    checksums[f"{org_dir.name}/g3-report.json"] = _sha256(org_dir / "g3-report.json")
    cases = []
    for probe in probes:
        cases.append({
            "probe_id": probe["probe_id"], "cluster_id": probe["cluster_id"],
            "archetype": probe["archetype"], "principal": probe["principal"],
            "inject_after_event": probe["inject_after_event"], "task": probe["task"],
            "context_frame": probe["context_frame"], "targets": probe.get("targets", []),
            "must_not_use": probe.get("must_not_use", []),
            "sampling_anchor": probe["sampling_anchor"],
            "assertions": _assertions(probe, False),
            "counterfactual_assertions": _assertions(probe, True),
            "original": original[probe["probe_id"]], "twin": twin[probe["probe_id"]],
        })
    metadata = {
        "dataset": "notmehul/memory-bench v1",
        "dataset_url": DATASET_URL,
        "dataset_revision": args.dataset_revision,
        "license": LICENSE,
        "org_id": args.org,
        "run_id": run_id,
        "backend": args.backend,
        "retrieval_mode": os.getenv("RETRIEVAL_MODE", "hybrid"),
        "mode": args.mode,
        "model": os.getenv("MODEL_NAME", "qwen2.5:7b"),
        "answer_model": os.getenv("ANSWER_MODEL", "extractive"),
        "model_max_tokens": args.model_max_tokens,
        "gate_model": os.getenv("GATE_MODEL", "rules"),
        "selection": selection,
        "checksums": checksums,
        "caveat": "Diagnostic output only; alternate worker, no official paired scoring.",
        "bank_mapping": "Shared org bank per stream variant; witness ACL tags and local source filtering enforce per-principal event delivery.",
        "temporal_cutoff": "Each probe is evaluated at its inject_after_event timestamp on simulated 2026-01-01 UTC.",
    }
    payload = {"metadata": metadata, "cases": cases}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    _write_report(payload, args.output / "report.html")
    print(json.dumps({"run_id": run_id, "cases": len(cases),
                      "paired_outputs": len(cases) * 2,
                      "results": str(args.output / "results.json"),
                      "report": str(args.output / "report.html")}, indent=2))


if __name__ == "__main__":
    main()
