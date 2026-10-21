import json

import httpx
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from memory_poc.core import HindsightMemory, MemoryService, Principal, Record, now
from memory_poc.graph import build_graph, invoke, make_tools


def ids(rows):
    return {r["id"] for r in rows}


@pytest.mark.parametrize("as_of,known_at,want", [
    (None, None, "riverside-v3"),
    ("2026-02-01T10:00:00Z", None, "riverside-v1"),
    ("2026-02-01T15:00:00Z", "2026-02-01T16:00:00Z", "riverside-v2"),
    ("2026-02-01T15:00:00Z", "2026-02-02T09:00:00Z", "riverside-v3"),
])
def test_two_temporal_axes(service, alice, as_of, known_at, want):
    got = service.search(alice, "event", "Riverside occupancy capacity", "riverside", as_of, known_at)
    assert ids(got) == {want}


def test_history_preserves_revisions(service, alice):
    assert ids(service.search(alice, "event", "Riverside occupancy capacity", "riverside", history=True)) == {
        "riverside-v1", "riverside-v2", "riverside-v3"}


def test_scope_is_enforced_before_remote_recall(service, alice):
    with pytest.raises(PermissionError):
        service.search(alice, "event", "shelter", "hilltown")
    assert ids(service.search(alice, "personal", "briefings", pertinent=False)) == {"alice-format"}
    eve = Principal("other-org", "eve")
    assert service.search(eve, "org", "shelter") == []


def test_published_stream_witness_acl_covers_recall_and_source_open(service):
    org = "memorybench-fixture"
    witness = Principal(org, "persona:anouk")
    outsider = Principal(org, "persona:diego")
    record = Record(id="E-0001", scope="org", org=org, text="Private contract deadline is Friday.",
                    source="dataset/E-0001", valid_from="2026-01-01T00:00:00Z",
                    recorded_at="2026-01-01T00:00:00Z",
                    metadata={"witnesses": [witness.user]})
    service.write(record, witness)
    service.sync()
    assert ids(service.search(witness, "org", "contract deadline")) == {record.id}
    assert service.search(outsider, "org", "contract deadline") == []
    with pytest.raises(PermissionError, match="unavailable"):
        service.read_source(outsider, "org", record.id, None)
    assert service.read_source(witness, "org", record.id, None)["text"] == record.text


def test_event_tools_require_an_authorized_active_event(service, alice):
    assert "event" not in make_tools(service, alice)
    assert "remember_event" not in make_tools(service, alice)
    with pytest.raises(PermissionError):
        make_tools(service, alice, "hilltown")
    assert "remember_event" in make_tools(service, alice, "riverside")


def test_ground_numbers_from_source_not_memory_hint(service, alice):
    hint = service.search(alice, "event", "Riverside occupancy capacity", "riverside")[0]
    assert "16 people" not in hint["text"]
    graph = build_graph(service, alice, event="riverside", read_only=True)
    r = invoke(graph, service, alice, "Riverside occupancy capacity", "numbers", "n1", event="riverside")
    assert "16 people" in r["answer"] and "30 places" in r["answer"]
    source = next(s for s in r["sources"] if s["id"] == "riverside-v3")
    assert source["text"] in r["answer"]
    assert source["source"] in r["answer"] and len(source["sha256"]) == 64


def test_stale_pointer_is_rejected(service, alice):
    with pytest.raises(ValueError, match="Stale"):
        service.read_source(alice, "event", "riverside-v1", "riverside")
    assert service.read_source(alice, "event", "riverside-v1", "riverside", history=True)["text"]


def test_no_supporting_source_abstains(service, alice):
    g = build_graph(service, alice, read_only=True)
    r = invoke(g, service, alice, "chess tournament winner", "unknown", "u")
    assert r["sources"] == [] and "no supporting source" in r["answer"]


def test_auto_writes_are_selective_and_replay_safe(service, alice):
    g = build_graph(service, alice, event="riverside")
    q = "I prefer concise maps. Event update: the west road reopened. Org learning: verify maps with the duty officer."
    a = invoke(g, service, alice, q, "write", "same-turn", event="riverside")
    b = invoke(g, service, alice, q, "write", "same-turn", event="riverside")
    assert len(a["writes"]) == 3 and a["writes"] == b["writes"]
    records = [r for r in service.ledger.rows() if r.id in a["writes"]]
    assert len(records) == 3
    assert next(r for r in records if r.scope == "org").status == "candidate"
    assert next(r for r in records if r.scope == "event").metadata["trust"] == "user_report"
    assert "duty officer" not in next(r for r in records if r.scope == "personal").text
    assert "duty officer" not in next(r for r in records if r.scope == "event").text
    assert invoke(g, service, alice, "thanks", "write", "thanks", event="riverside")["writes"] == []


def test_org_candidate_promotion_is_explicit(service, alice):
    tools = make_tools(service, alice)
    doc = tools["propose_organization_learning"].invoke({"note": "Unique kumquat procedure: use a checklist."})
    assert doc not in ids(service.search(alice, "org", "kumquat"))
    with pytest.raises(PermissionError): service.ledger.promote(alice.bank("org"), doc, alice)
    curator = Principal("relief-demo", "curator", can_promote=True)
    service.ledger.promote(curator.bank("org"), doc, curator)
    service.sync()
    assert doc in ids(service.search(alice, "org", "kumquat"))


def test_outbox_survives_remote_failure(service, alice):
    class Down:
        def retain(self, record): raise httpx.ConnectError("unreachable")
    broken = MemoryService(service.ledger, Down())
    t = now()
    r = Record("queued", "personal", alice.org, "I prefer map summaries", "chat:test", t, t, user=alice.user)
    broken.write(r, alice)
    assert broken.sync()["errors"] == [{"id": "queued", "error": "ConnectError"}]
    assert "queued" in {r.id for r in service.ledger.rows(pending=True)}
    assert service.sync()["synced"] == 1
    assert "queued" not in {r.id for r in service.ledger.rows(pending=True)}


def test_checkpoint_thread_binding_and_reset(service, alice, tmp_path):
    with SqliteSaver.from_conn_string(str(tmp_path / "sessions.sqlite")) as cp:
        g = build_graph(service, alice, cp, event="riverside", read_only=True)
        a = invoke(g, service, alice, "Riverside occupancy", "shared", "a", event="riverside")
        b = invoke(g, service, alice, "chess tournament", "shared", "b", event="riverside")
        assert a["sources"] and not b["sources"]  # No stale evidence from checkpoint.
        assert len(b["messages"]) == 4
        bob = Principal(alice.org, "bob", ("hilltown",))
        h = build_graph(service, bob, cp, read_only=True)
        c = invoke(h, service, bob, "briefing format", "shared", "b")
        assert len(c["messages"]) == 2 and "BLUEBIRD" in c["answer"]
    with pytest.raises(PermissionError): service.ledger.bind_session(alice, "shared", None)


def test_http_adapter_contract(service, alice):
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append((request.method, request.url.path, body))
        if request.method == "PUT": return httpx.Response(200, json={})
        if request.url.path.endswith("/recall"):
            return httpx.Response(200, json={"results": [{"text": "not trusted as source", "metadata": {"source_id": "riverside-v3"}}]})
        return httpx.Response(200, json={"success": True, "async": False})
    remote = HindsightMemory("http://test", transport=httpx.MockTransport(handler))
    doc = service.ledger.get(alice.bank("event", "riverside"), "riverside-v3")
    remote.retain(doc)
    assert "16 people" not in requests[1][2]["items"][0]["content"]
    assert requests[1][1].endswith("/memories")
    assert requests[1][2]["items"][0]["metadata"]["source_id"] == doc.id
    svc = MemoryService(service.ledger, remote)
    assert ids(svc.search(alice, "event", "occupancy", "riverside")) == {doc.id}
    assert requests[-1][2]["tags_match"] == "all_strict"
    assert set(requests[-1][2]["tags"]) == {"org:relief-demo", "scope:event", "event:riverside"}


def test_live_adapter_does_not_fall_back_to_local_search(service, alice):
    remote = HindsightMemory("http://test", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"results": []})))
    svc = MemoryService(service.ledger, remote)
    assert svc.search(alice, "event", "occupancy", "riverside") == []


def test_hybrid_source_search_obeys_witness_time_and_sync(service):
    org = "memorybench-hybrid-fixture"
    witness = Principal(org, "persona:anouk")
    outsider = Principal(org, "persona:diego")
    for doc, text, owner, when in (
        ("target", "Growth runbooks need review every sixty days.", witness.user, "2026-01-01T00:00:00Z"),
        ("private", "Growth runbooks have a private review rule.", outsider.user, "2026-01-01T00:00:00Z"),
        ("future", "Growth runbooks need review tomorrow.", witness.user, "2027-01-01T00:00:00Z"),
    ):
        service.write(Record(doc, "org", org, text, f"dataset/{doc}", when, when,
                             metadata={"witnesses": [owner]}), witness)
    service.sync()
    service.write(Record("pending", "org", org, "Growth runbooks review is pending.",
                         "dataset/pending", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
                         metadata={"witnesses": [witness.user]}), witness)
    remote = HindsightMemory("http://test", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"results": []})))
    hybrid = MemoryService(service.ledger, remote, retrieval_mode="hybrid")
    assert ids(hybrid.search(witness, "org", "runbooks review", as_of="2026-06-01T00:00:00Z")) == {"target"}
    assert "target" not in ids(hybrid.search(outsider, "org", "runbooks review",
                                             as_of="2026-06-01T00:00:00Z"))


def test_hybrid_does_not_hide_hindsight_failure(service, alice):
    remote = HindsightMemory("http://test", transport=httpx.MockTransport(
        lambda request: httpx.Response(503, json={"error": "unavailable"})))
    hybrid = MemoryService(service.ledger, remote, retrieval_mode="hybrid")
    with pytest.raises(httpx.HTTPStatusError):
        hybrid.search(alice, "event", "Riverside occupancy", "riverside")


def test_unregistered_remote_pointer_is_not_grounded(service, alice):
    remote = HindsightMemory("http://test", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"results": [{"document_id": "unknown-source"}]})))
    assert MemoryService(service.ledger, remote).search(alice, "event", "occupancy", "riverside") == []


def test_optional_semantic_capture_only_accepts_verbatim_user_spans(service, alice, monkeypatch):
    import memory_poc.graph as graph_module
    class FakeModel:
        def with_structured_output(self, *args, **kwargs): return self
        def invoke(self, messages):
            return graph_module.WriteSelection(personal=["Use concise maps for future briefings."],
                event=["The west road reopened.", "A fabricated statement that was never said."],
                org_candidates=[])
    monkeypatch.setenv("WRITE_MODEL", "llm")
    monkeypatch.setattr(graph_module, "model", lambda: FakeModel())
    q = "Use concise maps for future briefings. The west road reopened."
    r = invoke(build_graph(service, alice, event="riverside"), service, alice, q, "semantic", "1", event="riverside")
    assert len(r["writes"]) == 2
    assert all(x.text in q for x in service.ledger.rows() if x.id in r["writes"])


def test_optional_llm_gate_cannot_invent_source_ids(service, alice, monkeypatch):
    import memory_poc.graph as graph_module
    class FakeModel:
        def with_structured_output(self, *args, **kwargs): return self
        def invoke(self, messages): return graph_module.Selection(ids=["invented-source"])
    monkeypatch.setenv("GATE_MODEL", "llm")
    monkeypatch.setattr(graph_module, "model", lambda: FakeModel())
    r = invoke(build_graph(service, alice, read_only=True), service, alice, "briefing format", "gate", "1")
    assert r["sources"] == [] and "no supporting source" in r["answer"]


def test_optional_answer_model_receives_opened_sources(service, alice, monkeypatch):
    import memory_poc.graph as graph_module
    from langchain_core.messages import AIMessage

    class FakeModel:
        def invoke(self, messages):
            payload = json.loads(messages[1][1])
            assert any(s["id"] == "riverside-v3" and "16 people" in s["text"]
                       for s in payload["sources"])
            return AIMessage(content="The correction reports 16 people [riverside-v3].")

    monkeypatch.setenv("ANSWER_MODEL", "llm")
    monkeypatch.setattr(graph_module, "model", lambda: FakeModel())
    result = invoke(build_graph(service, alice, event="riverside", read_only=True),
                    service, alice, "Riverside occupancy capacity", "model", "1", event="riverside")
    assert result["answer"] == "The correction reports 16 people [riverside-v3]."
