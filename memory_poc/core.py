from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import httpx

SCOPES = ("personal", "org", "event")
STOP = set("a an the is are was were and or to of in on at for with my me i what how do does did please tell about now current latest our this it should we use".split())


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc(value: str) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamps require a timezone")
    return dt.astimezone(timezone.utc).isoformat()


def digest(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:24]


def terms(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def relevant_overlap(query: str, hint: str, scope: str, event: str | None = None) -> set[str]:
    overlap = terms(query) & terms(hint)
    # The event name alone establishes scope, not topical pertinence.
    return overlap - terms(event or "") if scope == "event" else overlap


@dataclass(frozen=True)
class Principal:
    org: str
    user: str
    events: tuple[str, ...] = ()
    can_ingest: bool = False
    can_promote: bool = False

    def bank(self, scope: str, event: str | None = None) -> str:
        if scope not in SCOPES:
            raise ValueError("Unknown scope")
        if scope == "event" and (not event or event not in self.events):
            raise PermissionError("Event not authorized for this principal")
        subject = self.user if scope == "personal" else event if scope == "event" else "shared"
        return f"{os.getenv('BANK_PREFIX', 'poc')}-{scope}-{digest(self.org, str(subject))}"


@dataclass
class Record:
    id: str
    scope: str
    org: str
    text: str
    source: str
    valid_from: str
    recorded_at: str
    user: str | None = None
    event: str | None = None
    supersedes: str | None = None
    status: str = "published"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.valid_from, self.recorded_at = utc(self.valid_from), utc(self.recorded_at)
        if self.scope not in SCOPES or self.status not in ("published", "candidate"):
            raise ValueError("Invalid scope/status")
        if self.scope == "personal" and not self.user:
            raise ValueError("Personal memory requires a user")
        if self.scope == "event" and not self.event:
            raise ValueError("Event memory requires an event")

    def bank(self) -> str:
        return Principal(self.org, self.user or "", (self.event,) if self.event else ()).bank(self.scope, self.event)

    def authorized(self, p: Principal, event: str | None) -> bool:
        return self.org == p.org and (
            self.scope == "org" or self.scope == "personal" and self.user == p.user
            or self.scope == "event" and self.event == event and event in p.events
        )

    def tags(self) -> list[str]:
        witness_tags = [f"witness:{pid}" for pid in self.metadata.get("witnesses", [])]
        return [f"org:{self.org}", f"scope:{self.scope}"] + witness_tags + (
            [f"user:{self.user}"] if self.scope == "personal" else
            [f"event:{self.event}"] if self.scope == "event" else []
        )

    def memory_hint(self) -> str:
        return self.metadata.get("memory_summary", re.sub(r"\b\d+(?:[.,]\d+)*\b", "[see source]", self.text))

    def memory_text(self) -> str:
        # Exact quantities are resolved from the immutable source passage, never this hint.
        return f"{self.memory_hint()}\nSource pointer: {self.source}; source_id={self.id}"


class Ledger:
    """Single-process DuckDB source registry and retry queue; not a vector DB."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS records (bank VARCHAR, id VARCHAR, body VARCHAR, synced BOOLEAN DEFAULT false, error VARCHAR, PRIMARY KEY(bank,id))")
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions (id VARCHAR PRIMARY KEY, owner VARCHAR)")

    def close(self):
        self.db.close()

    def put(self, r: Record):
        # IDs are immutable revisions. Corrections use a new ID plus supersedes.
        old = self.get(r.bank(), r.id)
        body = json.dumps(asdict(r), sort_keys=True)
        if old and json.dumps(asdict(old), sort_keys=True) != body:
            raise ValueError(f"Immutable document changed: {r.id}; create a revision")
        if r.supersedes:
            prev = self.get(r.bank(), r.supersedes)
            if not prev or r.valid_from < prev.valid_from or r.id == prev.id:
                raise ValueError("Superseded revision must exist in the same bank and precede this one")
        self.db.execute("INSERT INTO records(bank,id,body) VALUES (?,?,?) ON CONFLICT DO NOTHING", [r.bank(), r.id, body])

    def get(self, bank: str, doc: str) -> Record | None:
        row = self.db.execute("SELECT body FROM records WHERE bank=? AND id=?", [bank, doc]).fetchone()
        return Record(**json.loads(row[0])) if row else None

    def rows(self, bank: str | None = None, pending: bool = False) -> list[Record]:
        sql, args = "SELECT body FROM records WHERE 1=1", []
        if bank:
            sql += " AND bank=?"
            args.append(bank)
        if pending:
            sql += " AND NOT synced"
        return [Record(**json.loads(r[0])) for r in self.db.execute(sql + " ORDER BY id", args).fetchall()]

    def synced_ids(self, bank: str) -> set[str]:
        return {row[0] for row in self.db.execute(
            "SELECT id FROM records WHERE bank=? AND synced", [bank]).fetchall()}

    def mark(self, r: Record, error: str | None = None):
        self.db.execute("UPDATE records SET synced=?, error=? WHERE bank=? AND id=?", [error is None, error, r.bank(), r.id])

    def bind_session(self, p: Principal, thread: str, event: str | None) -> str:
        # Checkpoint IDs cannot be reused by another principal or event.
        sid = digest(p.org, p.user, thread)
        owner = json.dumps([p.org, p.user, event])
        row = self.db.execute("SELECT owner FROM sessions WHERE id=?", [sid]).fetchone()
        if row and row[0] != owner:
            raise PermissionError("Start a new thread when changing event context")
        self.db.execute("INSERT INTO sessions VALUES (?,?) ON CONFLICT DO NOTHING", [sid, owner])
        return sid

    def promote(self, bank: str, doc: str, p: Principal) -> Record:
        if not p.can_promote or bank != p.bank("org"):
            raise PermissionError("Organizational promotion requires a curator")
        r = self.get(bank, doc)
        if not r or r.scope != "org" or r.status != "candidate":
            raise ValueError("Not an organizational candidate")
        r.status = "published"
        r.metadata["approved_by"] = p.user
        r.metadata["approved_at"] = now()
        self.db.execute("UPDATE records SET body=?, synced=false WHERE bank=? AND id=?", [json.dumps(asdict(r), sort_keys=True), bank, doc])
        return r


class DuckMemory:
    """Deterministic lexical TEST DOUBLE. Does not implement Hindsight semantics."""
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def retain(self, r: Record):
        pass

    def recall(self, bank: str, query: str, tags: list[str], timestamp: str) -> list[str]:
        q = terms(query)
        records = [r for r in self.ledger.rows(bank) if r.status == "published" and set(tags) <= set(r.tags())]
        records.sort(key=lambda r: (-len(q & terms(r.memory_text())), r.id))
        # Like nearest-neighbor retrieval without a score floor, return some
        # distractors too. The gated arm must explicitly reject them.
        return [r.id for r in records[:12]]


class HindsightMemory:
    """Minimal documented HTTP adapter. No fallback to DuckDB search on errors."""
    def __init__(self, base_url: str, api_key: str = "", transport=None):
        self.client = httpx.Client(base_url=base_url, timeout=180, transport=transport,
                                   headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
        self.banks: set[str] = set()

    def retain(self, r: Record):
        if r.bank() not in self.banks:
            response = self.client.put(f"/v1/default/banks/{r.bank()}", json={
                "retain_mission": f"Retain source-grounded {r.scope} facts. Preserve dates, uncertainty and attribution.",
                "enable_observations": False,  # Keep the first experiment inexpensive.
            })
            response.raise_for_status()
            self.banks.add(r.bank())
        response = self.client.post(f"/v1/default/banks/{r.bank()}/memories", json={
            "async": False,
            "items": [{"content": r.memory_text(), "document_id": r.id, "timestamp": r.valid_from,
                       "context": f"{r.scope}; source={r.source}; reported={r.recorded_at}",
                       "tags": r.tags(), "metadata": {"source_id": r.id, "source": r.source,
                           "valid_from": r.valid_from, "recorded_at": r.recorded_at}}],
        })
        response.raise_for_status()
        body = response.json()
        if not body.get("success") or body.get("async"):
            raise RuntimeError("Expected a successful synchronous retain")

    def recall(self, bank: str, query: str, tags: list[str], timestamp: str) -> list[str]:
        response = self.client.post(f"/v1/default/banks/{bank}/memories/recall", json={
            "query": query, "tags": tags, "tags_match": "all_strict", "types": ["world", "experience"],
            "query_timestamp": timestamp, "budget": "low", "max_tokens": 1800,
        })
        if response.status_code == 404:
            return []  # A new user/event has no bank yet.
        response.raise_for_status()
        ids = []
        for hit in response.json()["results"]:
            doc = (hit.get("metadata") or {}).get("source_id") or hit.get("document_id")
            if not doc:
                raise RuntimeError("Recall result lacks source_id/document_id; verify the Hindsight API version")
            if doc not in ids:
                ids.append(doc)
        return ids


def _rank_source_lines(query: str, records: list[Record], limit: int) -> list[str]:
    """Rank authorized source records by their best matching line."""
    query_terms = terms(query)
    if not query_terms:
        return []
    lines = [(record.id, terms(line)) for record in records for line in record.text.splitlines()]
    frequencies = Counter(term for _, words in lines for term in words)
    count = len(lines)
    scores: dict[str, float] = {}
    for doc, words in lines:
        overlap = query_terms & words
        if not overlap:
            continue
        score = sum(math.log1p((count + 1) / (frequencies[term] + 1)) for term in overlap)
        scores[doc] = max(score, scores.get(doc, 0.0))
    return sorted(scores, key=lambda doc: (-scores[doc], doc))[:limit]


def backend(ledger: Ledger):
    mode = os.getenv("MEMORY_BACKEND", "duckdb")
    if mode == "duckdb":
        return DuckMemory(ledger)
    if mode == "hindsight":
        return HindsightMemory(os.getenv("HINDSIGHT_URL", "http://localhost:8888"), os.getenv("HINDSIGHT_API_KEY", ""))
    raise ValueError("MEMORY_BACKEND must be duckdb or hindsight")


class MemoryService:
    def __init__(self, ledger: Ledger, memory, retrieval_mode: str = "hindsight"):
        if retrieval_mode not in ("hindsight", "hybrid"):
            raise ValueError("RETRIEVAL_MODE must be hindsight or hybrid")
        self.ledger, self.memory = ledger, memory
        self.retrieval_mode = retrieval_mode

    def sync(self) -> dict:
        done, errors = 0, []
        for r in self.ledger.rows(pending=True):
            if r.status != "published":
                continue
            try:
                self.memory.retain(r)
                self.ledger.mark(r)
                done += 1
            except Exception as e:
                # Do not persist remote error bodies, credentials, or source text in errors.
                self.ledger.mark(r, type(e).__name__)
                errors.append({"id": r.id, "error": type(e).__name__})
        return {"synced": done, "errors": errors}

    def write(self, r: Record, p: Principal, event: str | None = None):
        if not r.authorized(p, event):
            raise PermissionError("Cannot write outside principal scope")
        self.ledger.put(r)  # Durable queue first; the graph can finish if Hindsight is down.

    def search(self, p: Principal, scope: str, query: str, event: str | None = None,
               as_of: str | None = None, known_at: str | None = None, history: bool = False,
               pertinent: bool = True) -> list[dict]:
        bank = p.bank(scope, event)
        as_of, known_at = utc(as_of or now()), utc(known_at or now())
        tags = [f"org:{p.org}", f"scope:{scope}"]
        # memory-bench uses one shared organization bank with an explicit
        # per-principal witness ACL on each event. Filter remotely and again
        # against the local source ledger before any passage reaches the graph.
        if scope == "org" and p.org.startswith("memorybench-"):
            tags.append(f"witness:{p.user}")
        if scope == "personal": tags.append(f"user:{p.user}")
        if scope == "event": tags.append(f"event:{event}")
        ids = self.memory.recall(bank, query, tags, as_of)
        # Hindsight temporal retrieval ranks, rather than strictly filters. Enforce cutoffs here.
        eligible = {r.id: r for r in self.ledger.rows(bank)
                    if r.status == "published" and r.authorized(p, event)
                    and ("witnesses" not in r.metadata or p.user in r.metadata["witnesses"])
                    and r.valid_from <= as_of and r.recorded_at <= known_at}
        superseded = {r.supersedes for r in eligible.values()} if not history else set()
        hybrid = self.retrieval_mode == "hybrid" and isinstance(self.memory, HindsightMemory)
        synced = self.ledger.synced_ids(bank) if hybrid else set()
        local_ids = (_rank_source_lines(query, [r for r in eligible.values()
                                                if r.id not in superseded and r.id in synced], 3)
                     if hybrid else [])
        candidate_ids = list(dict.fromkeys(local_ids + ids))[:5] if hybrid else ids
        result = []
        for doc in candidate_ids:
            r = eligible.get(doc)
            if not r or doc in superseded:
                continue
            overlap = relevant_overlap(query, r.memory_hint(), scope, event)
            # Transparent cheap pertinence baseline; replace with a calibrated classifier later.
            if pertinent and not overlap:
                continue
            result.append({**asdict(r), "text": r.memory_text(), "hint": r.memory_hint(),
                           "reason": ("source-line match" if doc in local_ids else "Hindsight recall")
                           + "; lexical overlap: " + ", ".join(sorted(overlap))})
        return result if hybrid else result[:3]  # Per-bank quota; don't compare raw scores across banks.

    def read_source(self, p: Principal, scope: str, doc: str, event: str | None,
                    as_of: str | None = None, known_at: str | None = None, history: bool = False) -> dict:
        bank = p.bank(scope, event)
        r = self.ledger.get(bank, doc)
        cutoff, known = utc(as_of or now()), utc(known_at or now())
        if (not r or not r.authorized(p, event) or r.status != "published"
                or ("witnesses" in r.metadata and p.user not in r.metadata["witnesses"])):
            raise PermissionError("Source unavailable in this scope")
        if r.valid_from > cutoff or r.recorded_at > known:
            raise PermissionError("Source outside requested time")
        if not history and any(x.supersedes == doc and x.status == "published" and
                              x.valid_from <= cutoff and x.recorded_at <= known for x in self.ledger.rows(bank)):
            raise ValueError("Stale source pointer; retrieve the current revision")
        return {**asdict(r), "sha256": hashlib.sha256(r.text.encode()).hexdigest()}
