import json
from pathlib import Path

import yaml

from .core import MemoryService, Principal, Record


def ingest(service: MemoryService, root: Path, p: Principal) -> int:
    if not p.can_ingest:
        raise PermissionError("Document ingestion requires an ingestion principal")
    records = []
    for line in (root / "records.jsonl").read_text().splitlines():
        if line.strip():
            r = Record(**json.loads(line))
            relative = Path(r.source.split("#", 1)[0]).relative_to("data")
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or r.text not in path.read_text():
                raise ValueError(f"Source pointer does not resolve to the passage: {r.id}")
            records.append(r)
    for path in sorted((root / "knowledge").rglob("*.md")):
        if path.name in ("index.md", "log.md"): continue
        raw = path.read_text()
        if not raw.startswith("---\n"):
            raise ValueError(f"Missing OKF frontmatter: {path}")
        _, header, body = raw.split("---", 2)
        meta = yaml.safe_load(header)
        if not isinstance(meta, dict) or not meta.get("type"):
            raise ValueError(f"Missing OKF type: {path}")
        # POC ingestion contract adds these fields on top of OKF's minimal spec.
        records.append(Record(id=meta["id"], scope="org", org=p.org, text=body.strip(),
                              source=f"data/knowledge/{path.relative_to(root / 'knowledge')}#passage",
                              valid_from=meta["valid_from"], recorded_at=meta["recorded_at"],
                              supersedes=meta.get("supersedes"),
                              metadata=meta))
    # Fixture importer is admin-only, but still validates every scope and event.
    for r in records:
        owner = Principal(p.org, r.user or p.user, p.events, can_ingest=True)
        if r.status != "published" or not r.authorized(owner, r.event):
            raise PermissionError("Fixture exceeds ingestion principal scope")
        service.write(r, owner, r.event)
    return len(records)
