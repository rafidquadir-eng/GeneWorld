"""Backup audit before migration (completion plan v2, section 3, step 1).

For every latent shard, cache file, completed run summary and checkpoint (plus the annotation,
action-encoding, reference and metadata tables that are not in git): SHA-256 of the local file,
a ledger row with that hash, and the copy on the persistent filesystem re-hashed. Files missing
from the backup or the ledger, or whose copy does not match, are backed up with
`geneworld.backup.backup` (copy + verified hash + ledger row), one at a time. Writes
`results/a2/backup_audit.json`.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import backup, cache  # noqa: E402

GROUPS = {
    "latent_shards": ["data/latents/*/*.npz", "data/latents_s2/*/*.npz",
                      "data/latents_static/*/*.npz"],
    "cache_files": ["data/cache/*/*"],
    "run_summaries": ["results/a2/train/*.json", "results/train/*.json"],
    "checkpoints": ["checkpoints/a2/*.pt", "checkpoints/predictors/*.pt",
                    "checkpoints/backbones/*"],
    "tables_not_in_git": ["data/annotations/*", "data/actions/*.npz", "data/actions/*.tsv",
                          "data/actions/raw/*", "data/ref/*", "data/meta/*"],
}


def hashes(rel: str) -> tuple[str, str, str | None, int]:
    p = REPO / rel
    d = backup.DEST_ROOT / rel
    src = backup.sha256(p)
    dst = backup.sha256(d) if d.exists() else None
    return rel, src, dst, p.stat().st_size


def main() -> int:
    t0 = time.time()
    cache.check_shard_manifest()     # stray or missing latent shards: refuse, listing them
    files = {}
    for g, pats in GROUPS.items():
        for pat in pats:
            for p in sorted(REPO.glob(pat)):
                if p.is_file() and not p.name.endswith((".part", ".tmp", ".lock")):
                    files[str(p.relative_to(REPO))] = g
    ledger = {}
    with open(backup.LEDGER) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ledger.setdefault(row["path"], set()).add(row["sha256"])
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(hashes, sorted(files), chunksize=4))
    fixed, ok = [], 0
    for rel, src, dst, _ in res:
        if dst == src and src in ledger.get(rel, set()):
            ok += 1
            continue
        backup.backup(REPO / rel)          # copies if needed, verifies, appends ledger row
        fixed.append({"path": rel, "had_backup": dst is not None,
                      "backup_matched": dst == src, "in_ledger": src in ledger.get(rel, set())})
    # final check after fixes
    ledger = {}
    with open(backup.LEDGER) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ledger.setdefault(row["path"], set()).add(row["sha256"])
    sizes = {rel: n for rel, _, _, n in res}
    src_of = {rel: s for rel, s, _, _ in res}
    bad = [rel for rel in files if src_of[rel] not in ledger.get(rel, set())
           or backup.sha256(backup.DEST_ROOT / rel) != src_of[rel]] if fixed else []
    groups = {}
    for rel, g in files.items():
        e = groups.setdefault(g, {"files": 0, "bytes": 0})
        e["files"] += 1
        e["bytes"] += sizes[rel]
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "dest_root": str(backup.DEST_ROOT), "files": len(files),
           "bytes": int(sum(sizes.values())), "groups": groups,
           "already_backed_up_and_ledgered": ok, "backed_up_or_ledgered_now": len(fixed),
           "fixed": fixed, "problems_after_fix": bad, "minutes": (time.time() - t0) / 60}
    (REPO / "results" / "a2" / "backup_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "fixed"}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
