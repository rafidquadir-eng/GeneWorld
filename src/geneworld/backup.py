"""Backups to the Lambda persistent filesystem with SHA-256 verification.

Every checkpoint, latent shard and annotation table that GeneWorld produces is copied to
``~/LambdaFilesystem/geneworld_backups/<relative path>`` and the copy's SHA-256 is compared
with the original's. A ledger (``reports/BACKUP_LEDGER.tsv``) records path, bytes, hash,
destination and time, so a later wipe can be checked against it (TiJepa PLAN_v3 rule 5:
a milestone that produces a checkpoint is not complete until it exists in a second place
with a verified hash).
"""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST_ROOT = Path.home() / "LambdaFilesystem" / "geneworld_backups"
LEDGER = REPO / "reports" / "BACKUP_LEDGER.tsv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def backup(path: Path, dest_root: Path = DEST_ROOT, repo: Path = REPO) -> dict:
    """Copy one file to the persistent filesystem and verify it; raise on mismatch."""
    path = Path(path).resolve()
    rel = path.relative_to(repo) if path.is_relative_to(repo) else Path(path.name)
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_hash = sha256(path)
    if not (dest.exists() and dest.stat().st_size == path.stat().st_size
            and sha256(dest) == src_hash):
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(path, tmp)
        with open(tmp, "rb+") as fh:
            fh.flush()
        tmp.replace(dest)
    dst_hash = sha256(dest)
    if dst_hash != src_hash:
        raise OSError(f"backup of {path} failed verification")
    row = {"path": str(rel), "bytes": path.stat().st_size, "sha256": src_hash,
           "dest": str(dest), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    new = not LEDGER.exists()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        if new:
            fh.write("\t".join(row) + "\n")
        fh.write("\t".join(str(v) for v in row.values()) + "\n")
    return row


def backup_tree(root: Path, patterns: tuple[str, ...] = ("*",)) -> list[dict]:
    out = []
    for pat in patterns:
        for p in sorted(Path(root).rglob(pat)):
            if p.is_file() and not p.name.endswith(".part"):
                out.append(backup(p))
    return out
