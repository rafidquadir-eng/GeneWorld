"""Restore on the new instance (completion plan v2, section 3, step 2).

Every path in `reports/BACKUP_LEDGER.tsv` is copied from the persistent filesystem to where the
code expects it (`hf_cache/...` -> `~/.cache/huggingface/hub/...`, everything else under the
repository). The expected hash is the path's latest ledger row. Each file is copied to a
`.part` file, the copy is hashed, and it is moved into place only if the hash matches. A file
already at the target (e.g. tracked in git) is hashed and never overwritten. Writes
`results/a2/restore_audit.json`; exit status 1 on any missing source or mismatch, except the
ledger rows listed in NOT_NEEDED (author decision 2026-10-23). Every ledger row is reconciled:
restored (or present) with a matching hash, superseded by a later row for the same path, or
without a backup copy and not needed. Ledger lines starting with "#" are notes and are skipped.
`--reconcile-only` recomputes the reconciliation from an existing restore_audit.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import backup  # noqa: E402

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
AUDIT = REPO / "results" / "a2" / "restore_audit.json"
NOT_NEEDED_LABEL = ("ledger rows without a backup copy: stage-1 smoke tests, unreferenced, "
                    "not needed")
#: author decision 2026-10-23: no code, results file or report references them, and the old
#: instance's 10-22 audit shows they were no longer on its disk
NOT_NEEDED = {"checkpoints/predictors/smoke_l.pt", "checkpoints/predictors/smoke_n.pt",
              "checkpoints/predictors/smoke_s.pt"}


def ledger_rows() -> list[dict]:
    with open(backup.LEDGER) as fh:
        return [r for r in csv.DictReader(fh, delimiter="\t")
                if not r["path"].startswith("#")]


def reconcile(rows: list[dict], status: dict[str, str]) -> dict:
    """Classify every ledger row; `status` = restore status per path."""
    latest = {r["path"]: r["sha256"] for r in rows}
    out = {"restored_or_present_matching_hash": 0, "superseded_by_later_row": 0,
           NOT_NEEDED_LABEL: [], "unaccounted": []}
    for r in rows:
        p, ok = r["path"], status.get(r["path"]) in ("restored", "present_match")
        if p in NOT_NEEDED and status.get(p) == "missing_source":
            out[NOT_NEEDED_LABEL].append({"path": p, "bytes": int(r["bytes"]),
                                          "sha256": r["sha256"], "ledgered_utc": r["utc"]})
        elif ok and r["sha256"] == latest[p]:
            out["restored_or_present_matching_hash"] += 1
        elif ok:
            out["superseded_by_later_row"] += 1
        else:
            out["unaccounted"].append(r)
    out["ledger_rows"] = len(rows)
    out["every_row_accounted_for"] = not out["unaccounted"]
    return out


def target(rel: str) -> Path:
    if rel.startswith("hf_cache/"):
        return HF_HUB / rel[len("hf_cache/"):]
    return REPO / rel


def restore(item: tuple[str, str]) -> dict:
    rel, want = item
    src, dst = backup.DEST_ROOT / rel, target(rel)
    if dst.exists():
        got = backup.sha256(dst)
        return {"path": rel, "status": "present_match" if got == want else "present_mismatch",
                "sha256": got}
    if not src.exists():
        return {"path": rel, "status": "missing_source"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    got = backup.sha256(tmp)
    if got != want:
        tmp.unlink()
        return {"path": rel, "status": "copy_mismatch", "sha256": got}
    os.replace(tmp, dst)
    return {"path": rel, "status": "restored", "bytes": dst.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconcile-only", action="store_true")
    args = ap.parse_args()
    rows = ledger_rows()
    if args.reconcile_only:
        out = json.loads(AUDIT.read_text())
        status = {r["path"]: r["status"] for r in out["problems"]}
        for r in rows:
            status.setdefault(r["path"], "restored")     # not a problem in the audit
        out["reconciliation"] = reconcile(rows, status)
        out["problems"] = [r for r in out["problems"] if r["path"] not in NOT_NEEDED]
        AUDIT.write_text(json.dumps(out, indent=1))
        print(json.dumps(out["reconciliation"], indent=1))
        return 0 if out["reconciliation"]["every_row_accounted_for"] else 1
    t0 = time.time()
    latest, hashes = {}, {}
    for row in rows:
        latest[row["path"]] = row["sha256"]
        hashes.setdefault(row["path"], []).append(row["sha256"])
    with ThreadPoolExecutor(16) as ex:
        res = list(ex.map(restore, sorted(latest.items())))
    counts: dict[str, int] = {}
    for r in res:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    bad = [r for r in res if r["status"] not in ("restored", "present_match")
           and r["path"] not in NOT_NEEDED]
    superseded = sorted(p for p, h in hashes.items() if len(set(h)) > 1)
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source_root": str(backup.DEST_ROOT), "ledger_rows": sum(map(len, hashes.values())),
           "paths": len(latest), "counts": counts,
           "bytes_restored": int(sum(r.get("bytes", 0) for r in res)),
           "paths_with_superseded_rows": superseded,
           "problems": bad, "minutes": (time.time() - t0) / 60,
           "reconciliation": reconcile(rows, {r["path"]: r["status"] for r in res})}
    AUDIT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "paths_with_superseded_rows"},
                     indent=1))
    return 1 if bad or not out["reconciliation"]["every_row_accounted_for"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
