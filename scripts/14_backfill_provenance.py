"""Backfill backbone provenance into the stage-1 results files (amendment 02 section 9).

Stage-1 results recorded backbone records with revision/sha256 = None. This writes the pinned
revision and the weights SHA-256 (verified on disk against manifests/models.yaml) into every
backbone record, adds a backbone record to files that had only a backbone name, and adds the
predictor checkpoint SHA-256 to training summaries. No number, label or design_commit is
changed; each touched file gets a `provenance_backfill` note.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.data import BACKBONE_KEY  # noqa: E402
from geneworld.eval.provenance import sha256_file, verified_backbone_record  # noqa: E402

NOTE = ("backbone revision and weights SHA-256 (and predictor checkpoint SHA-256 where "
        "missing) added 2026-10-20 under amendment 02 section 9; no result value changed")


def walk(x):
    n = 0
    if isinstance(x, dict):
        if x.get("kind") == "backbone" and x.get("key") in BACKBONE_KEY.values():
            v = verified_backbone_record(x["key"])
            if x.get("revision") != v["revision"] or x.get("sha256") != v["sha256"]:
                x["revision"], x["sha256"] = v["revision"], v["sha256"]
                n += 1
        for val in x.values():
            n += walk(val)
    elif isinstance(x, list):
        for val in x:
            n += walk(val)
    return n


def main() -> int:
    touched = []
    for f in sorted((REPO / "results").rglob("*.json")):
        d = json.loads(f.read_text())
        if not isinstance(d, dict):
            continue
        n = walk(d)
        if f.parent.name == "controls" and isinstance(d.get("backbone"), str) \
                and "backbone_record" not in d:
            d["backbone_record"] = verified_backbone_record(BACKBONE_KEY[d["backbone"]])
            n += 1
        if f.name == "leakage.json" and "latents_backbone_record" not in d:
            d["latents_backbone_record"] = verified_backbone_record("vjepa2_vitl")
            n += 1
        if f.parent.name == "train" and "name" in d and "checkpoint_sha256" not in d:
            ck = REPO / "checkpoints" / "predictors" / f"{d['name']}.pt"
            a = torch.load(ck, map_location="cpu", weights_only=False)["args"]
            d["checkpoint"] = ck.name
            d["checkpoint_sha256"] = sha256_file(ck)
            d["backbone"] = verified_backbone_record(BACKBONE_KEY[a["backbone"]])
            n += 1
        if n:
            d["provenance_backfill"] = NOTE
            f.write_text(json.dumps(d, indent=1))
            touched.append((str(f.relative_to(REPO)), n))
    for t in touched:
        print(*t)
    print(f"{len(touched)} files updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
