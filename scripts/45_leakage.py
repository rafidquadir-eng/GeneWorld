"""Leakage control (design A.7(6)).

1. Movie duplicates: every validation/test-plate well's TiJepa fingerprint (three 25 x 40
   block-reduced, standardised frames of the level-1 movie) is correlated with every
   training-plate well's; r >= 0.98 is a match, r >= 0.90 is reported. Exact frame-0 SHA-1
   duplicates are counted over all wells.
2. Reagent sharing across the gene split: siRNAs whose sense sequence targets genes in
   different gene-split parts (train vs val/test), and genes that share an identical siRNA
   sequence with a gene in another part. Reported for the gene and the complex split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.data import load  # noqa: E402
from geneworld.preprocess.fingerprint import MATCH_R, REPORT_R  # noqa: E402


def main() -> int:
    lat = load("vjepa2")
    m = lat.meta
    fp = lat.fingerprint.astype(np.float32)
    fp = fp - fp.mean(1, keepdims=True)
    fp /= np.linalg.norm(fp, axis=1, keepdims=True) + 1e-9
    tr = np.flatnonzero((m.plate_split == "train").to_numpy())
    ev = np.flatnonzero((m.plate_split != "train").to_numpy())
    best = np.full(len(ev), -1.0, np.float32)
    arg = np.zeros(len(ev), int)
    for i in range(0, len(ev), 1024):
        c = fp[ev[i:i + 1024]] @ fp[tr].T
        best[i:i + 1024] = c.max(1)
        arg[i:i + 1024] = tr[c.argmax(1)]
    matches = [{"eval": f"{m.plate[e]}:{m.well[e]}", "train": f"{m.plate[t]}:{m.well[t]}",
                "r": float(r)} for e, t, r in zip(ev, arg, best, strict=True) if r >= REPORT_R]
    from geneworld.eval.provenance import verified_backbone_record
    out = {"latents_backbone_record": verified_backbone_record("vjepa2_vitl"),
           "n_eval_wells": int(len(ev)), "n_train_wells": int(len(tr)),
           "max_r": float(best.max()), "median_max_r": float(np.median(best)),
           "n_match_r>=0.98": int((best >= MATCH_R).sum()),
           "n_report_r>=0.90": int((best >= REPORT_R).sum()), "reported": matches[:200]}
    try:
        import glob
        sha = np.concatenate([np.load(f)["frame0_sha1"] for f in
                              sorted(glob.glob(str(REPO / "data/latents/vjepa2/*.npz")))])
        out["frame0_sha1_duplicates"] = int(len(sha) - len(set(sha)))
    except KeyError:
        out["frame0_sha1_duplicates"] = None

    a = pd.read_csv(REPO / "data" / "annotations" / "idr0013-screenA-annotation.csv.gz",
                    usecols=["Gene Symbol", "Sense Sequence", "Control Type"], low_memory=False)
    a = a[a["Control Type"].isna() & a["Gene Symbol"].notna() & a["Sense Sequence"].notna()]
    a = a.drop_duplicates(["Gene Symbol", "Sense Sequence"])
    reag = {}
    for name, f in (("gene", "gene_split.tsv"), ("complex", "gene_split_complex.tsv")):
        sp = pd.read_csv(REPO / "splits" / f, sep="\t")
        smap = dict(zip(sp.gene, sp.split, strict=False))
        a["part"] = a["Gene Symbol"].map(smap)
        multi = a.groupby("Sense Sequence").filter(lambda g: g["Gene Symbol"].nunique() > 1)
        cross = multi.groupby("Sense Sequence").filter(lambda g: g.part.nunique() > 1)
        reag[name] = {"sirnas_targeting_>1_gene": int(multi["Sense Sequence"].nunique()),
                      "sirnas_crossing_split_parts": int(cross["Sense Sequence"].nunique()),
                      "genes_involved_in_crossing": sorted(set(cross["Gene Symbol"]))[:100]}
    out["shared_reagents"] = reag
    (REPO / "results" / "controls").mkdir(parents=True, exist_ok=True)
    (REPO / "results" / "controls" / "leakage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "reported"}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
