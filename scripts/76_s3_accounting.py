"""S3 accounting of wells lost to missing frames (completion plan v2 S3, v4 D3; protocol
`missing_frames_rule`), for test, screenB or the rehearsal. Descriptive only.

Inputs: the split's embed list and the merged embed failure list
(scripts/78_parallel_embed.py). Each failed well is re-requested frame by frame as in S2 (HEAD,
3 attempts, back-off) to record its missing frames and whether all of them are >= 88 (F9).
Reported: wells, plates (whole plates lost listed), H1 primary wells lost (test:
splits/amend02_h1_primary_wells.tsv; screenB: splits/amend02_screenB_primary_wells.tsv),
primary-gene wells lost
(splits/amend02_h2_primary_genes.tsv), and, among library wells, per class (mitotic_delay,
polylobed, binuclear, any reproducible) a two-sided Fisher exact test of lost vs embedded x
class vs not, Benjamini-Hochberg over the 4. Class of a well = the classes its siRNA carries on
screenA training plates (screenB: its gene's classes on screenA training plates), read from
training-plate annotation rows only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.annotations import well_annotations  # noqa: E402
from geneworld.io.mitocheck import N_FRAMES, chunk_url  # noqa: E402

CLASSES = ["mitotic_delay", "polylobed", "binuclear"]
PRIMARY = {"test": "splits/amend02_h1_primary_wells.tsv",
           "screenB": "splits/amend02_screenB_primary_wells.tsv",
           "rehearsal": "splits/amend02_h1_primary_wells.tsv"}
OUT = {"test": "results/a2/missing_frames_test.json",
       "screenB": "results/a2/missing_frames_screenB.json",
       "rehearsal": "results/a2/rehearsal/missing_frames_rehearsal.json"}
LATE = 88


def bh(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out.tolist()


def fisher_tables(lost: np.ndarray, cls: list[list[str]]) -> dict:
    from scipy.stats import fisher_exact
    rows = {}
    for k in CLASSES + ["any_reproducible"]:
        has = np.array([bool(c) if k == "any_reproducible" else k in c for c in cls])
        t = [[int((lost & has).sum()), int((lost & ~has).sum())],
             [int((~lost & has).sum()), int((~lost & ~has).sum())]]
        rows[k] = {"table_lost_embedded_x_class_not": t,
                   "p_two_sided": float(fisher_exact(t)[1]) if lost.any() else 1.0}
    for k, q in zip(rows, bh([r["p_two_sided"] for r in rows.values()]), strict=True):
        rows[k]["q_bh"] = q
    return rows


def training_classes(by: str) -> dict:
    """siRNA (or gene) -> classes on screenA training plates."""
    ann = well_annotations("A").reset_index()
    ps = pd.read_csv(REPO / "splits" / "plate_split.tsv", sep="\t")
    tr = ann[ann.Plate.isin(set(ps.loc[ps.split == "train", "plate"]))]
    key = tr["siRNA Identifier"].astype(str) if by == "sirna" else tr["Gene Symbol"]
    return tr.groupby(key).classes.agg(lambda s: sorted(set().union(*s))).to_dict()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(OUT))
    ap.add_argument("--wells", required=True)
    ap.add_argument("--failures", required=True)
    args = ap.parse_args()
    wells = pd.read_csv(REPO / args.wells, sep="\t")
    fails = json.loads((REPO / args.failures).read_text())
    failed = {tuple(x.split(":")[:2]) for x in fails}
    keys = list(zip(wells.Plate, wells.Well.astype(str), strict=True))
    lost = np.array([k in failed for k in keys])
    spec = importlib.util.spec_from_file_location(
        "mf", REPO / "scripts" / "72_missing_frames.py")
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    fw = wells[lost].reset_index(drop=True)
    urls = [(i, t, chunk_url(r.uuid, str(r.Well), 1, t)) for i, r in enumerate(fw.itertuples())
            for t in range(N_FRAMES)]
    with ThreadPoolExecutor(64) as ex:
        codes = list(ex.map(lambda u: mf.head(u[2]), urls))
    miss = {i: [] for i in range(len(fw))}
    for (i, t, _), c in zip(urls, codes, strict=True):
        if c != 200:
            miss[i].append(t)
    prim = pd.read_csv(REPO / PRIMARY[args.split], sep="\t")
    prim = set(zip(prim.Plate, prim.Well.astype(str), strict=True))
    h2 = set(pd.read_csv(REPO / "splits" / "amend02_h2_primary_genes.tsv", sep="\t").gene)
    by = "gene" if args.split == "screenB" else "sirna"
    tc = training_classes(by)
    sid = wells["siRNA Identifier"].astype(str) if by == "sirna" else wells["Gene Symbol"]
    cls = [tc.get(s, []) for s in sid]
    lib = wells.control.fillna("").eq("").to_numpy()
    per_plate = wells.groupby("Plate").size()
    lost_pp = wells[lost].groupby("Plate").size()
    out = {"split": args.split, "rule": "manifests/amend02_protocol.json missing_frames_rule",
           "n_wells": int(len(wells)), "n_plates": int(wells.Plate.nunique()),
           "wells_lost": int(lost.sum()),
           "plates_with_losses": int(len(lost_pp)),
           "whole_plates_lost": sorted(p for p, n in lost_pp.items() if n == per_plate[p]),
           "h1_primary_wells_lost": int(sum(k in prim for k, x in zip(keys, lost, strict=True)
                                            if x)),
           "primary_gene_wells_lost": int(wells[lost]["Gene Symbol"].isin(h2).sum()),
           "lost_only_frames_ge_88": int(sum(bool(m) and min(m) >= LATE
                                              for m in miss.values())),
           "frames_recovered_on_recheck_wells": int(sum(not m for m in miss.values())),
           "fisher_library_wells": fisher_tables(lost[lib], [c for c, x in zip(cls, lib,
                                                                              strict=True)
                                                              if x]),
           "class_source": f"{by}-level classes on screenA training plates",
           "wells_lost_detail": [{"plate": r.Plate, "well": str(r.Well),
                                  "missing_frames": miss[i]}
                                 for i, r in enumerate(fw.itertuples())]}
    p = REPO / OUT[args.split]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"{args.split}: {out['wells_lost']} of {out['n_wells']} wells lost "
          f"({out['lost_only_frames_ge_88']} only frames >= 88)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
