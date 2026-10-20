"""Stage-2 well selection (amendment 02 section 7), frozen before any stage-2 embedding.

Every library series (LTxxxx layout) of screenA. Per series, one well layout used on all of
its replicate plates (same-wells design): every negative and positive control, every library
well whose siRNA carries a reproducible-phenotype term, and seeded random library wells up to
`PER_PLATE` wells per plate. For the 50 stage-1 series the stage-1 wells are kept and the
layout is extended to the same budget. Quality control is per plate (a well failing QC on a
plate is skipped there). Splits are the frozen whole-screen splits; nothing is re-split.

Outputs
  splits/stage2_layout.tsv       the per-series layout (all splits; test plates of every
                                 series use it at the freeze, section 8)
  splits/stage2_wells.tsv        wells to embed now: train and validation plates only, on S3,
                                 not already embedded in stage 1
  splits/stage2_test_wells.tsv   test-plate wells of the same layout (embedded after freeze)
`PER_PLATE` is chosen as the smallest budget (step 10) giving >= 60,000 stage-2 wells.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEED = 20261020
TARGET = 60_000
LEVEL = "multiple replicates of reagent"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    a = pd.read_csv(REPO / "data" / "annotations" / "idr0013-screenA-annotation.csv.gz",
                    low_memory=False)
    a["qc"] = a["Quality Control"].astype(str).eq("True")
    a["control"] = a["Control Type"].fillna("")
    a["series"] = a.Plate.str.slice(0, 6)
    tcols = [c for c in a.columns if c.startswith("Phenotype ") and "Term Name" in c]
    at = (a["Phenotype Annotation Level"] == LEVEL).to_numpy()
    a["phenotypic"] = [ok and any(isinstance(v, str) for v in r)
                       for r, ok in zip(a[tcols].itertuples(index=False), at, strict=True)]
    ps = pd.read_csv(REPO / "splits" / "plate_split.tsv", sep="\t")
    psplit = dict(zip(ps.plate, ps.split, strict=True))
    a["plate_split"] = a.Plate.map(psplit)
    s3 = json.loads((REPO / "data" / "meta" / "s3_plate_map.json").read_text())
    st1 = pd.read_csv(REPO / "splits" / "stage1_wells.tsv", sep="\t")
    st1_wells = st1.groupby("series").Well.apply(set).to_dict()
    embedded = set()
    for f in glob.glob(str(REPO / "data" / "latents" / "vjepa2" / "*.npz")):
        for w in np.load(f)["wells"]:
            embedded.add((Path(f).stem, str(w)))

    usable = a[a.control.isin(["negative control", "positive control"])
               | ((a.control == "") & a["Gene Symbol"].notna())]
    lay_base, rest = {}, {}
    rng = np.random.default_rng(SEED)
    for sr, w in usable.groupby("series"):
        lay = w.drop_duplicates("Well").set_index("Well")
        must = set(lay.index[lay.control != ""]) | set(lay.index[lay.phenotypic.astype(bool)])
        must |= st1_wells.get(sr, set())
        others = sorted(set(lay.index[lay.control == ""]) - must)
        lay_base[sr] = must
        rest[sr] = [others[i] for i in rng.permutation(len(others))]

    def build(budget):
        rows = []
        for sr, must in lay_base.items():
            k = max(0, budget - len(must))
            rows.append((sr, sorted(must | set(rest[sr][:k]))))
        lay = pd.DataFrame([(sr, w) for sr, ws in rows for w in ws], columns=["series", "Well"])
        cand = usable[usable.qc & usable.Plate.isin(set(s3))].merge(lay, on=["series", "Well"])
        new = cand[[(p, w) not in embedded for p, w in zip(cand.Plate, cand.Well, strict=True)]]
        return lay, cand, new

    for budget in range(90, 400, 10):
        lay, cand, new = build(budget)
        n_now = int(new.plate_split.isin(["train", "val"]).sum())
        if n_now >= TARGET:
            break
    cols = ["Plate", "Well", "Plate_Well", "Gene Symbol", "siRNA Identifier", "control",
            "series", "plate_split", "phenotypic"]
    now = new[new.plate_split.isin(["train", "val"])][cols].copy()
    test = cand[cand.plate_split == "test"][cols].copy()
    for df in (now, test):
        df["uuid"] = df.Plate.map(lambda p: s3[p]["uuid"])
        df["control"] = df.control.replace("", np.nan)
    gs = pd.read_csv(REPO / "splits" / "gene_split.tsv", sep="\t")
    gmap = dict(zip(gs.gene, gs.split, strict=True))
    for df in (now, test):
        df["gene_split"] = df["Gene Symbol"].map(gmap).fillna("control")
        df.loc[df.control.notna(), "gene_split"] = "control"
    out = REPO / "splits"
    lay.to_csv(out / "stage2_layout.tsv", sep="\t", index=False)
    now.sort_values(["Plate", "Well"]).to_csv(out / "stage2_wells.tsv", sep="\t", index=False)
    test.sort_values(["Plate", "Well"]).to_csv(out / "stage2_test_wells.tsv", sep="\t",
                                               index=False)
    summ = {"design_commit": "dd57079", "seed": SEED, "per_plate_budget": budget,
            "series": int(lay.series.nunique()),
            "stage2_wells_train_val": int(len(now)),
            "stage2_by_split": {k: int(v) for k, v in now.plate_split.value_counts().items()},
            "stage2_plates": int(now.Plate.nunique()),
            "stage2_controls": {k: int(v) for k, v in now.control.value_counts().items()},
            "stage2_phenotypic_wells": int(now.phenotypic.sum()),
            "test_wells_same_layout": int(len(test)),
            "test_wells_already_embedded_in_stage1": int(sum(
                (p, w) in embedded for p, w in zip(test.Plate, test.Well, strict=True))),
            "screenA_plates_not_on_s3": sorted(set(a.Plate) - set(s3)),
            "sha256": {f: sha(out / f) for f in ("stage2_layout.tsv", "stage2_wells.tsv",
                                                  "stage2_test_wells.tsv")}}
    (out / "stage2_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
