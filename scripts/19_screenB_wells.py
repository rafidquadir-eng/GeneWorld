"""ScreenB wells and primary wells (author instruction 2026-10-21), frozen before the freeze.

ScreenB (MitoCheck validation screen, independent siRNAs; 28 plates on EBI S3) is evaluated
after the freeze under the global preprocessing only, outside the gates, as a secondary
independent-reagent test (it has no negative-control wells, so section 1 cannot be applied).

* `splits/screenB_wells.tsv`: every quality-controlled screenB library well with a gene symbol
  and every positive (transfection) control; plate_split = "screenB"; gene_split from the frozen
  gene split (controls = "control").
* `splits/amend02_screenB_primary_wells.tsv`: screenB library wells whose gene carries a
  MitoCheck reproducible-phenotype term on **screenA training plates** (training-plate
  annotation rows only; the screenB well's own annotation is not used). This is the screenB
  analogue of the frozen H1 primary-well rule (screenB siRNAs never appear in screenA, so the
  rule is applied at gene level).
* `manifests/amend02_screenB.json`: counts and SHA-256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LEVEL = "multiple replicates of reagent"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    s3 = json.loads((REPO / "data" / "meta" / "s3_plate_map.json").read_text())
    b = pd.read_csv(REPO / "data" / "annotations" / "idr0013-screenB-annotation.csv",
                    low_memory=False)
    b["qc"] = b["Quality Control"].astype(str).eq("True")
    b["control"] = b["Control Type"].fillna("")
    keep = b.qc & b.Plate.isin(set(s3)) & (
        ((b.control == "") & b["Gene Symbol"].notna()) | (b.control == "positive control"))
    w = b[keep].copy()
    gs = pd.read_csv(REPO / "splits" / "gene_split.tsv", sep="\t")
    gmap = dict(zip(gs.gene, gs.split, strict=True))
    w["gene_split"] = w["Gene Symbol"].map(gmap).fillna("unmapped")
    w.loc[w.control != "", "gene_split"] = "control"
    w["plate_split"] = "screenB"
    w["series"] = w.Plate.str.replace(r"_\d+$", "", regex=True)
    w["uuid"] = w.Plate.map(lambda p: s3[p]["uuid"])
    w["control"] = w.control.replace("", pd.NA)
    cols = ["Plate", "Well", "Plate_Well", "Gene Symbol", "siRNA Identifier", "control", "series",
            "plate_split", "gene_split", "uuid"]
    f_w = REPO / "splits" / "screenB_wells.tsv"
    w[cols].sort_values(["Plate", "Well"]).to_csv(f_w, sep="\t", index=False)

    # gene-level reproducible terms on screenA training plates only
    a = pd.read_csv(REPO / "data" / "annotations" / "idr0013-screenA-annotation.csv.gz",
                    low_memory=False)
    ps = pd.read_csv(REPO / "splits" / "plate_split.tsv", sep="\t")
    trp = set(ps.plate[ps.split == "train"])
    a = a[a.Plate.isin(trp) & a["Control Type"].isna() & a["Gene Symbol"].notna()
          & (a["Phenotype Annotation Level"] == LEVEL)]
    tcols = [c for c in a.columns if c.startswith("Phenotype ") and "Term Name" in c]
    has = set(a.loc[a[tcols].notna().any(axis=1), "Gene Symbol"])
    lib = w[w.control.isna()]
    prim = lib[lib["Gene Symbol"].isin(has)][["Plate", "Well", "Gene Symbol",
                                               "siRNA Identifier", "gene_split"]]
    f_p = REPO / "splits" / "amend02_screenB_primary_wells.tsv"
    prim.sort_values(["Plate", "Well"]).to_csv(f_p, sep="\t", index=False)
    out = {"instruction": "author 2026-10-21: screenB after the freeze, global preprocessing "
                          "only, outside the gates; siRNA token masked for screenB siRNAs",
           "plates": int(w.Plate.nunique()), "wells": int(len(w)),
           "library_wells": int(len(lib)), "positive_controls": int(w.control.notna().sum()),
           "library_wells_by_gene_split": {k: int(v) for k, v in
                                           lib.gene_split.value_counts().items()},
           "primary_wells": int(len(prim)),
           "primary_by_gene_split": {k: int(v) for k, v in prim.gene_split.value_counts().items()},
           "sha256": {f_w.name: sha(f_w), f_p.name: sha(f_p)}}
    (REPO / "manifests" / "amend02_screenB.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
