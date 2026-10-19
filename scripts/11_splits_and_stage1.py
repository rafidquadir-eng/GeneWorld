"""Gene and plate splits (all of MitoCheck screenA) and the stratified stage-1 well subset.

Splits are defined once, over the whole screen, before any movie is read, and frozen in
`data/splits/` (committed). Seeded and deterministic.

* Gene split (H2/H3): every library gene goes to train / val / test (70 / 10 / 20), stratified
  by its dominant reproducible phenotype class (or none) and by MitoCheck's validated-mitotic-hit
  flag. Positive-control genes (on every plate) are always train.
* Plate split (H1): every plate goes to train / val / test (70 / 10 / 20), stratified by plate
  series (the LTxxxx prefix).
* Stage 1: a stratified subset of ~15,000 quality-controlled wells. Plates are sampled
  (seeded, stratified by split and series); on each sampled plate every negative and positive
  control is taken, every well whose siRNA has a reproducible phenotype, and random other wells
  up to the per-plate budget. The per-class counts are reported so that every phenotype class is
  shown to be covered.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ANN = REPO / "data" / "annotations"
OUT = REPO / "splits"
SEED = 20261019
TARGET_WELLS = 15000
PER_PLATE = 100
N_PLATES = 150

# reproducible phenotype term -> planning/stratification class
CLASS_OF = {
    "mitosis delayed phenotype": "mitotic_delay", "metaphase delayed phenotype": "mitotic_delay",
    "prometaphase delayed phenotype": "mitotic_delay",
    "polylobed nuclear phenotype": "polylobed", "binuclear cell phenotype": "binuclear",
    "cell death phenotype": "cell_death", "graped micronucleus phenotype": "grape",
    "increased nucleus size phenotype": "large",
    "increased cell movement speed": "migration", "increased cell movement distance": "migration",
    "proliferating cells": "proliferation",
    "abnormal chromosome segregation phenotype": "segregation",
    "abnormal nucleus shape phenotype": "nuclear_shape",
    "increased variability of nuclear shape in population": "nuclear_shape",
    "abnormal cell cycle phenotype": "cell_cycle",
    "mitotic chromosome condensation phenotype": "condensation",
    "absence of mitotic chromosome decondensation phenotype": "condensation",
}


def phenotype_terms(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in df.columns if c.startswith("Phenotype ") and "Term Name" in c]
    return df[cols].apply(lambda r: sorted({CLASS_OF[v] for v in r if isinstance(v, str)
                                            and v in CLASS_OF}), axis=1)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    rng = np.random.default_rng(SEED)
    a = pd.read_csv(ANN / "idr0013-screenA-annotation.csv.gz", low_memory=False)
    a["qc"] = a["Quality Control"].astype(str).eq("True")
    a["control"] = a["Control Type"].fillna("")
    a["classes"] = phenotype_terms(a)
    a["series"] = a["Plate"].str.slice(0, 6)

    # ---------------- gene split
    lib = a[(a.control == "") & a["Gene Symbol"].notna()]
    pos_genes = set(a.loc[a.control == "positive control", "Gene Symbol"].dropna())
    g = lib.groupby("Gene Symbol").agg(
        classes=("classes", lambda s: sorted({c for x in s for c in x})),
        validated=("Validated Mitotic Hit At Gene Level", lambda s: (s == "yes").any()))
    g["stratum"] = g["classes"].apply(lambda c: c[0] if c else "none") + "|" + \
        g["validated"].astype(str)
    split = {}
    for _, grp in g.groupby("stratum"):
        genes = sorted(grp.index)
        rng.shuffle(genes)
        n = len(genes)
        n_te, n_va = int(round(0.2 * n)), int(round(0.1 * n))
        for i, gene in enumerate(genes):
            split[gene] = "test" if i < n_te else "val" if i < n_te + n_va else "train"
    for gene in pos_genes:
        split[gene] = "train"
    gene_split = pd.DataFrame({"gene": list(split), "split": list(split.values())})
    gene_split = gene_split.merge(g[["classes", "validated"]], left_on="gene",
                                  right_index=True, how="left")

    # ---------------- plate split
    plates = sorted(a["Plate"].unique())
    pdf = pd.DataFrame({"plate": plates, "series": [p[:6] for p in plates]})
    # systematic assignment over plates sorted by series (and a seeded shuffle inside each
    # series), so every part of the screen is represented in every split
    order_p = []
    for _, grp in pdf.groupby("series"):
        ps = sorted(grp.plate)
        rng.shuffle(ps)
        order_p += ps
    offset = int(rng.integers(10))
    psplit = {}
    for i, p in enumerate(order_p):
        r = (i + offset) % 10
        psplit[p] = "test" if r < 2 else "val" if r == 2 else "train"
    pdf["split"] = pdf.plate.map(psplit)

    # ---------------- stage-1 wells
    s3 = json.loads((REPO / "data" / "meta" / "s3_plate_map.json").read_text())
    avail = [p for p in plates if p in s3]
    n_plates = N_PLATES
    order = []
    for sp in ("train", "val", "test"):
        ps = [p for p in avail if psplit[p] == sp]
        rng.shuffle(ps)
        k = int(round(n_plates * {"train": 0.7, "val": 0.1, "test": 0.2}[sp]))
        order += ps[:k]
    rows = []
    # quality-controlled wells with an action: a control, or a library siRNA mapped to a gene
    q = a[a.qc & (a.control.isin(["negative control", "positive control"])
                  | ((a.control == "") & a["Gene Symbol"].notna()))]
    for p in order:
        w = q[q.Plate == p]
        ctrl = w[w.control.isin(["negative control", "positive control"])]
        phen = w[(w.control == "") & (w.classes.str.len() > 0)]
        rest = w[(w.control == "") & (w.classes.str.len() == 0)]
        k = max(0, PER_PLATE - len(ctrl) - len(phen))
        rest = rest.iloc[rng.permutation(len(rest))[:k]]
        rows.append(pd.concat([ctrl, phen, rest]))
    st = pd.concat(rows)
    st = st[["Plate", "Well", "Plate_Well", "Gene Symbol", "siRNA Identifier", "control",
             "classes", "series"]].copy()
    st["plate_split"] = st.Plate.map(psplit)
    st["gene_split"] = st["Gene Symbol"].map(split).fillna("control")
    st.loc[st.control != "", "gene_split"] = "control"
    st["uuid"] = st.Plate.map(lambda p: s3[p]["uuid"])

    OUT.mkdir(parents=True, exist_ok=True)
    gene_split.to_csv(OUT / "gene_split.tsv", sep="\t", index=False)
    pdf.to_csv(OUT / "plate_split.tsv", sep="\t", index=False)
    st.to_csv(OUT / "stage1_wells.tsv", sep="\t", index=False)
    cls_counts = {}
    for c in sorted(set(CLASS_OF.values())):
        cls_counts[c] = int(st.classes.apply(lambda x, c=c: c in x).sum())
    summary = {
        "seed": SEED, "annotation_sha256": sha(ANN / "idr0013-screenA-annotation.csv.gz"),
        "genes": {k: int(v) for k, v in gene_split.split.value_counts().items()},
        "plates": {k: int(v) for k, v in pdf.split.value_counts().items()},
        "plates_on_s3": len(avail), "stage1_plates": len(order),
        "stage1_wells": len(st),
        "stage1_by_plate_split": {k: int(v) for k, v in st.plate_split.value_counts().items()},
        "stage1_by_gene_split": {k: int(v) for k, v in st.gene_split.value_counts().items()},
        "stage1_controls": {k: int(v) for k, v in st.control.value_counts().items() if k},
        "stage1_phenotype_class_wells": cls_counts,
        "stage1_genes": int(st["Gene Symbol"].nunique()),
    }
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
