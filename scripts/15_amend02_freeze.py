"""Amendment 02 sections 2 and 6: freeze primary wells, primary genes and planning truth
before any amended model is trained.

Reproducible-phenotype terms: the MitoCheck screenA terms recorded at the annotation level
"multiple replicates of reagent" (21 terms; `Phenotype N Term Name` columns).

* `splits/amend02_h1_primary_wells.tsv`: quality-controlled library wells on held-out
  (validation or test) plates, over the whole screen, whose siRNA carries a reproducible term
  **on training plates** (decided from training-plate annotation rows only).
* `splits/amend02_h2_primary_genes.tsv`: genes whose screenA annotation carries any
  reproducible term, with their gene-split and complex-split parts.
* `data/annotations/amend02_truth_<P>.tsv` (P = mitotic_delay, polylobed, binuclear): genes
  whose screenA annotation carries P's term (term sets of amendment 01 /
  `scripts/12_validated_hits.py`), with the number of siRNAs carrying it.
* `manifests/amend02_freeze.json`: SHA-256 of every list and of the inputs, counts per phenotype
  among unseen validation and test genes (gene split and complex split), primary-well counts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ANN = REPO / "data" / "annotations"
LEVEL = "multiple replicates of reagent"
PRIMARY = ["mitotic_delay", "polylobed", "binuclear"]

spec = importlib.util.spec_from_file_location("vh", REPO / "scripts" / "12_validated_hits.py")
vh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vh)
TERMS = {p: vh.TERMS[p] for p in PRIMARY}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    src = ANN / "idr0013-screenA-annotation.csv.gz"
    a = pd.read_csv(src, low_memory=False)
    a["qc"] = a["Quality Control"].astype(str).eq("True")
    lib = a["Control Type"].isna() & a["Gene Symbol"].notna()
    tcols = [c for c in a.columns if c.startswith("Phenotype ") and "Term Name" in c]
    at_level = (a["Phenotype Annotation Level"] == LEVEL).to_numpy()
    a["terms"] = [sorted({v for v in r if isinstance(v, str)}) if ok else []
                  for r, ok in zip(a[tcols].itertuples(index=False), at_level, strict=True)]
    plates = pd.read_csv(REPO / "splits" / "plate_split.tsv", sep="\t")
    psplit = dict(zip(plates.plate, plates.split, strict=True))
    a["plate_split"] = a.Plate.map(psplit)

    # H1 primary wells: siRNA terms from training-plate rows only
    tr = a[lib & (a.plate_split == "train")]
    sirna_terms = tr.groupby("siRNA Identifier").terms.apply(
        lambda s: sorted({t for x in s for t in x}))
    ho = a[lib & a.qc & a.plate_split.isin(["val", "test"])].copy()
    ho["terms_on_training_plates"] = ho["siRNA Identifier"].map(sirna_terms)
    ho["has_training_replicate"] = ho["terms_on_training_plates"].notna()
    prim = ho[ho.terms_on_training_plates.apply(lambda x: isinstance(x, list) and len(x) > 0)]
    prim = prim[["Plate", "Well", "Gene Symbol", "siRNA Identifier", "plate_split",
                 "terms_on_training_plates"]].copy()
    prim["terms_on_training_plates"] = prim.terms_on_training_plates.map("|".join)
    f_wells = REPO / "splits" / "amend02_h1_primary_wells.tsv"
    prim.sort_values(["Plate", "Well"]).to_csv(f_wells, sep="\t", index=False)

    # H2/H3 primary genes and truth lists (gene-level, external annotation)
    gs = pd.read_csv(REPO / "splits" / "gene_split.tsv", sep="\t")
    cs = pd.read_csv(REPO / "splits" / "gene_split_complex.tsv", sep="\t")
    gmap = dict(zip(gs.gene, gs.split, strict=True))
    cmap = dict(zip(cs.gene, cs.split, strict=True))
    lt = a[lib].explode("terms").dropna(subset=["terms"])
    gene_terms = lt.groupby("Gene Symbol").terms.apply(lambda s: sorted(set(s)))
    pg = pd.DataFrame({"gene": gene_terms.index, "terms": gene_terms.map("|".join).values})
    pg["gene_split"] = pg.gene.map(gmap)
    pg["complex_split"] = pg.gene.map(cmap)
    f_genes = REPO / "splits" / "amend02_h2_primary_genes.tsv"
    pg.sort_values("gene").to_csv(f_genes, sep="\t", index=False)

    out = {"design_commit": "dd57079", "annotation": src.name, "annotation_sha256": sha(src),
           "plate_split_sha256": sha(REPO / "splits" / "plate_split.tsv"),
           "gene_split_sha256": sha(REPO / "splits" / "gene_split.tsv"),
           "complex_split_sha256": sha(REPO / "splits" / "gene_split_complex.tsv"),
           "reproducible_terms": sorted({t for x in a.terms for t in x}),
           "h1_primary_wells": {"file": f_wells.name, "sha256": sha(f_wells),
                                "n": int(len(prim)),
                                "by_plate_split": {k: int(v) for k, v in
                                                   prim.plate_split.value_counts().items()},
                                "held_out_qc_library_wells": int(len(ho)),
                                "held_out_wells_with_training_replicate":
                                    int(ho.has_training_replicate.sum())},
           "h2_primary_genes": {"file": f_genes.name, "sha256": sha(f_genes), "n": int(len(pg)),
                                "gene_split": {k: int(v) for k, v in
                                               pg.gene_split.value_counts().items()},
                                "complex_split": {k: int(v) for k, v in
                                                  pg.complex_split.value_counts().items()}},
           "truth": {}}
    (REPO / "manifests" / "amend02_truth").mkdir(parents=True, exist_ok=True)
    for p, terms in TERMS.items():
        hit = lt[lt.terms.isin(terms)]
        g = hit.groupby("Gene Symbol")["siRNA Identifier"].nunique().rename("n_sirnas")
        f = ANN / f"amend02_truth_{p}.tsv"
        g.sort_index().to_csv(f, sep="\t")
        shutil.copy(f, REPO / "manifests" / "amend02_truth" / f.name)
        genes = set(g.index)
        out["truth"][p] = {
            "file": f.name, "sha256": sha(f), "terms": sorted(terms), "n_genes": len(genes),
            "unseen_counts": {
                f"{split}_{part}": int(sum(1 for x in genes if m.get(x) == part))
                for split, m in (("gene_split", gmap), ("complex_split", cmap))
                for part in ("val", "test")},
            "n_sirnas_ge2": int((g >= 2).sum())}
    (REPO / "manifests" / "amend02_freeze.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "reproducible_terms"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
