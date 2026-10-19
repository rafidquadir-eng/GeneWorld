"""Freeze the per-phenotype validated hit lists used as H3 phenotype-planning ground truth.

Design amendment 01, section 1. Source: MitoCheck validation screen (screenB; IDR idr0013
`idr0013-screenB-annotation.csv`, Neumann et al. 2010). A gene is a validated hit for phenotype
P when (a) MitoCheck marks it "Validated Mitotic Hit At Gene Level = yes" and (b) at least one
screenB siRNA (an independent reagent), annotated at the level "multiple replicates of reagent",
carries a phenotype term of class P. `n_sirnas` records how many independent screenB siRNAs
carry P (a stricter list, n_sirnas >= 2, is reported as a sensitivity analysis).

Classes (screenB term names):
  mitotic_delay  mitosis delayed / mitosis arrested / metaphase delayed / metaphase arrested /
                 prometaphase delayed / prometaphase arrested
  polylobed      polylobed nuclear phenotype
  binuclear      binuclear cell phenotype
  cell_death     cell death phenotype (kept only if >= 20 genes, per the amendment)

Writes `data/annotations/validated_hits_<P>.tsv` and `manifests/validated_hits.json`
(gene counts and SHA-256 of each list and of the source file).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ANN = REPO / "data" / "annotations"
SRC = ANN / "idr0013-screenB-annotation.csv"
TERMS = {
    "mitotic_delay": {"mitosis delayed phenotype", "mitosis arrested",
                      "metaphase delayed phenotype", "metaphase arrested phenotype",
                      "prometaphase delayed phenotype", "prometaphase arrested phenotype"},
    "polylobed": {"polylobed nuclear phenotype"},
    "binuclear": {"binuclear cell phenotype"},
    "cell_death": {"cell death phenotype"},
}
MIN_GENES = {"cell_death": 20}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    b = pd.read_csv(SRC, low_memory=False)
    b = b[(b["Phenotype Annotation Level"] == "multiple replicates of reagent")
          & (b["Validated Mitotic Hit At Gene Level"] == "yes") & b["Gene Symbol"].notna()]
    cols = [c for c in b.columns if "Term Name" in c]
    long = b.melt(id_vars=["Gene Symbol", "siRNA Identifier"], value_vars=cols).dropna(
        subset=["value"])
    out = {"source": SRC.name, "source_sha256": sha(SRC), "lists": {}}
    for p, terms in TERMS.items():
        hit = long[long.value.isin(terms)]
        g = hit.groupby("Gene Symbol")["siRNA Identifier"].nunique().rename("n_sirnas")
        rec = {"genes": int(len(g)), "genes_n_sirnas_ge2": int((g >= 2).sum())}
        if len(g) < MIN_GENES.get(p, 1):
            rec["used"] = False
            rec["reason"] = f"fewer than {MIN_GENES.get(p, 1)} validated genes"
        else:
            path = ANN / f"validated_hits_{p}.tsv"
            g.sort_index().to_csv(path, sep="\t")
            rec.update(used=True, file=path.name, sha256=sha(path))
        out["lists"][p] = rec
    (REPO / "manifests").mkdir(exist_ok=True)
    (REPO / "manifests" / "validated_hits.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
