"""Per-well MitoCheck annotation used across GeneWorld (screenA and screenB).

`classes` are the phenotype classes of a well's reproducible-phenotype terms (annotation level
"multiple replicates of reagent"). The three primary classes use the term sets frozen in
amendment 01 / 02 (`scripts/12_validated_hits.py`, `manifests/amend02_freeze.json`); the other
classes follow `scripts/11_splits_and_stage1.py`.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from geneworld.data import REPO

LEVEL = "multiple replicates of reagent"
CLASS_OF = {
    "mitosis delayed phenotype": "mitotic_delay", "mitosis arrested": "mitotic_delay",
    "metaphase delayed phenotype": "mitotic_delay", "metaphase arrested phenotype":
        "mitotic_delay",
    "prometaphase delayed phenotype": "mitotic_delay",
    "prometaphase arrested phenotype": "mitotic_delay",
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
    "mitotic metaphase plate congression phenotype": "congression",
}
PRIMARY = ["mitotic_delay", "polylobed", "binuclear"]


@lru_cache(maxsize=2)
def well_annotations(screen: str = "A") -> pd.DataFrame:
    """Index (Plate, Well): Gene Symbol, siRNA Identifier, control, qc, terms, classes."""
    f = (REPO / "data" / "annotations" /
         ("idr0013-screenA-annotation.csv.gz" if screen == "A"
          else "idr0013-screenB-annotation.csv"))
    a = pd.read_csv(f, low_memory=False)
    tcols = [c for c in a.columns if c.startswith("Phenotype ") and "Term Name" in c]
    at = (a["Phenotype Annotation Level"] == LEVEL).to_numpy()
    a["terms"] = [sorted({v for v in r if isinstance(v, str)}) if ok else []
                  for r, ok in zip(a[tcols].itertuples(index=False), at, strict=True)]
    a["classes"] = [sorted({CLASS_OF[t] for t in ts if t in CLASS_OF}) for ts in a.terms]
    a["control"] = a["Control Type"].fillna("")
    a["qc"] = a["Quality Control"].astype(str).eq("True")
    return a.set_index(["Plate", "Well"])[["Gene Symbol", "siRNA Identifier", "control", "qc",
                                           "terms", "classes"]]
