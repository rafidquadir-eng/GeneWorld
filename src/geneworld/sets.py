"""Training and evaluation well sets, action vocabularies and gene encodings.

  train    training plates x (training genes or controls)
  h1_val   validation plates x actions seen in training (selection for H1)
  h1_test  test plates x actions seen in training (H1, evaluated once after freezing)
  h2_val   validation genes, on training or validation plates (selection for H2/H3)
  h2_test  test genes, on any plate (H2/H3, evaluated once after freezing)
Library wells and negative controls are flagged so that H1 can be reported on library wells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from geneworld.data import REPO

ACTIONS = REPO / "data" / "actions"
SOURCES = ["perturbseq", "jump_crispr", "jump_orf", "string", "esm2"]


def masks(meta: pd.DataFrame, vocab: set[str]) -> dict[str, np.ndarray]:
    seen = meta.action.isin(vocab).to_numpy()
    ps, gs = meta.plate_split.to_numpy(), meta.gene_split.to_numpy()
    lib = (meta.control == "").to_numpy()
    neg = (meta.control == "negative control").to_numpy()
    return {
        "train": meta.train.to_numpy(),
        "h1_val": (ps == "val") & seen & (lib | neg),
        "h1_test": (ps == "test") & seen & (lib | neg),
        "h2_val": (gs == "val") & (ps != "test") & lib,
        "h2_test": (gs == "test") & lib,
        "library": lib, "neg": neg,
    }


def gene_to_ensg() -> dict[str, str]:
    t = pd.read_csv(ACTIONS / "mitocheck_gene_to_ensg.tsv", sep="\t")
    return dict(zip(t.gene, t.ensembl_gene_id, strict=False))


def encoding(source: str) -> dict[str, np.ndarray]:
    """MitoCheck gene symbol -> frozen encoding vector (standardised per dimension)."""
    d = np.load(ACTIONS / f"{source}.npz")
    x = d["X"].astype(np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    by_ensg = dict(zip(d["genes"], x, strict=False))
    return {g: by_ensg[e] for g, e in gene_to_ensg().items() if e in by_ensg}
