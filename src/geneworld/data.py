"""Cached latents and well metadata for GeneWorld.

A well's state at clip index k (k = 0..12; clip k covers frames 6k..6k+15, i.e. it ends
(6k + 15) x 30 min after imaging starts) is the set of its four 256 x 256 tile tokens, each the
mean of the backbone's 4 x 4 pooled patch tokens. Clips are 3 h apart, so a rollout of h steps
from clip 0 forecasts 3h hours ahead. Clip index 13 (frames 77–92) is the endpoint clip and is
not part of the dynamics.

Latents are standardised per dimension with the mean and SD of training wells (all clips).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
N_STEPS = 13            # dynamic clips 0..12 (3 h apart)
ENDPOINT = 13
HOURS_PER_STEP = 3.0
BACKBONE_KEY = {"vjepa2": "vjepa2_vitl", "dinov2": "dinov2_vits14", "tijepa1": "tijepa1"}


@dataclass
class Latents:
    z: np.ndarray            # (N, 14, 4, D) float32, standardised
    meta: pd.DataFrame       # one row per well, aligned with z
    mean: np.ndarray
    std: np.ndarray
    backbone: str
    fingerprint: np.ndarray  # (N, F)


def load(backbone: str = "vjepa2", root: Path | None = None,
         wells_tsv: Path | None = None, gene_split: str = "gene") -> Latents:
    """`gene_split`: "gene" (default) or "complex" (whole Complex Portal components held out)."""
    root = root or REPO / "data" / "latents"
    wells = pd.read_csv(wells_tsv or REPO / "splits" / "stage1_wells.tsv", sep="\t")
    wells = wells.set_index(["Plate", "Well"])
    zs, rows, fps = [], [], []
    for f in sorted((root / backbone).glob("*.npz")):
        d = np.load(f)
        lat = d["latents"].astype(np.float32).mean(3)          # (n, 14, 4, D)
        for i, w in enumerate(d["wells"]):
            key = (f.stem, str(w))
            if key not in wells.index:
                continue
            r = wells.loc[key]
            rows.append({"plate": f.stem, "well": str(w), **r.to_dict()})
            zs.append(lat[i])
            fps.append(d["fingerprint"][i])
    meta = pd.DataFrame(rows)
    z = np.stack(zs)
    meta["control"] = meta["control"].fillna("")
    meta["action"] = np.where(meta.control == "negative control", "NEG",
                              meta["Gene Symbol"].fillna("NEG"))
    meta["sirna"] = meta["siRNA Identifier"].fillna("")
    if gene_split == "complex":
        cs = pd.read_csv(REPO / "splits" / "gene_split_complex.tsv", sep="\t")
        cmap = dict(zip(cs.gene, cs.split, strict=False))
        lib = meta.control == ""
        meta.loc[lib, "gene_split"] = meta.loc[lib, "Gene Symbol"].map(cmap).fillna("train")
    meta["train"] = (meta.plate_split == "train") & meta.gene_split.isin(["train", "control"])
    tr = z[meta.train.to_numpy()]
    mean = tr.mean((0, 1, 2))
    std = tr.std((0, 1, 2)) + 1e-6
    z = (z - mean) / std
    return Latents(z.astype(np.float32), meta, mean, std, backbone, np.stack(fps))
