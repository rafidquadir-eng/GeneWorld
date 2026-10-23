"""Amendment 02 training/evaluation helpers: protocol, configurations, primary sets, families.

Everything fixed here is read from `manifests/amend02_protocol.json` (committed before any
amended model was trained) and the frozen lists of amendment 02 sections 2 and 6.
"""
from __future__ import annotations

import json
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
import torch

from geneworld import sets
from geneworld.data import REPO

DESIGN_COMMIT = "dd57079"
PRIMARY_STEPS = [8, 12]           # 24 h and 36 h from clip 0
SECONDARY_STEPS = [1, 4]          # 3 h and 12 h
ALL_STEPS = list(range(1, 13))
# GW_A2_DEV=1 redirects smoke-test outputs away from the real run directories
_DEV = __import__("os").environ.get("GW_A2_DEV") == "1"
# GENEWORLD_REHEARSAL=1 (completion plan v4 S18, `run_stage1_test.sh --rehearsal`) writes every
# evaluation output to results/a2/rehearsal/; models are still read from checkpoints/a2/
_REHEARSAL = __import__("os").environ.get("GENEWORLD_REHEARSAL") == "1"
CK_DIR = REPO / "checkpoints" / ("a2_dev" if _DEV else "a2")
RES_DIR = REPO / "results" / ("a2_dev" if _DEV else "a2/rehearsal" if _REHEARSAL else "a2")


@lru_cache(maxsize=1)
def protocol() -> dict:
    return json.loads((REPO / "manifests" / "amend02_protocol.json").read_text())


def configs() -> list[dict]:
    g = protocol()["grid"]
    out = []
    for size, drop, wd in product(sorted(g["predictor"]), g["dropout"], g["weight_decay"]):
        p = g["predictor"][size]
        out.append({"size": size, "width": p["width"], "depth": p["depth"], "dropout": drop,
                    "weight_decay": wd, "id": f"{size}_d{drop}_w{wd}"})
    return out


def config_by_id(cid: str) -> dict:
    return next(c for c in configs() if c["id"] == cid)


def family_mode(family: str) -> tuple[str, str, bool]:
    """'h1_learned' | 'h1_none' | 'h2:<src>' | 'h2shuf:<src>' -> (mode, source, shuffled)."""
    if family == "h1_learned":
        return "learned", "", False
    if family == "h1_none":
        return "none", "", False
    kind, src = family.split(":")
    return "encoding", src, kind == "h2shuf"


def family_encoding(family: str, seed: int) -> dict[str, np.ndarray]:
    """Gene -> encoding; for a shuffled family a fixed random permutation of genes to
    encodings (permutation seed = model seed), so the model can memorise but not transfer."""
    mode, src, shuf = family_mode(family)
    if mode != "encoding":
        return {}
    enc = sets.encoding(src)
    if not shuf:
        return enc
    genes = sorted(enc)
    perm = np.random.default_rng(10_000 + seed).permutation(len(genes))
    return {g: enc[genes[j]] for g, j in zip(genes, perm, strict=True)}


@lru_cache(maxsize=1)
def h1_primary() -> set[tuple[str, str]]:
    """Frozen H1 primary wells (screenA held-out plates) and screenB primary wells."""
    out = set()
    for f in ("amend02_h1_primary_wells.tsv", "amend02_screenB_primary_wells.tsv"):
        t = pd.read_csv(REPO / "splits" / f, sep="\t")
        out |= set(zip(t.Plate, t.Well, strict=True))
    return out


@lru_cache(maxsize=1)
def h2_primary_genes() -> set[str]:
    return set(pd.read_csv(REPO / "splits" / "amend02_h2_primary_genes.tsv", sep="\t").gene)


def eval_mask(meta: pd.DataFrame, which: str, vocab: set[str], enc: dict | None = None,
              primary: bool = True) -> np.ndarray:
    """which: h1_val | h1_test | h1_screenB | h2_val | h2_test | h2_screenB."""
    lib = (meta.control == "").to_numpy()
    ps = meta.plate_split.to_numpy()
    if which.startswith("h1"):
        part = which.split("_")[1]
        m = (ps == part) & lib & meta.action.isin(vocab).to_numpy()
        if primary:
            prim = h1_primary()
            m &= np.array([(p, w) in prim for p, w in zip(meta.plate, meta.well, strict=True)])
        return m
    part = which.split("_")[1]
    if part == "screenB":       # unseen (test-split) genes, screenB's independent siRNAs
        m = (meta.gene_split.to_numpy() == "test") & lib & (ps == "screenB")
    else:
        m = (meta.gene_split.to_numpy() == part) & lib & (ps != "screenB")
    if part == "val":
        m &= ps != "test"
    if enc is not None:
        m &= meta.action.isin(set(enc)).to_numpy()
    if primary:
        m &= meta.action.isin(h2_primary_genes()).to_numpy()
    return m


def train_mask(meta: pd.DataFrame, family: str, seed: int) -> np.ndarray:
    m = meta.train.to_numpy().copy()
    if family_mode(family)[0] == "encoding":
        enc = family_encoding(family, seed)
        m &= meta.action.isin(set(enc) | {"NEG"}).to_numpy()
    return m


def train_wells_sha256(meta: pd.DataFrame, family: str, seed: int) -> str:
    import hashlib
    m = train_mask(meta, family, seed)
    keys = sorted(f"{p}:{w}" for p, w in zip(meta.plate[m], meta.well[m], strict=True))
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def run_name(norm: str, backbone: str, family: str, cid: str, seed: int,
             gene_split: str = "gene") -> str:
    fam = family.replace(":", "-")
    gs = "" if gene_split == "gene" else f"_{gene_split}"
    return f"a2_{norm}_{backbone}{gs}_{fam}_{cid}_s{seed}"


def load_model(path, cached):
    """(model, batcher, record, checkpoint dict) for an amended checkpoint applied to `cached`;
    refuses on any mismatch of backbone, preprocessing, gene split or cache identity."""
    from geneworld.data import BACKBONE_KEY
    from geneworld.eval.provenance import predictor_record, verified_backbone_record
    from geneworld.models.predictor import ActionPredictor
    from geneworld.train.batcher import DEV, Batcher
    d = torch.load(path, map_location="cpu", weights_only=False)
    a = d["args"]
    for k, v in (("backbone", cached.backbone), ("norm", cached.norm)):
        if a[k] != v:
            raise ValueError(f"{path}: trained with {k}={a[k]}, evaluated with {v}")
    if d["train_wells_sha256"] != train_wells_sha256(cached.meta, a["family"], a["seed"]):
        raise ValueError(f"{path}: training wells differ from the evaluation cache's")
    enc = family_encoding(a["family"], a["seed"])
    mode = family_mode(a["family"])[0]
    vocab = {g: i for i, g in enumerate(d["vocab"])}
    sirnas = {s: i + 1 for i, s in enumerate(d["sirnas"])}
    model = ActionPredictor(d["dim"], mode, n_actions=len(vocab), n_sirna=len(sirnas),
                            enc_dim=d["enc_dim"], width=a["width"], depth=a["depth"],
                            drop=a["dropout"]).to(DEV)
    model.load_state_dict(d["model"])
    model.eval()
    bat = Batcher(cached, mode, vocab, sirnas, enc)
    rec = predictor_record(path, d["step"], d["split_sha256"],
                           verified_backbone_record(BACKBONE_KEY[a["backbone"]]),
                           a["family"])
    rec.update({"norm": a["norm"], "config": a["config"], "seed": a["seed"],
                "gene_split": a["gene_split"], "design_commit": DESIGN_COMMIT})
    return model, bat, rec, d
