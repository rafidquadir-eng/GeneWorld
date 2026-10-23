"""Tile-mean latent cache and the two preprocessings of amendment 02 section 1.

`build` reads every per-plate shard under the given roots once and writes, under
`data/cache/<backbone><suffix>/`:
  meta.tsv        one row per well (plate, well, split columns, gene, siRNA, control)
  raw.npy         (N, 14, 4, D) float16 tile tokens (mean of the 4 x 4 pooled patch tokens)
  plate.npy       per-plate negative-control normalisation (section 1, used by the gates):
                  for plate p, clip t, dim d, mu and sigma over the plate's negative-control
                  wells of the tile-mean latent; z' = (z - mu) / max(sigma, floor_d), with
                  floor_d the 5th percentile of sigma over all plates and clips; plates with
                  < 4 usable negative-control wells are dropped (listed in stats.json)
  global.npy      global standardisation (stage-1 pipeline, the sensitivity analysis): mean
                  and SD per dim over training wells (training plates x training genes or
                  controls), all clips and tiles
  plate_stats.npz mu, sigma (after flooring) per plate, for normalising other latents
                  (static-clip control, later test plates) with the same statistics
  fingerprint.npy (N, 3000) float16 leakage fingerprints
  stats.json      counts, dropped plates, floors, input shard list hash
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geneworld.annotations import well_annotations
from geneworld.data import REPO

CACHE = REPO / "data" / "cache"
MIN_NEG = 4
WELL_TSVS = ["splits/stage1_wells.tsv", "splits/stage2_wells.tsv"]
ROOTS = ["data/latents", "data/latents_s2"]
#: every latent shard under data/latents*/ (backbone, path, SHA-256), author instruction
#: 2026-10-23; data/latents_unused_1019/ (early 10-19 shards, not used) is excluded by name
SHARD_MANIFEST = REPO / "manifests" / "cache_shards.tsv"
DATA = REPO / "data"
UNUSED_DIRS = {"latents_unused_1019"}
#: rehearsal of the test path (completion plan v4 S17/S18): its shards are listed in their own
#: manifest, read only when GENEWORLD_REHEARSAL=1; otherwise data/latents_rehearsal/ is ignored
REHEARSAL_DIR = "latents_rehearsal"
REHEARSAL_MANIFEST = REPO / "manifests" / "cache_shards_rehearsal.tsv"


def rehearsal_mode() -> bool:
    return os.environ.get("GENEWORLD_REHEARSAL") == "1"


class ShardManifestError(RuntimeError):
    pass


def check_shard_manifest() -> None:
    """Refuse (listing the paths) if a shard under data/latents*/ is not in the manifest, or a
    manifest shard is missing from disk. Paths only; backup and restore check the hashes."""
    want = set(pd.read_csv(SHARD_MANIFEST, sep="\t").path)
    skip = set(UNUSED_DIRS)
    if rehearsal_mode():
        if REHEARSAL_MANIFEST.exists():
            want |= set(pd.read_csv(REHEARSAL_MANIFEST, sep="\t").path)
    else:
        skip.add(REHEARSAL_DIR)
    have = {str(p.relative_to(DATA.parent)) for d in DATA.glob("latents*")
            if d.is_dir() and d.name not in skip for p in d.rglob("*.npz")}
    extra, missing = sorted(have - want), sorted(want - have)
    if extra or missing:
        raise ShardManifestError(
            f"latent shards do not match {SHARD_MANIFEST.name}: "
            f"{len(extra)} not in the manifest {extra}; {len(missing)} missing {missing}")


def well_table(tsvs) -> pd.DataFrame:
    cols = ["Plate", "Well", "Gene Symbol", "siRNA Identifier", "control", "series",
            "plate_split", "gene_split"]
    t = pd.concat([pd.read_csv(REPO / f, sep="\t")[cols] for f in tsvs])
    return t.drop_duplicates(["Plate", "Well"]).set_index(["Plate", "Well"])


def build(backbone: str, roots=ROOTS, tsvs=WELL_TSVS, include_test: bool = False,
          suffix: str = "", exclude: set[str] | None = None) -> Path:
    """`exclude`: shard paths (relative to the repository) to skip; used only by the
    rehearsal, so that re-embedded shards replace the originals of the same plates."""
    check_shard_manifest()
    out = CACHE / f"{backbone}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    wt = well_table(tsvs)
    shards = sorted(f for r in roots for f in (REPO / r / backbone).glob("*.npz")
                    if not exclude or str(f.relative_to(REPO)) not in exclude)
    h = hashlib.sha256()
    rows, toks, fps = [], [], []
    seen = set()
    for f in shards:
        h.update(f"{f.relative_to(REPO)}:{f.stat().st_size}".encode())
        d = np.load(f)
        lat = None
        for i, w in enumerate(d["wells"]):
            key = (f.stem, str(w))
            if key not in wt.index or key in seen:
                continue
            r = wt.loc[key]
            if r.plate_split == "test" and not include_test:
                continue
            if lat is None:
                lat = d["latents"].astype(np.float32).mean(3)          # (n, 14, 4, D)
            seen.add(key)
            rows.append({"plate": key[0], "well": key[1], **r.to_dict()})
            toks.append(lat[i].astype(np.float16))
            fps.append(d["fingerprint"][i].astype(np.float16))
    meta = pd.DataFrame(rows)
    meta["control"] = meta["control"].fillna("")
    raw = np.stack(toks)                                               # (N, 14, 4, D) f16
    np.save(out / "raw.npy", raw)
    np.save(out / "fingerprint.npy", np.stack(fps))
    meta.to_csv(out / "meta.tsv", sep="\t", index=False)
    stats = {"backbone": backbone, "n_wells": int(len(meta)), "n_plates": int(
        meta.plate.nunique()), "include_test": include_test, "shards_sha256": h.hexdigest(),
        "n_shards": len(shards)}

    # ---- global standardisation (training wells)
    train = ((meta.plate_split == "train") & meta.gene_split.isin(["train", "control"])).to_numpy()
    acc = np.zeros(raw.shape[-1]), np.zeros(raw.shape[-1]), 0
    s1, s2, n = acc
    for i in np.flatnonzero(train):
        x = raw[i].astype(np.float64).reshape(-1, raw.shape[-1])
        s1 += x.sum(0)
        s2 += (x ** 2).sum(0)
        n += len(x)
    gm = s1 / n
    gs = np.sqrt(np.maximum(s2 / n - gm ** 2, 0)) + 1e-6
    glob_ = np.empty_like(raw)
    for i in range(0, len(raw), 2048):
        glob_[i:i + 2048] = ((raw[i:i + 2048].astype(np.float32) - gm) / gs).astype(np.float16)
    np.save(out / "global.npy", glob_)
    del glob_
    np.savez(out / "global_stats.npz", mean=gm.astype(np.float32), std=gs.astype(np.float32))

    # ---- per-plate negative-control normalisation
    neg = (meta.control == "negative control").to_numpy()
    plates = sorted(meta.plate.unique())
    mu, sd, nneg, dropped = {}, {}, {}, []
    for p in plates:
        w = np.flatnonzero(neg & (meta.plate == p).to_numpy())
        nneg[p] = int(len(w))
        if len(w) < MIN_NEG:
            dropped.append(p)
            continue
        tm = raw[w].astype(np.float32).mean(2)                         # (n_neg, 14, D)
        mu[p] = tm.mean(0)
        sd[p] = tm.std(0, ddof=1)
    allsd = np.stack(list(sd.values()))                                # (P, 14, D)
    floor = np.percentile(allsd.reshape(-1, allsd.shape[-1]), 5, axis=0)
    for p in sd:
        sd[p] = np.maximum(sd[p], floor)
    keep = ~meta.plate.isin(dropped).to_numpy()
    norm = np.empty_like(raw)
    for p in sd:
        w = np.flatnonzero((meta.plate == p).to_numpy())
        norm[w] = ((raw[w].astype(np.float32) - mu[p][None, :, None]) /
                   sd[p][None, :, None]).astype(np.float16)
    norm[~keep] = 0
    np.save(out / "plate.npy", norm)
    np.savez(out / "plate_stats.npz", plates=np.array(list(sd)),
             mu=np.stack([mu[p] for p in sd]), sigma=np.stack([sd[p] for p in sd]),
             floor=floor.astype(np.float32))
    meta["plate_norm_ok"] = keep
    meta.to_csv(out / "meta.tsv", sep="\t", index=False)
    stats.update({"plate_norm": {"min_neg_wells": MIN_NEG, "dropped_plates": dropped,
                                 "n_wells_dropped": int((~keep).sum()),
                                 "neg_wells_per_plate": {"min": int(min(nneg.values())),
                                                         "median": float(np.median(list(
                                                             nneg.values()))),
                                                         "max": int(max(nneg.values()))},
                                 "sigma_floor_median": float(np.median(floor)),
                                 "frac_sigma_floored": float(np.mean(allsd < floor))}})
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return out


@dataclass
class Cached:
    z: np.ndarray            # (N, 14, 4, D) float16, rows aligned with meta
    meta: pd.DataFrame
    backbone: str
    norm: str
    stats: dict
    path: Path

    @property
    def fingerprint(self) -> np.ndarray:
        keep = pd.read_csv(self.path / "meta.tsv", sep="\t").plate_norm_ok.to_numpy()
        return np.asarray(np.load(self.path / "fingerprint.npy", mmap_mode="r")[keep])


def load(backbone: str = "vjepa2", norm: str = "plate", gene_split: str = "gene",
         suffix: str = "") -> Cached:
    """`norm`: "plate" (gates) or "global" (sensitivity). Wells of plates dropped by the
    plate normalisation are removed under both, so the two analyses use the same wells."""
    path = CACHE / f"{backbone}{suffix}"
    meta = pd.read_csv(path / "meta.tsv", sep="\t")
    zm = np.load(path / f"{norm}.npy", mmap_mode="r")
    stats = json.loads((path / "stats.json").read_text())
    meta["control"] = meta["control"].fillna("")
    if gene_split == "complex":
        cs = pd.read_csv(REPO / "splits" / "gene_split_complex.tsv", sep="\t")
        cmap = dict(zip(cs.gene, cs.split, strict=False))
        lib = meta.control == ""
        meta.loc[lib, "gene_split"] = meta.loc[lib, "Gene Symbol"].map(cmap).fillna("train")
    meta["action"] = np.where(meta.control == "negative control", "NEG",
                              meta["Gene Symbol"].fillna("NEG"))
    meta["sirna"] = meta["siRNA Identifier"].fillna("")
    meta["train"] = (meta.plate_split == "train") & meta.gene_split.isin(["train", "control"])
    idx = pd.MultiIndex.from_arrays([meta.plate, meta.well])
    cl = well_annotations("A").classes.reindex(idx)
    isb = (meta.plate_split == "screenB").to_numpy()
    if isb.any():   # screenB wells take their classes from the screenB annotation
        clb = well_annotations("B").classes.reindex(idx)
        cl = pd.Series(np.where(isb, clb.values, cl.values))
    meta["classes"] = [c if isinstance(c, list) else [] for c in cl]
    keep = meta.plate_norm_ok.to_numpy()
    if norm == "global":
        # screenB has no negative controls (dropped by the plate normalisation); it is
        # evaluated under the global preprocessing only (author instruction 2026-10-21)
        keep = keep | isb
    meta = meta[keep].reset_index(drop=True)
    z = np.ascontiguousarray(zm[keep])       # in-memory copy aligned with meta
    return Cached(z, meta, backbone, norm, stats, path)
