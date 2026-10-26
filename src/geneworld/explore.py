"""Exploratory analyses after the test (completion plan v2 S7, v3 S15): helpers for E1-E3.

Nothing here feeds a gate, a selection or a frozen list; every output carries LABEL.

* E1, spatial batch correction: per plate and clip, a row/column median polish of the library
  wells' tile-mean latents after section-1 normalisation (B-score style); the fitted overall,
  row and column effects are subtracted from every well of the plate (all four tiles).
* E2, batch carried by the forecast: the within-series replicate probe of
  `scripts/53_plate_probe.py` (a), applied to forecast and observed clip-12 latents.
* E3, wells excluded only for missing frames >= 88: clips 0-12 (frames 0-87) of a partial
  movie, intensity-normalised over its real frames; clip 13 is marked invalid (NaN). New wells
  are normalised with the stored per-plate statistics, so existing wells keep their values.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = "exploratory, after the test"
N_CLIPS = 14
CLIP_STARTS = list(range(0, 73, 6)) + [77]     # as scripts/20_stream_embed.py
NF, TILE = 16, 256
VALID_FRAMES_E3 = 88                           # frames 0-87: clips 0-12


# ---------------------------------------------------------------- E1: median polish
def well_rc(well: str) -> tuple[int, int]:
    """'A1' -> (0, 0); 'P24' -> (15, 23)."""
    return ord(well[0].upper()) - ord("A"), int(well[1:]) - 1


def median_polish(x: np.ndarray, n_iter: int = 10) -> tuple[np.ndarray, ...]:
    """Tukey median polish of x (R, C, ...) with NaN for empty cells.
    Returns (overall (...), row (R, ...), col (C, ...)); empty rows/columns get effect 0."""
    res = x.astype(np.float64).copy()
    row = np.zeros((x.shape[0],) + x.shape[2:])
    col = np.zeros((x.shape[1],) + x.shape[2:])
    overall = np.zeros(x.shape[2:])
    for _ in range(n_iter):
        with np.errstate(all="ignore"):
            r = np.nan_to_num(np.nanmedian(res, axis=1))
        res -= r[:, None]
        row += r
        with np.errstate(all="ignore"):
            m = np.nan_to_num(np.nanmedian(col, axis=0))
        col -= m
        overall += m
        with np.errstate(all="ignore"):
            c = np.nan_to_num(np.nanmedian(res, axis=0))
        res -= c[None]
        col += c
        with np.errstate(all="ignore"):
            m = np.nan_to_num(np.nanmedian(row, axis=0))
        row -= m
        overall += m
    return overall, row, col


def bscore_correct(z: np.ndarray, meta: pd.DataFrame, lib: np.ndarray,
                   n_iter: int = 10) -> np.ndarray:
    """E1: z (N, T, K, D) section-1-normalised latents -> spatially corrected latents.
    Effects are estimated per plate, clip and dimension from the library wells' tile-mean and
    subtracted from every well of the plate."""
    out = np.empty_like(z)
    plates = meta.plate.to_numpy()
    rc = np.array([well_rc(w) for w in meta.well])
    for p in np.unique(plates):
        w = np.flatnonzero(plates == p)
        tm = z[w].astype(np.float32).mean(2)                       # (n, T, D)
        R, C = rc[w, 0].max() + 1, rc[w, 1].max() + 1
        grid = np.full((R, C) + tm.shape[1:], np.nan, np.float32)
        lw = lib[w]
        grid[rc[w][lw, 0], rc[w][lw, 1]] = tm[lw]
        ov, row, col = median_polish(grid, n_iter)
        eff = ov[None] + row[rc[w, 0]] + col[rc[w, 1]]             # (n, T, D)
        out[w] = (z[w].astype(np.float32) - eff[:, :, None]).astype(z.dtype)
    return out


# ---------------------------------------------------------------- E2: replicate probe
def within_series_probe(x: np.ndarray, meta: pd.DataFrame, rows: np.ndarray | None = None,
                        min_plates: int = 3, min_positions: int = 10) -> dict:
    """Probe (a) of scripts/53_plate_probe.py for one feature matrix x (N, F): training series
    with >= 3 training plates, well positions present on every one of them, decode the
    replicate plate with cross-validation grouped by well position. `rows` restricts the wells
    (default: all training-plate wells)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    tr = (meta.plate_split == "train").to_numpy()
    if rows is not None:
        keep = np.zeros(len(meta), bool)
        keep[rows] = True
        tr = tr & keep
    series, plates = meta.series.to_numpy(), meta.plate.to_numpy()
    per = []
    for sr in sorted(set(series[tr])):
        ps = sorted(set(plates[tr & (series == sr)]))
        if len(ps) < min_plates:
            continue
        sub = meta[tr & (series == sr)]
        common = set.intersection(*[set(sub.well[sub.plate == p]) for p in ps])
        if len(common) < min_positions:
            continue
        w = np.flatnonzero(tr & (series == sr) & meta.well.isin(common).to_numpy())
        cv = GroupKFold(n_splits=min(5, len(common)))
        acc = cross_val_score(LogisticRegression(C=0.1, max_iter=3000), x[w], plates[w],
                              groups=meta.well.to_numpy()[w], cv=cv).mean()
        per.append({"series": sr, "n_plates": len(ps), "n_positions": len(common),
                    "chance": 1 / len(ps), "accuracy": float(acc)})
    return {"n_series": len(per),
            "mean_chance": float(np.mean([r["chance"] for r in per])) if per else None,
            "mean_accuracy": float(np.mean([r["accuracy"] for r in per])) if per else None,
            "per_series": per}


def probe_wells(meta: pd.DataFrame, min_plates: int = 3,
                min_positions: int = 10) -> np.ndarray:
    """Rows used by `within_series_probe` (to forecast only those wells in E2)."""
    tr = (meta.plate_split == "train").to_numpy()
    series, plates = meta.series.to_numpy(), meta.plate.to_numpy()
    rows = []
    for sr in sorted(set(series[tr])):
        ps = sorted(set(plates[tr & (series == sr)]))
        if len(ps) < min_plates:
            continue
        sub = meta[tr & (series == sr)]
        common = set.intersection(*[set(sub.well[sub.plate == p]) for p in ps])
        if len(common) >= min_positions:
            rows.append(np.flatnonzero(tr & (series == sr)
                                       & meta.well.isin(common).to_numpy()))
    return np.concatenate(rows) if rows else np.array([], int)


# ---------------------------------------------------------------- E3: partial movies
def partial_clips(frames: np.ndarray, n_valid: int = VALID_FRAMES_E3, device: str = "cuda"):
    """frames (>= n_valid, H, W) uint8 of which the first n_valid are real -> (clips
    (14*4, 16, 256, 256) float in [0, 1], valid_clips (14,) bool, frame means of the real
    frames). As `tiles_and_clips` of scripts/20_stream_embed.py, except that the intensity
    quantiles use the real frames only and frames past n_valid repeat the last real frame;
    clips that need them are marked invalid."""
    import torch
    real = frames[:n_valid]
    f = torch.from_numpy(real).to(device).float()
    sub = f[::4, ::4, ::4].flatten()
    lo, hi = torch.quantile(sub, torch.tensor([0.005, 0.998], device=device))
    f = ((f - lo) / torch.clamp(hi - lo, min=1.0)).clamp_(0, 1)
    need = max(CLIP_STARTS) + NF
    if len(f) < need:
        f = torch.cat([f, f[-1:].expand(need - len(f), -1, -1)])
    x0 = (f.shape[2] - 512) // 2
    f = f[:, :, x0:x0 + 512]
    tiles = [f[:, :TILE, :TILE], f[:, :TILE, TILE:], f[:, TILE:, :TILE], f[:, TILE:, TILE:]]
    clips = torch.stack([t[s:s + NF] for s in CLIP_STARTS for t in tiles])
    valid = np.array([s + NF <= n_valid for s in CLIP_STARTS])
    return clips, valid, real.reshape(len(real), -1).mean(1)


def plate_stats_for(raw: np.ndarray, plates: np.ndarray, neg: np.ndarray, stats_npz: Path,
                    min_neg: int = 4) -> tuple[dict, dict, list]:
    """Per-plate (mu, sigma) (14, D): the stored section-1 statistics where the plate has them;
    otherwise computed from the plate's own negative-control wells in `raw` (tile-mean, >=
    min_neg wells, sigma floored with the stored floor). Returns (mu, sigma, plates without
    usable statistics)."""
    st = np.load(stats_npz)
    mu = {p: m for p, m in zip(st["plates"], st["mu"], strict=True)}
    sd = {p: s for p, s in zip(st["plates"], st["sigma"], strict=True)}
    floor, missing = st["floor"], []
    for p in np.unique(plates):
        if p in mu:
            continue
        w = np.flatnonzero((plates == p) & neg)
        if len(w) < min_neg:
            missing.append(str(p))
            continue
        tm = raw[w].astype(np.float32).mean(2)
        mu[p] = np.nanmean(tm, 0)
        sd[p] = np.maximum(np.nanstd(tm, 0, ddof=1), floor)
    return mu, sd, missing


def normalise(raw: np.ndarray, plates: np.ndarray, mu: dict, sd: dict) -> np.ndarray:
    """Section-1 normalisation of raw (n, 14, K, D) with given per-plate statistics."""
    out = np.empty(raw.shape, np.float16)
    for p in np.unique(plates):
        w = np.flatnonzero(plates == p)
        out[w] = ((raw[w].astype(np.float32) - mu[p][None, :, None]) /
                  sd[p][None, :, None]).astype(np.float16)
    return out


def extend_cache(base: Path, out: Path, raw_new: np.ndarray, meta_new: pd.DataFrame,
                 note: dict) -> dict:
    """E3: copy cache `base` to `out` and append wells (plate-normalised; clip 13 may be NaN).
    Existing rows are copied unchanged. New training-plate rows get plate_split 'train_e3', so
    they enter no training set and no replicate baseline. Writes plate.npy, meta.tsv,
    fingerprint.npy (zeros for new rows), stats.json (with an 'e3' record) and plate_stats."""
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(base / "meta.tsv", sep="\t", low_memory=False)
    z = np.load(base / "plate.npy", mmap_mode="r")
    mn = meta_new.copy()
    neg = (mn.control.fillna("") == "negative control").to_numpy()
    mu, sd, missing = plate_stats_for(raw_new, mn.plate.to_numpy(), neg,
                                      base / "plate_stats.npz")
    ok = ~mn.plate.isin(missing).to_numpy()
    mn = mn[ok].reset_index(drop=True)
    zn = normalise(raw_new[ok], mn.plate.to_numpy(), mu, sd)
    mn.loc[mn.plate_split == "train", "plate_split"] = "train_e3"
    mn["plate_norm_ok"] = True
    mn["e3_partial"] = True
    meta["e3_partial"] = False
    allz = np.lib.format.open_memmap(out / "plate.npy", "w+", np.float16,
                                     (len(z) + len(zn),) + z.shape[1:])
    for i in range(0, len(z), 4096):
        j = min(i + 4096, len(z))
        allz[i:j] = z[i:j]
    allz[len(z):] = zn
    allz.flush()
    fp = np.load(base / "fingerprint.npy", mmap_mode="r")
    np.save(out / "fingerprint.npy", np.concatenate(
        [np.asarray(fp), np.zeros((len(zn),) + fp.shape[1:], fp.dtype)]))
    pd.concat([meta, mn[[c for c in meta.columns if c in mn.columns]]],
              ignore_index=True).to_csv(out / "meta.tsv", sep="\t", index=False)
    shutil.copy(base / "plate_stats.npz", out / "plate_stats.npz")
    stats = json.loads((base / "stats.json").read_text())
    stats["e3"] = {"label": LABEL, "base_cache": base.name, "n_wells_added": int(len(mn)),
                   "plates_without_statistics": missing, **note}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats["e3"]
