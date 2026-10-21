"""Plate-identity probes at 36 h (amendment 02 section 1 check, with the author's diagnostics).

Three versions of the clip-12 (36 h) tile-mean latent, all reported side by side:
  global        global standardisation (stage-1 pipeline; sensitivity analysis)
  plate         per-plate negative-control normalisation (section 1; used by the gates)
  centre_only   global latents centred on the plate's negative-control mean (no SD division;
                diagnostic only, not a pipeline option)
Probes (logistic regression, C = 0.1):
  main               validation plates, all wells, decode the plate (5-fold stratified CV,
                     plates with >= 5 wells). Validation plates are one per library series, so
                     this probe mixes batch effect with gene content.
  within_series      (a) training series with >= 3 training plates; well positions present on
                     every one of those plates; decode which replicate plate a well came from;
                     cross-validation grouped by well position (a position's wells are never in
                     both train and test), so gene content cannot help. Accuracy per series vs
                     chance 1 / n_plates; mean over series.
  negative_controls  (b) negative-control wells of validation plates only; decode the plate;
                     for the plate and centre-only versions each scored control well is
                     normalised with its plate's statistics computed without it (leave one out;
                     the per-dimension sigma floor is the cached one).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import cache  # noqa: E402
from geneworld.data import BACKBONE_KEY  # noqa: E402
from geneworld.eval.provenance import verified_backbone_record  # noqa: E402

CLIP = 12
VERSIONS = ("global", "plate", "centre_only")


def clf():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.1, max_iter=3000)


def stratified(x, y, seed=0):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    ok = np.isin(y, [p for p in np.unique(y) if (y == p).sum() >= 5])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = cross_val_score(clf(), x[ok], y[ok], cv=cv)
    return {"accuracy": float(acc.mean()), "n_wells": int(ok.sum()),
            "n_plates": int(len(np.unique(y[ok]))), "chance": float(1 / len(np.unique(y[ok])))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default="results/a2/plate_probe.json")
    ap.add_argument("--split", default="val", choices=["val", "train"],
                    help="kept for the running orchestrator; 'train' is superseded by the "
                         "within-series replicate probe and only writes a note")
    args = ap.parse_args()
    if args.split == "train":
        (REPO / args.out).write_text(json.dumps({"superseded_by": "within_series_replicates "
                                                 "in results/a2/plate_probe.json"}, indent=1))
        return 0
    G = cache.load(args.backbone, "global", suffix=args.suffix)
    N = cache.load(args.backbone, "plate", suffix=args.suffix)
    m = G.meta
    assert (m.plate.values == N.meta.plate.values).all() and (m.well.values == N.meta.well.values).all()
    plates = m.plate.to_numpy()
    neg = (m.control == "negative control").to_numpy()
    X = {"global": G.z[:, CLIP].astype(np.float32).mean(1),
         "plate": N.z[:, CLIP].astype(np.float32).mean(1)}
    cen = X["global"].copy()
    for p in np.unique(plates):
        w = plates == p
        cen[w] -= X["global"][w & neg].mean(0)
    X["centre_only"] = cen
    res = {"design_commit": "dd57079", "clip": CLIP,
           "backbone": verified_backbone_record(BACKBONE_KEY[args.backbone]),
           "cache_plate_norm": N.stats.get("plate_norm")}

    # ---- main probe: validation plates, all wells
    val = (m.plate_split == "val").to_numpy()
    res["main_validation_plates"] = {v: stratified(X[v][val], plates[val]) for v in VERSIONS}
    res["main_validation_plates"]["note"] = ("validation plates are one per library series: "
                                             "batch effect and gene content are confounded")

    # ---- (a) within-series replicate probe on training series
    from sklearn.model_selection import GroupKFold, cross_val_score
    tr = (m.plate_split == "train").to_numpy()
    series = m.series.to_numpy()
    per = []
    for sr in sorted(set(series[tr])):
        ps = sorted(set(plates[tr & (series == sr)]))
        if len(ps) < 3:
            continue
        sub = m[tr & (series == sr)]
        common = set.intersection(*[set(sub.well[sub.plate == p]) for p in ps])
        if len(common) < 10:
            continue
        w = np.flatnonzero(tr & (series == sr) & m.well.isin(common).to_numpy())
        y, grp = plates[w], m.well.to_numpy()[w]
        cv = GroupKFold(n_splits=min(5, len(common)))
        rec = {"series": sr, "n_plates": len(ps), "n_positions": len(common),
               "chance": 1 / len(ps)}
        for v in VERSIONS:
            rec[v] = float(cross_val_score(clf(), X[v][w], y, groups=grp, cv=cv).mean())
        per.append(rec)
    res["within_series_replicates"] = {
        "n_series": len(per),
        "mean_chance": float(np.mean([r["chance"] for r in per])),
        **{f"mean_accuracy_{v}": float(np.mean([r[v] for r in per])) for v in VERSIONS},
        "per_series": per}

    # ---- (b) negative-control-only probe, leave-one-out normalisation statistics
    raw_all = np.load(G.path / "raw.npy", mmap_mode="r")
    keep = pd.read_csv(G.path / "meta.tsv", sep="\t").plate_norm_ok.to_numpy()
    rows = np.flatnonzero(keep)
    floor = np.load(N.path / "plate_stats.npz")["floor"]
    gst = np.load(G.path / "global_stats.npz")
    nv = np.flatnonzero(neg & val)
    xb = {v: [] for v in VERSIONS}
    for i in nv:
        others = np.flatnonzero(neg & (plates == plates[i]))
        others = others[others != i]
        r_i = raw_all[rows[i], CLIP].astype(np.float32).mean(0)
        r_o = np.stack([raw_all[rows[j], CLIP].astype(np.float32).mean(0) for j in others])
        xb["global"].append(X["global"][i])
        xb["plate"].append((r_i - r_o.mean(0)) / np.maximum(r_o.std(0, ddof=1), floor))
        g_o = (r_o - gst["mean"]) / gst["std"]
        xb["centre_only"].append(((r_i - gst["mean"]) / gst["std"]) - g_o.mean(0))
    res["negative_controls_loo"] = {v: stratified(np.stack(xb[v]), plates[nv]) for v in VERSIONS}

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    for k in ("main_validation_plates", "negative_controls_loo"):
        print(k, {v: round(res[k][v]["accuracy"], 4) for v in VERSIONS},
              "chance", round(res[k]["global"]["chance"], 4))
    ws = res["within_series_replicates"]
    print("within_series", ws["n_series"], "series; chance", round(ws["mean_chance"], 4),
          {v: round(ws[f"mean_accuracy_{v}"], 4) for v in VERSIONS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
