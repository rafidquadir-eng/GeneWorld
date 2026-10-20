"""Plate-identity probe at 36 h under both preprocessings (amendment 02 section 1 check).

On validation plates: the observed clip-12 (36 h) tile-mean latent of every well; a logistic
regression (C = 0.1) decodes the plate, 5-fold stratified cross-validation, plates with >= 5
wells. The same probe as stage 1 (`scripts/41_eval_h1.py:plate_probe`). Reported for the
per-plate control-normalised latents (gates) and the global standardisation (sensitivity),
plus a diagnostic (centring on the plate's control mean only, no SD division);
`--split train` repeats it on a seeded sample of 50 training plates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import cache  # noqa: E402
from geneworld.data import BACKBONE_KEY  # noqa: E402
from geneworld.eval.provenance import verified_backbone_record  # noqa: E402


def probe(x, plates, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    ok = np.isin(plates, [p for p in np.unique(plates) if (plates == p).sum() >= 5])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = cross_val_score(LogisticRegression(C=0.1, max_iter=3000), x[ok], plates[ok], cv=cv)
    return {"accuracy": float(acc.mean()), "fold_accuracy": [float(a) for a in acc],
            "n_wells": int(ok.sum()), "n_plates": int(len(np.unique(plates[ok]))),
            "chance": float(1 / len(np.unique(plates[ok])))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--out", default="results/a2/plate_probe.json")
    args = ap.parse_args()
    res = {"design_commit": "dd57079",
           "backbone": verified_backbone_record(BACKBONE_KEY[args.backbone]),
           "clip": 12, "split": args.split}
    for norm in ("plate", "global"):
        c = cache.load(args.backbone, norm, suffix=args.suffix)
        sel = (c.meta.plate_split == args.split).to_numpy()
        if args.split == "train":
            ps = sorted(c.meta.plate[sel].unique())
            pick = set(np.random.default_rng(0).choice(ps, min(50, len(ps)), replace=False))
            sel &= c.meta.plate.isin(pick).to_numpy()
        x = c.z[sel, 12].astype(np.float32).mean(1)
        res[norm] = probe(x, c.meta.plate.to_numpy()[sel])
        res[norm]["cache_stats"] = c.stats.get("plate_norm") if norm == "plate" else None
        print(norm, {k: v for k, v in res[norm].items() if k != "cache_stats"})
        if norm == "global":
            # diagnostic only (not a pipeline option): centre each well on its plate's
            # negative-control mean without dividing by the plate SD
            m = c.meta
            neg = (m.control == "negative control").to_numpy()
            pl = m.plate.to_numpy()
            xs = c.z[:, 12].astype(np.float32).mean(1)
            cen = xs.copy()
            for p in np.unique(pl[sel]):
                w = pl == p
                cen[w] -= xs[w & neg].mean(0)
            res["diagnostic_centre_on_plate_controls_only"] = probe(cen[sel], pl[sel])
            print("diagnostic centre-only", res["diagnostic_centre_on_plate_controls_only"]
                  ["accuracy"])
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
