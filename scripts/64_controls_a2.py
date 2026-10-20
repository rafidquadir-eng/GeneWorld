"""Amendment 02 controls on validation (or, after the freeze, test) plates.

* Static clips (A.7(2)): the 300 static-control wells (each clip = its middle frame repeated)
  normalised with the same per-plate statistics (plate_stats.npz) or global statistics.
  Encoder motion sensitivity = real-vs-static distance / median distance between two wells of
  the same plate (same clip). Predictor: 24/36 h error from the static versus the real clip-0
  context, selected h1_learned models, errors averaged over seeds.
* Leakage (A.7(6)): fingerprint correlation of every held-out-plate well against every
  training-plate well (stage 1 + stage 2); match r >= 0.98, reported r >= 0.90.
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
from geneworld.eval import a2eval as E  # noqa: E402
from geneworld.eval.provenance import verified_backbone_record  # noqa: E402
from geneworld.preprocess.fingerprint import MATCH_R, REPORT_R  # noqa: E402
from geneworld.train import a2  # noqa: E402


def static_latents(c, backbone):
    keys = {(p, w): i for i, (p, w) in enumerate(zip(c.meta.plate, c.meta.well, strict=True))}
    if c.norm == "plate":
        st = np.load(c.path / "plate_stats.npz")
        stats = {p: (mu, sg) for p, mu, sg in zip(st["plates"], st["mu"], st["sigma"],
                                                   strict=True)}
    else:
        g = np.load(c.path / "global_stats.npz")
    zs, idx = [], []
    for f in sorted((REPO / "data" / "latents_static" / backbone).glob("*.npz")):
        d = np.load(f)
        lat = d["latents"].astype(np.float32).mean(3)               # (n, 14, 4, D)
        for i, w in enumerate(d["wells"]):
            k = (f.stem, str(w))
            if k not in keys:
                continue
            if c.norm == "plate":
                mu, sg = stats[f.stem]
                zs.append((lat[i] - mu[:, None]) / sg[:, None])
            else:
                zs.append((lat[i] - g["mean"]) / g["std"])
            idx.append(keys[k])
    return np.stack(zs).astype(np.float32), np.array(idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="plate", choices=["plate", "global"])
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--freeze-commit", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.freeze_commit:
        sys.exit("test plates are evaluated once, after freezing: pass --freeze-commit")
    c = cache.load(args.backbone, args.norm, suffix=args.suffix)
    m, z = c.meta, c.z
    out = {"norm": args.norm, "backbone": verified_backbone_record(BACKBONE_KEY[args.backbone]),
           "design_commit": a2.DESIGN_COMMIT, "split": args.split,
           "freeze_commit": args.freeze_commit or None}
    zs, idx = static_latents(c, args.backbone)
    real = z[idx].astype(np.float32)
    d_rs = ((real - zs) ** 2).mean((2, 3))
    pl = m.plate.to_numpy()
    between = []
    for p in np.unique(pl[idx]):
        w = np.flatnonzero(pl == p)
        if len(w) > 1:
            between.append(((z[w[:-1]].astype(np.float32) - z[w[1:]].astype(np.float32)) ** 2)
                           .mean((2, 3)))
    ref = np.median(np.concatenate(between), 0)
    out["static"] = {"n_wells": int(len(idx)),
                     "real_vs_static_over_between_wells_median": float(np.median(d_rs / ref)),
                     "per_clip": (np.median(d_rs, 0) / ref).round(4).tolist()}
    sel_p = a2.RES_DIR / "selection.json"
    if args.backbone == "vjepa2" and sel_p.exists():
        sel = json.loads(sel_p.read_text())["selected"]["h1_learned"]
        models = []
        for s in a2.protocol()["grid"]["seeds"]:
            p = a2.CK_DIR / f"{a2.run_name(args.norm, 'vjepa2', 'h1_learned', sel, s)}.pt"
            mod, bat, rec, _ = a2.load_model(p, c)
            models.append((mod, bat))
        zmod = z.copy()
        zmod[idx, 0] = zs[:, 0].astype(z.dtype)
        e_real, _ = E.seed_errors(models, z, idx)
        e_stat, _ = E.seed_errors(models, zmod, idx)
        out["static"]["predictor_error_24_36h"] = {"real_context": float(E.primary(e_real).mean()),
                                                    "static_context": float(E.primary(e_stat)
                                                                            .mean())}
    fp = c.fingerprint.astype(np.float32)
    fp -= fp.mean(1, keepdims=True)
    fp /= np.linalg.norm(fp, axis=1, keepdims=True) + 1e-9
    tr = np.flatnonzero((m.plate_split == "train").to_numpy())
    ev = np.flatnonzero((m.plate_split == args.split).to_numpy())
    best = np.full(len(ev), -1.0, np.float32)
    for i in range(0, len(ev), 1024):
        best[i:i + 1024] = (fp[ev[i:i + 1024]] @ fp[tr].T).max(1)
    out["leakage"] = {"n_eval_wells": int(len(ev)), "n_train_wells": int(len(tr)),
                      "max_r": float(best.max()), "median_max_r": float(np.median(best)),
                      "n_match_r>=0.98": int((best >= MATCH_R).sum()),
                      "n_report_r>=0.90": int((best >= REPORT_R).sum())}
    o = a2.RES_DIR / "controls"
    o.mkdir(parents=True, exist_ok=True)
    (o / f"controls_{args.split}_{args.norm}_{args.backbone}{args.tag}.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("static", "leakage")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
