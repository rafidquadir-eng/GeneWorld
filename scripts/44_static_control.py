"""Static-clip control (design A.7(2)).

`--make-wells` writes `splits/static_control_wells.tsv`: 300 seeded validation-plate wells
(library and negative controls); they are then embedded with `20_stream_embed.py --static`
(each 16-frame clip replaced by its middle frame repeated 16 times).

Evaluation (V-JEPA 2 and any backbone with static latents):
  * encoder motion sensitivity: per well and clip, the squared distance between the real and
    the static clip latent, relative to the median squared distance between the real latents
    of two different wells of the same plate at the same clip;
  * predictor: forecast MSE against the real future when the clip-0 context is the static
    clip versus the real clip, and the mean predicted one-step change in each case.
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

from geneworld.data import N_STEPS, load  # noqa: E402
from geneworld.eval import h1  # noqa: E402
from geneworld.train.batcher import from_checkpoint  # noqa: E402

WELLS = REPO / "splits" / "static_control_wells.tsv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-wells", action="store_true")
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--model", default="")
    args = ap.parse_args()
    if args.make_wells:
        st = pd.read_csv(REPO / "splits" / "stage1_wells.tsv", sep="\t")
        v = st[(st.plate_split == "val") & (st.control.isna() | (st.control == "negative control"))]
        v.sample(300, random_state=0).sort_values(["Plate", "Well"]).to_csv(WELLS, sep="\t",
                                                                            index=False)
        print(f"wrote {WELLS}")
        return 0
    lat = load(args.backbone)
    m = lat.meta
    key = {(p, w): i for i, (p, w) in enumerate(zip(m.plate, m.well, strict=True))}
    zs, idx = [], []
    for f in sorted((REPO / "data" / "latents_static" / args.backbone).glob("*.npz")):
        d = np.load(f)
        s = (d["latents"].astype(np.float32).mean(3) - lat.mean) / lat.std
        for i, w in enumerate(d["wells"]):
            if (f.stem, str(w)) in key:
                zs.append(s[i])
                idx.append(key[(f.stem, str(w))])
    zs, idx = np.stack(zs), np.array(idx)
    real = lat.z[idx]
    d_rs = ((real - zs) ** 2).mean((2, 3))                          # (n, 14)
    plates = m.plate.to_numpy()
    between = []
    for p in np.unique(plates[idx]):
        w = np.flatnonzero(plates == p)
        if len(w) > 1:
            a, b = lat.z[w[:-1]], lat.z[w[1:]]
            between.append(((a - b) ** 2).mean((2, 3)))
    ref = np.median(np.concatenate(between), 0)                    # (14,)
    out = {"backbone": args.backbone, "n_wells": int(len(idx)),
           "real_vs_static_over_between_wells": {
               "per_clip": (np.median(d_rs, 0) / ref).round(4).tolist(),
               "median": float(np.median(d_rs / ref))}}
    if args.model:
        model, bat, rec, _ = from_checkpoint(REPO / "checkpoints" / "predictors" /
                                             f"{args.model}.pt", lat)
        z_mod = lat.z.copy()
        z_mod[idx, 0] = zs[:, 0]
        p_real = h1.model_forecast(model, lat.z, bat, idx)
        p_stat = h1.model_forecast(model, z_mod, bat, idx)
        e_real = h1.errors(p_real, lat.z, idx)[:, [h - 1 for h in h1.HORIZONS]].mean()
        e_stat = h1.errors(p_stat, lat.z, idx)[:, [h - 1 for h in h1.HORIZONS]].mean()
        out["predictor"] = {
            "model": rec, "error_real_context": float(e_real),
            "error_static_context": float(e_stat),
            "one_step_change_real_context": float(((p_real[:, 0] - lat.z[idx, :1][:, 0]) ** 2)
                                                  .mean()),
            "one_step_change_static_context": float(((p_stat[:, 0] - zs[:, 0]) ** 2).mean())}
    assert N_STEPS == 13
    (REPO / "results" / "controls").mkdir(parents=True, exist_ok=True)
    (REPO / "results" / "controls" / f"static_{args.backbone}.json").write_text(
        json.dumps(out, indent=1))
    print(json.dumps(out, indent=1, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
