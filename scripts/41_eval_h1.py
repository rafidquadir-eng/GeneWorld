"""H1 evaluation: action-conditioned forecasting on held-out plates, with controls (A.7–A.9).

  --split val    validation plates (used for G-Signal and all selection)
  --split test   test plates; refused unless --freeze-commit names the commit that froze the
                 models (evaluated once)

Primary comparison (pre-specified): per plate, mean over library wells of the forecast MSE
averaged over horizons 3, 12, 24 and 36 h; model minus baseline; one-sided Monte Carlo
sign-flip test over plates. G-Signal: the action-conditioned predictor beats the
action-agnostic predictor and the time-slot mean with p < 0.05.

Controls: context swap (each well's clip-0 context replaced by that of a well with the same
action on another plate), a plate-identity probe on predicted versus observed 36 h latents,
and a phenotype-class probe (trained on observed training-well latents) applied to predicted
versus observed 36 h latents.
"""
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import sets  # noqa: E402
from geneworld.data import HOURS_PER_STEP, load  # noqa: E402
from geneworld.eval import h1  # noqa: E402
from geneworld.train.batcher import from_checkpoint  # noqa: E402

PHEN = ["mitotic_delay", "polylobed", "binuclear", "cell_death", "nuclear_shape", "segregation"]


def per_plate(err: np.ndarray, plates: np.ndarray) -> dict[str, float]:
    return {p: float(np.nanmean(err[plates == p])) for p in np.unique(plates)}


def probe_auc(x_tr, y_tr, x_ev, y_ev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    if y_tr.sum() < 10 or y_ev.sum() < 3 or y_ev.sum() == len(y_ev):
        return None
    clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(x_tr, y_tr)
    return [float(roc_auc_score(y_ev, clf.decision_function(x))) for x in x_ev]


def plate_probe(x_list, plates, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    keep = np.isin(plates, [p for p in np.unique(plates) if (plates == p).sum() >= 5])
    if len(np.unique(plates[keep])) < 2:
        return None
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = [float(cross_val_score(LogisticRegression(C=0.1, max_iter=2000), x[keep],
                                 plates[keep], cv=cv).mean()) for x in x_list]
    return {"accuracy": acc, "chance": float(1 / len(np.unique(plates[keep])))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--gene-split", choices=["gene", "complex"], default="gene")
    ap.add_argument("--model", required=True, help="action-conditioned checkpoint name")
    ap.add_argument("--agnostic", required=True, help="action-agnostic checkpoint name")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--freeze-commit", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.freeze_commit:
        sys.exit("test plates are evaluated once, after freezing: pass --freeze-commit")
    ck = REPO / "checkpoints" / "predictors"
    lat = load(args.backbone, gene_split=args.gene_split)
    m, z = lat.meta, lat.z
    model, bat, rec, d = from_checkpoint(ck / f"{args.model}.pt", lat)
    agn, bat_n, rec_n, _ = from_checkpoint(ck / f"{args.agnostic}.pt", lat)
    vocab = set(d["vocab"]) | {"NEG"}
    mk = sets.masks(m, vocab)
    idx = np.flatnonzero(mk[f"h1_{args.split}"])
    lib = mk["library"][idx]
    plates = m.plate.to_numpy()[idx]
    print(f"H1 {args.split}: {len(idx)} wells ({lib.sum()} library) on "
          f"{len(np.unique(plates))} plates")

    preds = h1.baselines(z, m, idx, mk["train"])
    preds["action_conditioned"] = h1.model_forecast(model, z, bat, idx)
    preds["action_agnostic"] = h1.model_forecast(agn, z, bat_n, idx)
    # context swap: same action, another plate
    rng = np.random.default_rng(0)
    act = m.action.to_numpy()
    allp = m.plate.to_numpy()
    swap = idx.copy()
    for r, i in enumerate(idx):
        cand = np.flatnonzero((act == act[i]) & (allp != allp[i]))
        if len(cand):
            swap[r] = rng.choice(cand)
    has_swap = swap != idx
    preds["context_swap"] = h1.model_forecast(model, z, bat, idx, ctx_from=swap)

    err = {k: h1.errors(v, z, idx) for k, v in preds.items()}
    hcols = [h - 1 for h in h1.HORIZONS]
    res = {"split": args.split, "backbone": args.backbone, "model": rec, "agnostic": rec_n,
           "n_wells": int(len(idx)), "n_library": int(lib.sum()),
           "n_plates": int(len(np.unique(plates))),
           "horizons_h": [h * HOURS_PER_STEP for h in h1.HORIZONS],
           "design_commit": "deb443d", "freeze_commit": args.freeze_commit or None,
           "mean_error": {}, "tests": {}}
    for k, e in err.items():
        res["mean_error"][k] = {
            "all_horizons": float(np.nanmean(e[lib][:, hcols])),
            **{f"{h * HOURS_PER_STEP:g}h": float(np.nanmean(e[lib][:, h - 1]))
               for h in h1.HORIZONS}}
    main_err = {k: np.nanmean(e[:, hcols], 1) for k, e in err.items()}
    for base in ["action_agnostic", "slot_mean", "slot_drift", "copy_last", "control_mean",
                 "control_drift"]:
        a = per_plate(main_err["action_conditioned"][lib], plates[lib])
        b = per_plate(main_err[base][lib], plates[lib])
        dlt = np.array([a[p] - b[p] for p in a])
        t = h1.sign_flip_mc(dlt)
        t["relative_change"] = float(np.nanmean([a[p] for p in a]) /
                                     np.nanmean([b[p] for p in a]) - 1)
        per_h = {}
        for h in h1.HORIZONS:
            ah = per_plate(err["action_conditioned"][lib][:, h - 1], plates[lib])
            bh_ = per_plate(err[base][lib][:, h - 1], plates[lib])
            per_h[f"{h * HOURS_PER_STEP:g}h"] = h1.sign_flip_mc(
                np.array([ah[p] - bh_[p] for p in ah]))
        t["per_horizon"] = per_h
        res["tests"][f"action_conditioned_vs_{base}"] = t
    sw = lib & has_swap
    res["context_swap"] = {
        "n_wells": int(sw.sum()),
        "own_context": float(np.nanmean(main_err["action_conditioned"][sw])),
        "swapped_context": float(np.nanmean(main_err["context_swap"][sw])),
        "action_agnostic_own_context": float(np.nanmean(main_err["action_agnostic"][sw])),
    }
    # phenotype-class probe at 36 h, trained on observed training-well latents
    tr = np.flatnonzero(mk["train"] & mk["library"])
    feat = lambda x: x.mean(1)  # noqa: E731  tile mean -> (n, D)
    cls = m.classes.astype(str).to_numpy()
    probes = {}
    for c in PHEN:
        y_tr = np.array([c in s for s in cls[tr]])
        y_ev = np.array([c in s for s in cls[idx]])[lib]
        aucs = probe_auc(feat(z[tr, 12]), y_tr, [feat(z[idx, 12])[lib],
                         feat(preds["action_conditioned"][:, 11])[lib],
                         feat(preds["action_agnostic"][:, 11])[lib]], y_ev)
        probes[c] = None if aucs is None else dict(zip(
            ["observed", "action_conditioned", "action_agnostic"], aucs, strict=True))
    res["phenotype_probe_auroc_36h"] = probes
    res["plate_probe_36h"] = plate_probe(
        [feat(z[idx, 12]), feat(preds["action_conditioned"][:, 11])], plates)
    g = res["tests"]
    res["G_Signal"] = {
        "rule": "action_conditioned beats action_agnostic and slot_mean, one-sided p < 0.05 "
                "over plates (mean over 3/12/24/36 h)",
        "pass": bool(g["action_conditioned_vs_action_agnostic"]["p_one_sided"] < 0.05 and
                     g["action_conditioned_vs_action_agnostic"]["mean"] < 0 and
                     g["action_conditioned_vs_slot_mean"]["p_one_sided"] < 0.05 and
                     g["action_conditioned_vs_slot_mean"]["mean"] < 0)}
    res["code_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short",
                                         "HEAD"], capture_output=True, text=True).stdout.strip()
    out = REPO / "results" / "h1"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"h1_{args.split}_{args.backbone}{args.tag}.json"
    path.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("mean_error", "G_Signal", "context_swap")}, indent=1))
    for k, t in g.items():
        print(f"{k}: mean diff {t['mean']:+.4f} ({t['relative_change']:+.1%}), "
              f"p = {t['p_one_sided']:.4g}, plates better {t.get('frac_groups_better', 0):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
