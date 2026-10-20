"""Amendment 02 H1 evaluation and G-Signal-2.

Models: the selected h1_learned and h1_none configurations (results/a2/selection.json), three
seeds each; per well and horizon the three seeds' errors are averaged (section 4).
Primary: primary wells (splits/amend02_h1_primary_wells.tsv) on the evaluated plates, mean
error at 24 h and 36 h; per-plate means; one-sided Monte Carlo sign-flip over plates.
G-Signal-2: action-conditioned beats (a) action-agnostic and (b) replicate drift, p < 0.05 each.
Secondary: all library wells of seen genes; 3 h and 12 h; per horizon; every stage-1 baseline;
replicate endpoint; context swap; phenotype probe with cluster-bootstrap CIs (section 9);
plate-identity probe on observed and forecast 36 h latents.

  --split val | test | screenB     (test / screenB require --freeze-commit)
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

from geneworld import cache  # noqa: E402
from geneworld.annotations import PRIMARY  # noqa: E402
from geneworld.eval import a2eval as E  # noqa: E402
from geneworld.train import a2  # noqa: E402

CLASSES = PRIMARY + ["cell_death", "nuclear_shape", "segregation"]


def models_for(family, cfg, norm, backbone, c, gene_split="gene"):
    out, recs = [], []
    for s in a2.protocol()["grid"]["seeds"]:
        p = a2.CK_DIR / f"{a2.run_name(norm, backbone, family, cfg, s, gene_split)}.pt"
        m, b, r, d = a2.load_model(p, c)
        out.append((m, b))
        recs.append(r)
    return out, recs, d


def plate_probe(xs, plates):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    keep = np.isin(plates, [p for p in np.unique(plates) if (plates == p).sum() >= 5])
    if len(np.unique(plates[keep])) < 2:
        return None
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return {k: float(cross_val_score(LogisticRegression(C=0.1, max_iter=3000), x[keep],
                                     plates[keep], cv=cv).mean()) for k, x in xs.items()} | {
        "chance": float(1 / len(np.unique(plates[keep]))),
        "n_plates": int(len(np.unique(plates[keep])))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test", "screenB"])
    ap.add_argument("--norm", default="plate", choices=["plate", "global"])
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--freeze-commit", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split != "val" and not args.freeze_commit:
        sys.exit("held-out test sets are evaluated once, after freezing: pass --freeze-commit")
    sel = json.loads((a2.RES_DIR / "selection.json").read_text())["selected"]
    c = cache.load(args.backbone, args.norm, suffix=args.suffix)
    m, z = c.meta, c.z
    AC, rec_ac, d = models_for("h1_learned", sel["h1_learned"], args.norm, args.backbone, c)
    AG, rec_ag, _ = models_for("h1_none", sel["h1_none"], args.norm, args.backbone, c)
    vocab = set(d["vocab"]) | {"NEG"}
    which = f"h1_{args.split}"
    prim = a2.eval_mask(m, which, vocab, primary=True)
    allw = a2.eval_mask(m, which, vocab, primary=False)
    idx = np.flatnonzero(allw)
    is_p = prim[idx]
    plates = m.plate.to_numpy()[idx]
    train = a2.train_mask(m, "h1_learned", 0)
    print(f"H1 {args.split} ({args.norm}): {len(idx)} library wells of seen genes, "
          f"{is_p.sum()} primary, {len(np.unique(plates[is_p]))} plates")

    err, P_ac = E.seed_errors(AC, z, idx)
    errs = {"action_conditioned": err}
    errs["action_agnostic"], P_ag = E.seed_errors(AG, z, idx)
    rep, rep_used = E.replicate_baselines(z, m, idx, train)
    base = E.stage1_baselines(z, m, idx, train)
    for k, v in {**rep, **base}.items():
        errs[k] = E.errors(v, z, idx)
    # context swap: clip 0 from a well with the same action on another plate
    rng = np.random.default_rng(0)
    act, allp = m.action.to_numpy(), m.plate.to_numpy()
    swap = idx.copy()
    for r, i in enumerate(idx):
        cand = np.flatnonzero((act == act[i]) & (allp != allp[i]))
        if len(cand):
            swap[r] = rng.choice(cand)
    errs["context_swap"], _ = E.seed_errors(AC, z, idx, ctx_from=swap)

    res = {"split": args.split, "norm": args.norm, "backbone": args.backbone,
           "design_commit": a2.DESIGN_COMMIT, "freeze_commit": args.freeze_commit or None,
           "models": {"action_conditioned": rec_ac, "action_agnostic": rec_ag},
           "selected_configs": {"h1_learned": sel["h1_learned"], "h1_none": sel["h1_none"]},
           "n_wells_all": int(len(idx)), "n_wells_primary": int(is_p.sum()),
           "n_plates_primary": int(len(np.unique(plates[is_p]))),
           "replicate_baseline_sources": rep_used,
           "cache": c.stats, "mean_error": {}, "tests": {}}
    for k, e in errs.items():
        res["mean_error"][k] = {
            "primary_wells_24_36h": float(np.nanmean(E.primary(e[is_p]))),
            "all_wells_24_36h": float(np.nanmean(E.primary(e))),
            **{f"primary_{h * 3}h": float(np.nanmean(e[is_p][:, h - 1])) for h in a2.ALL_STEPS}}
    for b in [k for k in errs if k != "action_conditioned"]:
        res["tests"][b] = {
            "primary": E.plate_test(E.primary(errs["action_conditioned"][is_p]),
                                    E.primary(errs[b][is_p]), plates[is_p]),
            "secondary_all_wells": E.plate_test(E.primary(errs["action_conditioned"]),
                                                E.primary(errs[b]), plates),
            "secondary_3_12h_primary": E.plate_test(
                E.primary(errs["action_conditioned"][is_p], a2.SECONDARY_STEPS),
                E.primary(errs[b][is_p], a2.SECONDARY_STEPS), plates[is_p]),
            "per_horizon_primary": {f"{h * 3}h": E.plate_test(
                errs["action_conditioned"][is_p][:, h - 1], errs[b][is_p][:, h - 1],
                plates[is_p]) for h in (1, 4, 8, 12)}}
    t = res["tests"]
    res["G_Signal_2"] = {
        "rule": "primary wells, 24/36 h: action-conditioned beats action-agnostic and "
                "replicate drift, one-sided p < 0.05 over plates each",
        "vs_action_agnostic": t["action_agnostic"]["primary"],
        "vs_replicate_drift": t["replicate_drift"]["primary"],
        "pass": bool(all(t[b]["primary"]["p_one_sided"] is not None and
                         t[b]["primary"]["p_one_sided"] < 0.05 and t[b]["primary"]["mean"] < 0
                         for b in ("action_agnostic", "replicate_drift")))}

    # phenotype probe (section 9) at 36 h on all evaluated library wells
    tr_lib = np.flatnonzero(train & (m.control == "").to_numpy())
    feat = lambda x: x.astype(np.float32).mean(1)  # noqa: E731  tile mean
    cls = m.classes.to_numpy()
    probes = {}
    for k in CLASSES:
        y_tr = np.array([k in x for x in cls[tr_lib]])
        y_ev = np.array([k in x for x in cls[idx]])
        probes[k] = E.probe_with_ci(
            feat(z[tr_lib, 12]), y_tr,
            {"observed": [feat(z[idx, 12])],
             "action_conditioned": [feat(p[:, 11]) for p in P_ac],
             "action_agnostic": [feat(p[:, 11]) for p in P_ag],
             "replicate_drift": [feat(rep["replicate_drift"][:, 11])],
             "replicate_endpoint": [feat(rep["replicate_endpoint"][:, 11])]},
            y_ev, plates)
    res["phenotype_probe_36h"] = probes
    res["plate_probe_36h"] = plate_probe(
        {"observed": feat(z[idx, 12]),
         "action_conditioned": np.mean([feat(p[:, 11]) for p in P_ac], 0)}, plates)
    res["code_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                        capture_output=True, text=True).stdout.strip()
    out = a2.RES_DIR / "h1"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"h1_{args.split}_{args.norm}_{args.backbone}{args.tag}.json"
    f.write_text(json.dumps(res, indent=1, default=float))
    g = res["G_Signal_2"]
    print(f"G-Signal-2 {'PASS' if g['pass'] else 'fail'}: vs agnostic "
          f"{g['vs_action_agnostic']['relative_change']:+.1%} p={g['vs_action_agnostic']['p_one_sided']:.3g}; "
          f"vs replicate drift {g['vs_replicate_drift']['relative_change']:+.1%} "
          f"p={g['vs_replicate_drift']['p_one_sided']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
