"""Amendment 02 evaluation helpers: seed-averaged forecasts and errors, replicate baselines,
phenotype probe with cluster-bootstrap CIs, plate-level and gene-level tests.

Latent arrays are (N, 14, 4, D) in the chosen preprocessing (per-plate control normalisation
for the gates). Forecasts cover clips 1..12 from clip 0; index h - 1 is horizon step h.
"""
from __future__ import annotations

import numpy as np
import torch

from geneworld.data import N_STEPS
from geneworld.train.a2 import PRIMARY_STEPS
from geneworld.train.batcher import DEV


@torch.no_grad()
def forecast(model, z, bat, idx, ctx_from=None, bs=256) -> np.ndarray:
    """(n, 12, K, D) float16 rollout from clip 0 (optionally another well's clip 0)."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        src = j if ctx_from is None else ctx_from[i:i + bs]
        a, c, s = bat.get(torch.tensor(j, device=DEV))
        z0 = torch.from_numpy(z[src, :1].astype(np.float32)).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model.rollout(z0, a, c, s, N_STEPS - 1)
        out.append(p.float().cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def errors(pred, z, idx) -> np.ndarray:
    """(n, 12) MSE per well and horizon."""
    obs = z[idx, 1:N_STEPS].astype(np.float32)
    return ((pred.astype(np.float32) - obs) ** 2).mean((2, 3))


def seed_errors(models, z, idx, **kw) -> tuple[np.ndarray, list[np.ndarray]]:
    """Errors averaged over seeds (section 4) and the per-seed forecasts."""
    preds = [forecast(m, z, b, idx, **kw) for m, b in models]
    return np.mean([errors(p, z, idx) for p in preds], 0), preds


def primary(err: np.ndarray, steps=PRIMARY_STEPS) -> np.ndarray:
    return err[:, [h - 1 for h in steps]].mean(1)


def replicate_baselines(z, meta, idx, train_mask) -> tuple[dict, dict]:
    """Replicate drift and replicate endpoint (section 3) with the protocol's fallbacks."""
    sir = meta.sirna.to_numpy()
    act = meta.action.to_numpy()
    plates = meta.plate.to_numpy()
    neg = (meta.control == "negative control").to_numpy()
    tm = z[:, :N_STEPS].astype(np.float32)
    tr = np.flatnonzero(train_mask)
    by_s, by_g = {}, {}
    for i in tr:
        if sir[i]:
            by_s.setdefault(sir[i], []).append(i)
        by_g.setdefault(act[i], []).append(i)
    clip_mean = tm[tr].mean((0, 2))[:, None, :]
    drift = np.empty((len(idx), N_STEPS - 1) + z.shape[2:], np.float32)
    endp = np.empty_like(drift)
    used = {"sirna": 0, "gene_other_sirnas": 0, "fallback": 0}
    for r, i in enumerate(idx):
        w = by_s.get(sir[i], [])
        kind = "sirna"
        if not w:
            w = [j for j in by_g.get(act[i], []) if sir[j] != sir[i]]
            kind = "gene_other_sirnas"
        if w:
            rep = tm[w].mean(2, keepdims=True)          # tiles are exchangeable fields of view
            drift[r] = tm[i, :1] + (rep[:, 1:] - rep[:, :1]).mean(0)
            endp[r] = rep[:, 1:].mean(0)
        else:
            kind = "fallback"
            pn = np.flatnonzero(neg & (plates == plates[i]))
            pn = pn[pn != i]
            cm = tm[pn].mean((0, 2))[:, None, :] if len(pn) else clip_mean
            drift[r] = tm[i, :1] + cm[1:] - cm[:1]
            endp[r] = clip_mean[1:]
        used[kind] += 1
    return {"replicate_drift": drift, "replicate_endpoint": endp}, used


def stage1_baselines(z, meta, idx, train_mask) -> dict:
    """copy-last, time-slot mean/drift and plate control mean/drift in this preprocessing."""
    tm = z[:, :N_STEPS].astype(np.float32)
    slot = tm[train_mask].mean(0).mean(1, keepdims=True)              # (13, 1, D)
    z0 = tm[idx, :1]
    out = {"copy_last": np.repeat(z0, N_STEPS - 1, 1),
           "slot_mean": np.broadcast_to(slot[1:], (len(idx), N_STEPS - 1) + z.shape[2:]),
           "slot_drift": z0 + (slot[1:] - slot[:1])[None]}
    neg = (meta.control == "negative control").to_numpy()
    plates = meta.plate.to_numpy()
    cm = np.zeros((len(idx), N_STEPS) + z.shape[2:], np.float32)
    for r, i in enumerate(idx):
        pn = np.flatnonzero(neg & (plates == plates[i]))
        pn = pn[pn != i]
        cm[r] = tm[pn].mean((0, 2))[:, None, :] if len(pn) else np.nan
    out["control_mean"] = cm[:, 1:]
    out["control_drift"] = z0 + cm[:, 1:] - cm[:, :1]
    return out


def sign_flip(d, n=100_000, seed=0) -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"n_groups": int(len(d)), "mean": None, "p_one_sided": None}
    rng = np.random.default_rng(seed)
    flips = (rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(1)
    return {"n_groups": int(len(d)), "mean": float(d.mean()),
            "p_one_sided": float((1 + np.sum(flips <= d.mean())) / (n + 1)),
            "frac_groups_better": float(np.mean(d < 0))}


def plate_test(e_model, e_base, plates) -> dict:
    """Per-plate mean of model - baseline; one-sided Monte Carlo sign flip over plates."""
    ps = np.unique(plates)
    a = np.array([np.nanmean(e_model[plates == p]) for p in ps])
    b = np.array([np.nanmean(e_base[plates == p]) for p in ps])
    t = sign_flip(a - b)
    t["model"] = float(np.nanmean(a))
    t["baseline"] = float(np.nanmean(b))
    t["relative_change"] = float(np.nanmean(a) / np.nanmean(b) - 1)
    return t


def well_weighted_plate_test(e_model, e_base, plates) -> dict:
    """Secondary (completion plan v2, F6): as `plate_test`, but each plate's mean difference
    is weighted by its number of wells, so the statistic is the well-weighted mean difference;
    one-sided Monte Carlo sign flip over plates (signs flip per plate, not per well)."""
    ps = np.unique(plates)
    d = np.array([np.nanmean(e_model[plates == p] - e_base[plates == p]) for p in ps])
    n = np.array([np.sum(plates == p) for p in ps], float)
    ok = np.isfinite(d)
    t = sign_flip(d[ok] * n[ok] * ok.sum() / n[ok].sum())
    t["weighting"] = "wells per plate"
    t["n_wells"] = int(n[ok].sum())
    t["wells_per_plate"] = {"min": int(n[ok].min()), "median": float(np.median(n[ok])),
                            "max": int(n[ok].max())} if ok.any() else None
    return t


def gene_boot(a: dict, b: dict, n=10_000, seed=0) -> dict:
    gs = sorted(set(a) & set(b))
    d = np.array([a[g] - b[g] for g in gs])
    if len(d) < 10:
        return {"n_genes": len(gs), "mean": float(d.mean()) if len(d) else None,
                "ci95": None, "reason": "fewer than 10 genes"}
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return {"n_genes": len(gs), "mean": float(d.mean()),
            "relative": float(np.mean([a[g] for g in gs]) / np.mean([b[g] for g in gs]) - 1),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "frac_genes_better": float(np.mean(d < 0))}


def per_gene(err: np.ndarray, genes: np.ndarray) -> dict:
    return {g: float(np.mean(err[genes == g])) for g in np.unique(genes)}


def probe_with_ci(x_tr, y_tr, evals: dict, y_ev, plates_ev, n_boot=2000, seed=0,
                  diffs=(("action_conditioned", "action_agnostic"),
                         ("action_conditioned", "replicate_drift"))) -> dict | None:
    """AUROC per evaluated representation, with cluster-bootstrap (plates) CIs, and of the
    named differences. `evals`: name -> list of (n, D) arrays (one per seed; scores averaged)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    n_pos, n_neg = int(y_ev.sum()), int((~y_ev).sum())
    res = {"n_pos_eval": n_pos, "n_neg_eval": n_neg, "n_pos_train": int(y_tr.sum())}
    if y_tr.sum() < 10 or n_pos < 3 or n_neg < 3:
        res["skipped"] = "too few positives"
        return res
    clf = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(x_tr, y_tr)
    scores = {k: np.mean([clf.decision_function(x) for x in xs], 0) for k, xs in evals.items()}
    ps = np.unique(plates_ev)
    idx = {p: np.flatnonzero(plates_ev == p) for p in ps}
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in scores}
    dboots = {f"{a}-{b}": [] for a, b in diffs if a in scores and b in scores}
    for _ in range(n_boot):
        take = np.concatenate([idx[p] for p in rng.choice(ps, len(ps), replace=True)])
        y = y_ev[take]
        if y.all() or not y.any():
            continue
        au = {k: roc_auc_score(y, s[take]) for k, s in scores.items()}
        for k in boots:
            boots[k].append(au[k])
        for a, b in diffs:
            if f"{a}-{b}" in dboots:
                dboots[f"{a}-{b}"].append(au[a] - au[b])
    res["auroc"] = {k: {"value": float(roc_auc_score(y_ev, s)),
                        "ci95": [float(np.percentile(boots[k], 2.5)),
                                 float(np.percentile(boots[k], 97.5))]}
                    for k, s in scores.items()}
    res["difference"] = {k: {"value": float(res["auroc"][k.split("-")[0]]["value"] -
                                            res["auroc"][k.split("-")[1]]["value"]),
                             "ci95": [float(np.percentile(v, 2.5)),
                                      float(np.percentile(v, 97.5))]}
                         for k, v in dboots.items()}
    res["n_plates"] = int(len(ps))
    return res


def mask_screenB_sirna(model_lists, meta) -> dict:
    """Set the reagent token of every screenB well to 'unknown' (index 0) in each batcher
    (author instruction 2026-10-21); returns how many screenB wells had a known siRNA."""
    isb = np.flatnonzero((meta.plate_split == "screenB").to_numpy())
    known = 0
    for models in model_lists:
        for _, bat in models:
            t = torch.tensor(isb, device=bat.sirna.device)
            known = max(known, int((bat.sirna[t] != 0).sum()))
            bat.sirna[t] = 0
    return {"screenB_wells": int(len(isb)), "wells_with_known_sirna_before_mask": known}
