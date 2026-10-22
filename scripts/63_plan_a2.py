"""Amendment 02 H3 planning and G-Plan-2 (sections 4 and 6; amendment 01 section 1).

Planner: the planning source's true-encoding models (three seeds; scores averaged over seeds).
(a) Phenotype targets, unseen genes of the split with an encoding. z*_P = mean observed clip-12
    (primary; clip 8 secondary) tile-mean latent of training wells (training plates x training
    genes) whose MitoCheck classes include P. Start = negative-control clip-0 tile tokens of 20
    seeded training plates; score = -distance of the rolled-out state to z*_P, averaged over
    starts and seeds. Truth: primary = amend02_truth_<P> (screenA terms); stringent =
    amendment-01 validated hits. Metrics AP, AUROC, hits/enrichment @10/50/100; nulls: 10,000
    label permutations and 10,000 within HeLa-expression bins; baselines: guilt-by-association
    (cosine to the mean encoding of training genes in the truth list), kNN-imputed endpoint
    retrieval, and the oracle ceiling (replicate-endpoint retrieval: observed clip-12 mean of
    the gene's own wells, i.e. data the planner never receives; only
    genes with embedded wells, reported on that subset with the planner on the same subset).
    Rule: AP beats permutation (BH q < 0.05) for >= 2 of 3 phenotypes and beats
    guilt-by-association for >= 1 (primary truth).
(b) Trajectory targets: up to 500 seeded wells of primary unseen genes; start = the well's clip 0;
    candidates = every embedded library gene with an encoding; planner score = -mean squared
    distance at clips 8 and 12; oracle ceiling (replicate-endpoint retrieval) = the candidate's
    observed mean
    clip-12 latent over its other wells; kNN-imputed retrieval as stage 1. Rule: hit@100 above
    random (one-sided binomial p < 0.05) and above kNN-imputed endpoint retrieval (erratum
    E-02.1, docs/DESIGN_AMENDMENT_02.md section 10; bootstrap over
    target genes, 95% CI of the hit@100 difference above 0).
G-Plan-2 passes when both (a) and (b) pass.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import cache  # noqa: E402
from geneworld.annotations import PRIMARY  # noqa: E402
from geneworld.eval.stats import bh  # noqa: E402
from geneworld.train import a2  # noqa: E402
from geneworld.train.batcher import DEV  # noqa: E402

K_NN = 10
# Erratum E-02.1 (2026-10-22): the trajectory gate comparator is kNN-imputed endpoint retrieval
# (no data from the unseen gene). Replicate-endpoint retrieval is reported as an oracle ceiling.
TRAJECTORY_GATE_COMPARATOR = "retrieval_knn"
ORACLE = "oracle_ceiling_replicate_endpoint"


@torch.no_grad()
def rollouts(model, start, acts, bs=2048):
    """start (K, D) tile tokens; acts (C, A) -> tile-mean forecast (C, 12, D)."""
    out = []
    for i in range(0, len(acts), bs):
        a = acts[i:i + bs]
        n = len(a)
        z0 = start[None, None].expand(n, 1, -1, -1).contiguous()
        ctrl = torch.zeros(n, dtype=torch.bool, device=DEV)
        sir = torch.zeros(n, dtype=torch.long, device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model.rollout(z0, a, ctrl, sir, 12)
        out.append(p.float().mean(2))
    return torch.cat(out)


def nrm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def impute(cands, enc, train_genes, ep, k=K_NN):
    tg = [g for g in train_genes if g in enc]
    e = nrm(np.stack([enc[g] for g in tg]))
    E = np.stack([ep[g] for g in tg])
    out = []
    for g in cands:
        s = e @ nrm(enc[g])
        order = [j for j in np.argsort(-s) if tg[j] != g][:k]
        out.append(E[order].mean(0))
    return np.stack(out)


def ap_score(s, y):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def metrics(s, y, ks=(10, 50, 100)):
    from sklearn.metrics import roc_auc_score
    o = np.argsort(-s)
    prev = float(y.mean())
    r = {"ap": ap_score(s, y), "auroc": float(roc_auc_score(y, s)), "prevalence": prev,
         "n_pos": int(y.sum()), "n": int(len(y))}
    for k in ks:
        h = int(y[o[:k]].sum())
        r[f"hits@{k}"] = h
        r[f"enrichment@{k}"] = h / k / prev if prev else None
    return r


def perm(s, y, n, bins=None, seed=0):
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    obs = ap_score(s, y)
    null = np.empty(n)
    groups = None if bins is None else [np.flatnonzero(bins == b) for b in np.unique(bins)]
    for i in range(n):
        if groups is None:
            lab = rng.permutation(y)
        else:
            lab = y.copy()
            for w in groups:
                lab[w] = y[rng.permutation(w)]
        null[i] = average_precision_score(lab, s)
    return {"ap": obs, "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95)),
            "p_one_sided": float((1 + np.sum(null >= obs)) / (n + 1))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--norm", default="plate", choices=["plate", "global"])
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--gene-split", default="gene", choices=["gene", "complex"])
    ap.add_argument("--source", default="", help="default: planning source from H2 validation")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n-targets", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--freeze-commit", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.freeze_commit:
        sys.exit("test genes are evaluated once, after freezing: pass --freeze-commit")
    src = args.source or json.loads((a2.RES_DIR / "h2" / "h2_val_plate_vjepa2.json")
                                    .read_text())["planning_source"]
    sel = json.loads((a2.RES_DIR / "selection.json").read_text())["selected"]
    c = cache.load(args.backbone, args.norm, gene_split=args.gene_split, suffix=args.suffix)
    m, z = c.meta, c.z
    gs = args.gene_split
    fam = f"h2:{src}"
    models, recs = [], []
    for s in a2.protocol()["grid"]["seeds"]:
        p = a2.CK_DIR / f"{a2.run_name(args.norm, args.backbone, fam, sel[fam], s, gs)}.pt"
        mod, bat, rec, d = a2.load_model(p, c)
        models.append(mod)
        recs.append(rec)
    enc = a2.family_encoding(fam, 0)
    act = m.action.to_numpy()
    lib = (m.control == "").to_numpy()
    zt = z[:, :13].astype(np.float32).mean(2)                      # (N, 13, D)
    train = a2.train_mask(m, fam, 0)
    train_genes = sorted({g for g in act[train & lib] if g in enc})
    ep_train = {g: zt[train & (act == g), 12].mean(0) for g in train_genes}
    split = pd.read_csv(REPO / "splits" / ("gene_split_complex.tsv" if gs == "complex"
                                           else "gene_split.tsv"), sep="\t")
    gsplit = dict(zip(split.gene, split.split, strict=True))
    res = {"split": args.split, "norm": args.norm, "backbone": args.backbone, "gene_split": gs,
           "source": src, "models": recs, "design_commit": a2.DESIGN_COMMIT,
           "freeze_commit": args.freeze_commit or None}

    # ---------------- (a) phenotype targets
    freeze = json.loads((REPO / "manifests" / "amend02_freeze.json").read_text())["truth"]
    vh = json.loads((REPO / "manifests" / "validated_hits.json").read_text())["lists"]
    truths = {}
    for p in PRIMARY:
        for kind, f, h in (("primary", f"amend02_truth_{p}.tsv", freeze[p]["sha256"]),
                           ("stringent", vh[p]["file"], vh[p]["sha256"])):
            path = REPO / "data" / "annotations" / f
            if hashlib.sha256(path.read_bytes()).hexdigest() != h:
                sys.exit(f"{f} does not match its frozen SHA-256")
            truths[(kind, p)] = set(pd.read_csv(path, sep="\t")["Gene Symbol"])
    bins = pd.read_csv(REPO / "data" / "ref" / "hela_expression_bins.tsv", sep="\t",
                       index_col=0)["bin"].to_dict()
    negtr = train & (m.control == "negative control").to_numpy()
    tplates = sorted(set(m.plate[negtr]))
    pick = np.random.default_rng(2).choice(tplates, min(20, len(tplates)), replace=False)
    starts = [torch.tensor(z[negtr & (m.plate == pl).to_numpy(), 0].astype(np.float32)
                           .mean(0), device=DEV) for pl in pick]
    cls = m.classes.to_numpy()
    ph = {"start_plates": list(map(str, pick)), "sets": {}}
    for set_name, genes in (("unseen", sorted(g for g, sp in gsplit.items()
                                              if sp == args.split and g in enc)),
                            ("seen", train_genes)):
        A = torch.tensor(np.stack([enc[g] for g in genes]), device=DEV)
        P = np.mean([[rollouts(mod, s0, A).cpu().numpy() for s0 in starts] for mod in models],
                    (0, 1))                                        # (G, 12, D)
        imp = impute(genes, enc, train_genes, ep_train)
        eb = np.array([bins.get(g, -1) for g in genes])
        obs_ep = {g: zt[lib & (act == g), 12].mean(0) for g in genes
                  if (lib & (act == g)).any()}
        has_obs = np.array([g in obs_ep for g in genes])
        out = {"n_genes": len(genes), "n_with_embedded_wells": int(has_obs.sum()), "truth": {}}
        for kind in ("primary", "stringent"):
            phen, pv, gba_wins = {}, [], []
            for p in PRIMARY:
                y = np.array([g in truths[(kind, p)] for g in genes])
                wells = train & lib & np.array([p in x for x in cls])
                r = {"n_target_wells": int(wells.sum())}
                if y.sum() == 0 or wells.sum() == 0:
                    phen[p] = {**r, "skipped": "no positives or no target wells"}
                    pv.append(None)
                    continue
                th = [g for g in train_genes if g in truths[(kind, p)]]
                gba = nrm(np.stack([enc[g] for g in genes])) @ nrm(
                    np.stack([enc[g] for g in th]).mean(0)) if th else np.zeros(len(genes))
                for hz, step in (("primary_clip12", 11), ("secondary_clip8", 7)):
                    tgt = zt[wells, step + 1].mean(0)
                    score = -((P[:, step] - tgt) ** 2).mean(1)
                    rr = {"planner": metrics(score, y),
                          "perm": perm(score, y, args.n_perm),
                          "perm_expression_matched": perm(score, y, args.n_perm, bins=eb)}
                    if step == 11:
                        rr["gba"] = metrics(gba, y)
                        rr["gba_perm"] = perm(gba, y, args.n_perm)
                        rr["n_train_truth_genes_for_gba"] = len(th)
                        rr["retrieval_knn"] = metrics(-((imp - tgt) ** 2).mean(1), y)
                        if has_obs.sum() and y[has_obs].sum() and (~y[has_obs]).sum():
                            ob = np.stack([obs_ep[g] for g in np.array(genes)[has_obs]])
                            rr["subset_with_embedded_wells"] = {
                                "planner": metrics(score[has_obs], y[has_obs]),
                                ORACLE: metrics(
                                    -((ob - tgt) ** 2).mean(1), y[has_obs])}
                    r[hz] = rr
                phen[p] = r
                pv.append(r["primary_clip12"]["perm"]["p_one_sided"])
                gba_wins.append(r["primary_clip12"]["planner"]["ap"] >
                                r["primary_clip12"]["gba"]["ap"])
            q = bh(pv)
            for p, qq in zip(PRIMARY, q, strict=True):
                if "primary_clip12" in phen[p]:
                    phen[p]["primary_clip12"]["perm"]["q_bh"] = qq
            n_sig = sum(1 for qq in q if qq is not None and qq < 0.05)
            out["truth"][kind] = {"phenotypes": phen, "decision": {
                "n_q<0.05": n_sig, "n_beats_gba": int(sum(gba_wins)),
                "pass": bool(n_sig >= 2 and sum(gba_wins) >= 1)}}
            print(set_name, kind, out["truth"][kind]["decision"],
                  {p: round(v["primary_clip12"]["planner"]["ap"], 4) for p, v in phen.items()
                   if "primary_clip12" in v})
        ph["sets"][set_name] = out
    res["phenotype_targets"] = ph

    # ---------------- (b) trajectory targets
    cands = sorted({g for g in act[lib] if g in enc})
    cix = {g: i for i, g in enumerate(cands)}
    A = torch.tensor(np.stack([enc[g] for g in cands]), device=DEV)
    tmask = a2.eval_mask(m, f"h2_{args.split}", set(), enc, primary=True)
    tgt = np.flatnonzero(tmask)
    tgt = np.sort(np.random.default_rng(0).choice(tgt, min(args.n_targets, len(tgt)),
                                                  replace=False))
    imp = impute(cands, enc, train_genes, ep_train)
    wells_of = {g: np.flatnonzero(lib & (act == g)) for g in cands}
    methods = ["planner", "retrieval_knn", ORACLE]
    ranks = {k: [] for k in methods}
    for i in tgt:
        g = act[i]
        obs = torch.tensor(zt[i, [8, 12]], device=DEV)
        s0 = torch.tensor(z[i, 0].astype(np.float32), device=DEV)
        pr = torch.stack([rollouts(mod, s0, A) for mod in models]).mean(0)   # (C, 12, D)
        sc = {"planner": -((pr[:, [7, 11]] - obs) ** 2).mean((1, 2)).cpu().numpy(),
              "retrieval_knn": -((imp - zt[i, 12]) ** 2).mean(1)}
        ob = np.full(len(cands), -np.inf)
        for cg, w in wells_of.items():
            w = w[w != i]
            if len(w):
                ob[cix[cg]] = -((zt[w, 12].mean(0) - zt[i, 12]) ** 2).mean()
        sc[ORACLE] = ob
        for k, s in sc.items():
            ranks[k].append(int(np.flatnonzero(np.argsort(-s) == cix[g])[0]) + 1)
    C = len(cands)
    tr = {"n_targets": int(len(tgt)), "n_target_genes": int(len(set(act[tgt]))),
          "n_candidates": C, "methods": {}}
    for k in methods:
        r = np.array(ranks[k])
        tr["methods"][k] = {"hit@1": float(np.mean(r <= 1)), "hit@10": float(np.mean(r <= 10)),
                            "hit@100": float(np.mean(r <= 100)), "mrr": float(np.mean(1 / r)),
                            "median_rank": float(np.median(r))}
    tr["random"] = {"hit@1": 1 / C, "hit@10": 10 / C, "hit@100": 100 / C}
    from scipy.stats import binomtest
    hp = np.array(ranks["planner"]) <= 100
    hr = np.array(ranks[TRAJECTORY_GATE_COMPARATOR]) <= 100
    tg = act[tgt]
    ug = np.unique(tg)
    per = np.array([[hp[tg == g].mean(), hr[tg == g].mean()] for g in ug])
    bs = per[np.random.default_rng(1).integers(0, len(ug), (10_000, len(ug)))]
    dd = bs[..., 0].mean(1) - bs[..., 1].mean(1)
    tr["planner_vs_random_hit@100_p"] = float(binomtest(int(hp.sum()), len(hp), 100 / C,
                                                        alternative="greater").pvalue)
    tr["gate_comparator"] = TRAJECTORY_GATE_COMPARATOR
    tr["oracle_ceiling_label"] = "oracle ceiling (uses the unseen gene's observed wells)"
    tr[f"planner_vs_{TRAJECTORY_GATE_COMPARATOR}_hit@100"] = {
        "difference": float(per[:, 0].mean() - per[:, 1].mean()),
        "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    tr["pass"] = bool(tr["planner_vs_random_hit@100_p"] < 0.05 and
                      tr[f"planner_vs_{TRAJECTORY_GATE_COMPARATOR}_hit@100"]["ci95"][0] > 0)
    res["trajectory_targets"] = tr
    print(json.dumps(tr["methods"]))
    phen_pass = ph["sets"]["unseen"]["truth"]["primary"]["decision"]["pass"]
    res["G_Plan_2"] = {"rule": "phenotype (primary truth, unseen genes) and trajectory rules; "
                               "trajectory: hit@100 above random (binomial p < 0.05) and above "
                               "kNN-imputed endpoint retrieval (bootstrap CI over target genes "
                               "above 0), erratum E-02.1",
                       "erratum": "E-02.1",
                       "phenotype_pass": phen_pass, "trajectory_pass": tr["pass"],
                       "pass": bool(phen_pass and tr["pass"])}
    res["code_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                        capture_output=True, text=True).stdout.strip()
    out_dir = a2.RES_DIR / "h3"
    out_dir.mkdir(parents=True, exist_ok=True)
    gtag = "" if gs == "gene" else "_complex"
    (out_dir / f"h3_{args.split}_{args.norm}_{args.backbone}_{src}{gtag}{args.tag}.json"
     ).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["G_Plan_2"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
