"""H3 planning: latent model-predictive search over gene actions (design A.2 H3, amendment 01).

Exhaustive scoring: every candidate gene's action is rolled out from a start state for 12 clips
(36 h) and scored by its latent distance to a target (tile-mean latents, standardised space).

(a) Held-out trajectory targets. Targets are wells of unseen genes (h2_val for selection,
    h2_test once after freezing; 500 seeded wells at most). Start = the target well's own clip
    0. Candidates = every stage-1 gene with an encoding in the source (seen and unseen).
    planner_dyn    mean squared distance over clips 1..12 to the target trajectory
    planner_end    distance at clip 12 only (same rollouts; dynamics vs endpoint)
    retrieval_knn  static endpoint retrieval without dynamics: each candidate's clip-12
                   endpoint imputed as the mean observed endpoint of its 10 nearest training
                   genes in the encoding (the candidate itself excluded)
    retrieval_obs  reference only: the candidate's observed mean endpoint over its other wells
                   (uses data of the unseen gene, so it is an oracle, not a baseline)
    Metrics: rank of the true gene; hit@1/10/100; MRR; same-phenotype-class and same-complex
    rates in the top 10. Null: uniform random ranking. Planner vs retrieval_knn: paired
    bootstrap over target genes of the hit@100 difference.
(b) Phenotype targets. z*_P = mean observed clip-12 latent (primary, the last dynamic clip,
    43.5 h after imaging start; secondary clip 6, 25.5 h) of training wells MitoCheck labels P.
    Start = the negative-control clip-0 mean of each of 20 seeded training plates (scores are
    averaged over starts). Candidates = genes of the evaluated gene split with an encoding.
    Ground truth = the frozen validated hits (manifests/validated_hits.json).
    Metrics: AP, AUROC, hits and enrichment at 10/50/100. Nulls: 10,000 label permutations;
    10,000 permutations within HeLa expression bins; baselines retrieval_knn (imputed endpoint
    to z*_P) and guilt-by-association (cosine of the encoding to the mean encoding of training
    genes in the validated list). BH across phenotypes. Decision rule (amendment): on unseen
    genes AP beats the permutation null (q < 0.05) for >= 2 of 3 phenotypes and beats
    guilt-by-association for >= 1.
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

from geneworld import sets  # noqa: E402
from geneworld.data import load  # noqa: E402
from geneworld.eval.stats import bh  # noqa: E402
DESIGN_COMMIT = "deb443d"  # stage-1 (pre-amendment) pipeline
from geneworld.train.batcher import DEV, from_checkpoint  # noqa: E402

PHENOTYPES = ["mitotic_delay", "polylobed", "binuclear"]
K_NN = 10


@torch.no_grad()
def rollouts(model, z0: torch.Tensor, acts: torch.Tensor, bs: int = 2048) -> torch.Tensor:
    """z0 (K, D) start; acts (C, A) -> tile-mean forecasts (C, 12, D)."""
    out = []
    for i in range(0, len(acts), bs):
        a = acts[i:i + bs]
        n = len(a)
        start = z0[None, None].expand(n, 1, -1, -1).contiguous()
        ctrl = torch.zeros(n, dtype=torch.bool, device=DEV)
        sir = torch.zeros(n, dtype=torch.long, device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model.rollout(start, a, ctrl, sir, 12)
        out.append(p.float().mean(2))
    return torch.cat(out)


def norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def impute_endpoints(cands, enc, train_genes, endpoint_of, k=K_NN):
    tg = [g for g in train_genes if g in enc]
    e_tr = norm_rows(np.stack([enc[g] for g in tg]))
    ep_tr = np.stack([endpoint_of[g] for g in tg])
    out = []
    for g in cands:
        s = e_tr @ (enc[g] / (np.linalg.norm(enc[g]) + 1e-9))
        order = [j for j in np.argsort(-s) if tg[j] != g][:k]
        out.append(ep_tr[order].mean(0))
    return np.stack(out)


def average_precision(scores, labels):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, scores))


def rank_metrics(scores, labels, ks=(10, 50, 100)):
    from sklearn.metrics import roc_auc_score
    order = np.argsort(-scores)
    prev = labels.mean()
    r = {"ap": average_precision(scores, labels), "auroc": float(roc_auc_score(labels, scores)),
         "prevalence": float(prev), "n_pos": int(labels.sum()), "n": int(len(labels))}
    for k in ks:
        h = int(labels[order[:k]].sum())
        r[f"hits@{k}"] = h
        r[f"enrichment@{k}"] = float(h / k / prev) if prev else None
    return r


def perm_p(scores, labels, n=10_000, bins=None, seed=0):
    rng = np.random.default_rng(seed)
    obs = average_precision(scores, labels)
    from sklearn.metrics import average_precision_score
    null = np.empty(n)
    for i in range(n):
        if bins is None:
            lab = rng.permutation(labels)
        else:
            lab = labels.copy()
            for b in np.unique(bins):
                w = np.flatnonzero(bins == b)
                lab[w] = labels[rng.permutation(w)]
        null[i] = average_precision_score(lab, scores)
    return {"ap": obs, "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95)),
            "p_one_sided": float((1 + np.sum(null >= obs)) / (n + 1))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--gene-split", choices=["gene", "complex"], default="gene")
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", required=True, help="encoding-mode checkpoint for --source")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--freeze-commit", default="")
    ap.add_argument("--n-targets", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.freeze_commit:
        sys.exit("test genes are evaluated once, after freezing: pass --freeze-commit")
    lat = load(args.backbone, gene_split=args.gene_split)
    m, z = lat.meta, lat.z
    model, bat, rec, d = from_checkpoint(REPO / "checkpoints" / "predictors" / f"{args.model}.pt",
                                         lat)
    enc = sets.encoding(args.source)
    mk = sets.masks(m, set(d["vocab"]) | {"NEG"})
    act = m.action.to_numpy()
    zt = z.mean(2)                                            # tile mean (N, 14, D)
    train = mk["train"]
    train_genes = sorted({g for g in act[train & mk["library"]] if g in enc})
    ep_train = {g: zt[train & (act == g), 12].mean(0) for g in train_genes}
    split = pd.read_csv(REPO / "splits" / ("gene_split_complex.tsv" if args.gene_split ==
                                           "complex" else "gene_split.tsv"), sep="\t")
    gsplit = dict(zip(split.gene, split.split, strict=False))
    res = {"split": args.split, "backbone": args.backbone, "source": args.source, "model": rec,
           "design_commit": DESIGN_COMMIT, "freeze_commit": args.freeze_commit or None}

    # ---------------- (a) held-out trajectory targets
    cands = sorted({g for g in act[mk["library"]] if g in enc})
    cand_ix = {g: i for i, g in enumerate(cands)}
    A = torch.tensor(np.stack([enc[g] for g in cands]), device=DEV)
    rng = np.random.default_rng(0)
    tgt = np.flatnonzero(mk[f"h2_{args.split}"] & m.action.isin(set(enc)).to_numpy())
    tgt = np.sort(rng.choice(tgt, min(args.n_targets, len(tgt)), replace=False))
    imp = impute_endpoints(cands, enc, train_genes, ep_train)
    obs_ep = {g: np.flatnonzero(mk["library"] & (act == g)) for g in cands}
    classes = m.drop_duplicates("action").set_index("action").classes.astype(str).to_dict()
    comp = pd.read_csv(REPO / "data" / "ref" / "gene_complexes.tsv", sep="\t")
    g2c = comp.groupby("gene").complex.apply(set).to_dict()
    methods = ["planner_dyn", "planner_end", "retrieval_knn", "retrieval_obs"]
    ranks = {k: [] for k in methods}
    top10 = {k: {"same_class": [], "same_complex": []} for k in methods}
    for i in tgt:
        g = act[i]
        traj = torch.tensor(zt[i, 1:13], device=DEV)
        pred = rollouts(model, torch.tensor(z[i, 0], device=DEV), A)
        sc = {"planner_dyn": -((pred - traj) ** 2).mean((1, 2)).cpu().numpy(),
              "planner_end": -((pred[:, 11] - traj[11]) ** 2).mean(1).cpu().numpy(),
              "retrieval_knn": -((imp - zt[i, 12]) ** 2).mean(1)}
        ob = np.full(len(cands), -np.inf)
        for c, w in obs_ep.items():
            w = w[w != i]
            if len(w):
                ob[cand_ix[c]] = -((zt[w, 12].mean(0) - zt[i, 12]) ** 2).mean()
        sc["retrieval_obs"] = ob
        for k, s in sc.items():
            order = np.argsort(-s)
            ranks[k].append(int(np.flatnonzero(order == cand_ix[g])[0]) + 1)
            top = [cands[j] for j in order[:10]]
            cl = set(x for x in classes.get(g, "[]").strip("[]").replace("'", "").split(", ")
                     if x)
            top10[k]["same_class"].append(
                np.mean([bool(cl & set(classes.get(t, "").strip("[]").replace("'", "")
                                       .split(", "))) for t in top]) if cl else np.nan)
            top10[k]["same_complex"].append(
                np.mean([bool(g2c.get(g, set()) & g2c.get(t, set())) for t in top])
                if g in g2c else np.nan)
    C = len(cands)
    traj_res = {"n_targets": int(len(tgt)), "n_target_genes": int(len(set(act[tgt]))),
                "n_candidates": C, "methods": {}}
    for k in methods:
        r = np.array(ranks[k])
        traj_res["methods"][k] = {
            "hit@1": float(np.mean(r <= 1)), "hit@10": float(np.mean(r <= 10)),
            "hit@100": float(np.mean(r <= 100)), "mrr": float(np.mean(1 / r)),
            "median_rank": float(np.median(r)),
            "top10_same_class": float(np.nanmean(top10[k]["same_class"])),
            "top10_same_complex": float(np.nanmean(top10[k]["same_complex"]))}
    traj_res["random"] = {"hit@1": 1 / C, "hit@10": 10 / C, "hit@100": 100 / C,
                          "mrr": float(np.mean(1 / np.arange(1, C + 1)))}
    # planner vs retrieval_knn, hit@100, bootstrap over target genes
    tg = act[tgt]
    ug = np.unique(tg)
    hp = np.array(ranks["planner_dyn"]) <= 100
    hr = np.array(ranks["retrieval_knn"]) <= 100
    per_g = np.array([[hp[tg == g].mean(), hr[tg == g].mean()] for g in ug])
    bs = per_g[np.random.default_rng(1).integers(0, len(ug), (10_000, len(ug)))]
    dd = bs[..., 0].mean(1) - bs[..., 1].mean(1)
    from scipy.stats import binomtest
    k100 = int(hp.sum())
    traj_res["planner_vs_retrieval_knn_hit@100"] = {
        "difference": float(per_g[:, 0].mean() - per_g[:, 1].mean()),
        "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    traj_res["planner_vs_random_hit@100_p"] = float(
        binomtest(k100, len(hp), 100 / C, alternative="greater").pvalue)
    res["trajectory_targets"] = traj_res
    print(json.dumps(traj_res["methods"], indent=1))

    # ---------------- (b) phenotype targets
    vh = json.loads((REPO / "manifests" / "validated_hits.json").read_text())
    for p in PHENOTYPES:
        f = REPO / "data" / "annotations" / vh["lists"][p]["file"]
        if hashlib.sha256(f.read_bytes()).hexdigest() != vh["lists"][p]["sha256"]:
            sys.exit(f"validated hit list {f.name} does not match its frozen SHA-256")
    hits = {p: set(pd.read_csv(REPO / "data" / "annotations" / vh["lists"][p]["file"],
                               sep="\t")["Gene Symbol"]) for p in PHENOTYPES}
    neg_plates = sorted(set(m.plate[train & mk["neg"]]))
    starts_pl = np.random.default_rng(2).choice(neg_plates, min(20, len(neg_plates)),
                                                replace=False)
    starts = [z[train & mk["neg"] & (m.plate == pl).to_numpy(), 0].mean(0) for pl in starts_pl]
    bins = pd.read_csv(REPO / "data" / "ref" / "hela_expression_bins.tsv", sep="\t",
                       index_col=0)["bin"].to_dict()
    sets_eval = {"unseen": sorted(g for g, s in gsplit.items() if s == args.split and g in enc),
                 "seen": sorted(g for g in train_genes)}
    ph_res = {"starts_plates": list(map(str, starts_pl)), "sets": {}}
    for set_name, genes in sets_eval.items():
        AA = torch.tensor(np.stack([enc[g] for g in genes]), device=DEV)
        preds = [rollouts(model, torch.tensor(s0, device=DEV), AA).cpu().numpy() for s0 in starts]
        imp_e = impute_endpoints(genes, enc, train_genes, ep_train)
        eb = np.array([bins.get(g, -1) for g in genes])
        out = {"n_genes": len(genes), "phenotypes": {}}
        pv_perm, pv_gba = [], []
        for p in PHENOTYPES:
            lab_w = np.array([p in str(c) for c in m.classes])
            wells = train & mk["library"] & lab_w
            r = {"n_target_wells": int(wells.sum())}
            labels = np.array([g in hits[p] for g in genes])
            if labels.sum() == 0 or wells.sum() == 0:
                out["phenotypes"][p] = {**r, "skipped": "no positives or no target wells"}
                pv_perm.append(None)
                continue
            th = [g for g in train_genes if g in hits[p]]
            gba_c = norm_rows(np.stack([enc[g] for g in th]).mean(0, keepdims=True))[0]
            gba = norm_rows(np.stack([enc[g] for g in genes])) @ gba_c
            for hz, step in (("primary_clip12", 11), ("secondary_clip6", 5)):
                target = zt[wells, step + 1].mean(0)
                score = -np.mean([((pr[:, step] - target) ** 2).mean(1) for pr in preds], 0)
                ret = -((imp_e - target) ** 2).mean(1) if step == 11 else None
                rr = {"planner": rank_metrics(score, labels),
                      "perm": perm_p(score, labels, args.n_perm),
                      "perm_expression_matched": perm_p(score, labels, args.n_perm, bins=eb)}
                if ret is not None:
                    rr["retrieval_knn"] = rank_metrics(ret, labels)
                    rr["gba"] = rank_metrics(gba, labels)
                    rr["gba_perm"] = perm_p(gba, labels, args.n_perm)
                    rr["n_train_hits_for_gba"] = len(th)
                r[hz] = rr
            out["phenotypes"][p] = r
            pv_perm.append(r["primary_clip12"]["perm"]["p_one_sided"])
            pv_gba.append(r["primary_clip12"]["planner"]["ap"] > r["primary_clip12"]["gba"]["ap"])
        q = bh(pv_perm)
        for p, qq in zip(PHENOTYPES, q, strict=True):
            if "primary_clip12" in out["phenotypes"][p]:
                out["phenotypes"][p]["primary_clip12"]["perm"]["q_bh"] = qq
        n_sig = sum(1 for qq in q if qq is not None and qq < 0.05)
        out["decision"] = {"n_phenotypes_q<0.05": n_sig, "n_beats_gba": int(sum(pv_gba)),
                           "pass": bool(n_sig >= 2 and sum(pv_gba) >= 1)}
        ph_res["sets"][set_name] = out
        print(set_name, json.dumps(out["decision"]),
              {p: round(v["primary_clip12"]["planner"]["ap"], 4)
               for p, v in out["phenotypes"].items() if "primary_clip12" in v})
    res["phenotype_targets"] = ph_res
    t = res["trajectory_targets"]
    res["G_Plan"] = {
        "rule": "trajectory: planner hit@100 above random (p < 0.05) and above retrieval_knn "
                "(bootstrap CI > 0); phenotype: amendment 01 rule on unseen genes",
        "trajectory_pass": bool(t["planner_vs_random_hit@100_p"] < 0.05 and
                                t["planner_vs_retrieval_knn_hit@100"]["ci95"][0] > 0),
        "phenotype_pass": ph_res["sets"]["unseen"]["decision"]["pass"]}
    res["code_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short",
                                         "HEAD"], capture_output=True, text=True).stdout.strip()
    out_dir = REPO / "results" / "h3"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"h3_{args.split}_{args.backbone}_{args.source}{args.tag}.json").write_text(
        json.dumps(res, indent=1, default=float))
    print(json.dumps(res["G_Plan"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
