"""Amendment 02 H2 evaluation and G-ZeroShot-2.

Per source: true-encoding models and trained shuffled-encoding models (selected configurations,
three seeds each), and the selected action-agnostic models (h1_none), on the wells of primary
genes of the evaluated gene split (val: on training or validation plates). Per gene, the mean
over its wells of the 24/36 h error, averaged over seeds. Bootstrap (10,000) 95% CI over genes.
G-ZeroShot-2 for a source: true - shuffled CI below 0 **and** true - agnostic CI below 0.
Secondary: all genes of the split; evaluation-time shuffle of the true model (stage-1 control,
derangement of evaluated genes, 5 draws); nearest-gene trajectory lookup; the source ranking on
genes every source covers. Writes the planning source (protocol rule) for H3.
"""
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import cache, sets  # noqa: E402
from geneworld.data import N_STEPS  # noqa: E402
from geneworld.eval import a2eval as E  # noqa: E402
from geneworld.train import a2  # noqa: E402

SOURCES = sets.SOURCES


def load3(family, cfg, norm, backbone, c, gs):
    ms, recs = [], []
    for s in a2.protocol()["grid"]["seeds"]:
        p = a2.CK_DIR / f"{a2.run_name(norm, backbone, family, cfg, s, gs)}.pt"
        m, b, r, _ = a2.load_model(p, c)
        ms.append((m, b))
        recs.append(r)
    return ms, recs


def knn(z, meta, idx, train, enc, k=10):
    act = meta.action.to_numpy()
    tg = sorted({g for g in act[train] if g in enc})
    e = np.stack([enc[g] for g in tg])
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
    tm = z[:, :N_STEPS].astype(np.float32)
    drift = {g: (tm[w, 1:] - tm[w, :1]).mean((0, 2))[:, None, :] for g in tg
             for w in [np.flatnonzero(train & (act == g))]}
    out = np.empty((len(idx), N_STEPS - 1) + z.shape[2:], np.float32)
    for r, i in enumerate(idx):
        q = enc[act[i]] / (np.linalg.norm(enc[act[i]]) + 1e-9)
        nn_ = np.argsort(-(e @ q))[:k]
        out[r] = tm[i, :1] + np.mean([drift[tg[j]] for j in nn_], 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test", "screenB"])
    ap.add_argument("--norm", default="plate", choices=["plate", "global"])
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--gene-split", default="gene", choices=["gene", "complex"])
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--suffix", default="")
    ap.add_argument("--freeze-commit", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split != "val" and not args.freeze_commit:
        sys.exit("held-out test sets are evaluated once, after freezing: pass --freeze-commit")
    sel = json.loads((a2.RES_DIR / "selection.json").read_text())["selected"]
    c = cache.load(args.backbone, args.norm, gene_split=args.gene_split, suffix=args.suffix)
    m, z = c.meta, c.z
    gs = args.gene_split
    AG, rec_ag = load3("h1_none", sel["h1_none"], args.norm, args.backbone, c, gs)
    res = {"split": args.split, "norm": args.norm, "backbone": args.backbone, "gene_split": gs,
           "design_commit": a2.DESIGN_COMMIT, "freeze_commit": args.freeze_commit or None,
           "agnostic_models": rec_ag, "sources": {}}
    per_src = {}
    for src in args.sources.split(","):
        enc = sets.encoding(src)
        T, rec_t = load3(f"h2:{src}", sel[f"h2:{src}"], args.norm, args.backbone, c, gs)
        S, rec_s = load3(f"h2shuf:{src}", sel[f"h2shuf:{src}"], args.norm, args.backbone, c, gs)
        if args.split == "screenB":
            res.setdefault("screenB_sirna_mask", {})[src] = E.mask_screenB_sirna([T, S, AG], m)
        out = {"models_true": rec_t, "models_shuffled": rec_s,
               "configs": {"true": sel[f"h2:{src}"], "shuffled": sel[f"h2shuf:{src}"]}}
        for scope, primary in (("primary", True), ("all_genes", False)):
            mask = a2.eval_mask(m, f"h2_{args.split}", set(), enc, primary=primary)
            idx = np.flatnonzero(mask)
            genes = m.action.to_numpy()[idx]
            e_t, _ = E.seed_errors(T, z, idx)
            e_s, _ = E.seed_errors(S, z, idx)
            e_a, _ = E.seed_errors(AG, z, idx)
            G = {"true": E.per_gene(E.primary(e_t), genes),
                 "shuffled_trained": E.per_gene(E.primary(e_s), genes),
                 "agnostic": E.per_gene(E.primary(e_a), genes)}
            if scope == "primary":
                # evaluation-time shuffle of the true models (stage-1 control)
                ug = np.unique(genes)
                origs = [bat.action.clone() for _, bat in T]
                ev = []
                for k in range(5):
                    rng = np.random.default_rng(100 + k)
                    while True:
                        perm = rng.permutation(len(ug))
                        if len(ug) < 2 or not np.any(perm == np.arange(len(ug))):
                            break
                    mp = dict(zip(ug, ug[perm], strict=True))
                    shuf = torch.tensor(np.stack([enc[mp[g]] for g in genes]))
                    for _, bat in T:
                        bat.action[torch.tensor(idx, device=bat.action.device)] = shuf.to(
                            bat.action.device)
                    ev.append(E.seed_errors(T, z, idx)[0])
                    for (_, bat), o in zip(T, origs, strict=True):
                        bat.action.copy_(o)
                G["shuffled_eval_time"] = E.per_gene(E.primary(np.mean(ev, 0)), genes)
                train = a2.train_mask(m, f"h2:{src}", 0)
                G["knn_gene"] = E.per_gene(E.primary(E.errors(knn(z, m, idx, train, enc), z,
                                                              idx)), genes)
                per_src[src] = G
            out[scope] = {"n_wells": int(len(idx)), "n_genes": int(len(np.unique(genes))),
                          "mean_error": {k: float(np.mean(list(v.values()))) for k, v in G.items()},
                          "true_vs_shuffled_trained": E.gene_boot(G["true"], G["shuffled_trained"]),
                          "true_vs_agnostic": E.gene_boot(G["true"], G["agnostic"])}
            if scope == "primary":
                out[scope]["true_vs_shuffled_eval_time"] = E.gene_boot(
                    G["true"], G["shuffled_eval_time"])
                out[scope]["true_vs_knn_gene"] = E.gene_boot(G["true"], G["knn_gene"])
        p = out["primary"]
        ci_s, ci_a = p["true_vs_shuffled_trained"]["ci95"], p["true_vs_agnostic"]["ci95"]
        out["G_ZeroShot_2_pass"] = bool(ci_s is not None and ci_a is not None and
                                        ci_s[1] < 0 and ci_a[1] < 0)
        res["sources"][src] = out
        print(f"{src}: {p['n_genes']} primary genes; true-shuffled "
              f"{p['true_vs_shuffled_trained']['mean']:+.4f} CI {ci_s}; true-agnostic "
              f"{p['true_vs_agnostic']['mean']:+.4f} CI {ci_a}; pass {out['G_ZeroShot_2_pass']}")
    common = set.intersection(*[set(G["true"]) for G in per_src.values()])
    res["common_primary_genes"] = {
        "n_genes": len(common),
        "by_source": {s: {k: E.gene_boot({g: G["true"][g] for g in common},
                                         {g: G[k][g] for g in common})
                          for k in ("shuffled_trained", "agnostic")}
                      for s, G in per_src.items()}}
    passing = [s for s, r in res["sources"].items() if r["G_ZeroShot_2_pass"]]
    key = lambda s: res["sources"][s]["primary"]["true_vs_shuffled_trained"]["mean"]  # noqa: E731
    res["G_ZeroShot_2"] = {
        "rule": "on primary genes at 24/36 h, >= 1 source: true - trained-shuffled 95% CI over "
                "genes below 0 and true - action-agnostic 95% CI below 0",
        "pass": bool(passing), "passing_sources": passing}
    res["planning_source"] = min(passing, key=key) if passing else min(res["sources"], key=key)
    res["planning_source_rule"] = a2.protocol()["fixed_before_any_amended_model_2026-10-20"][
        "planning_source"]
    res["code_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                        capture_output=True, text=True).stdout.strip()
    out_dir = a2.RES_DIR / "h2"
    out_dir.mkdir(parents=True, exist_ok=True)
    gtag = "" if gs == "gene" else "_complex"
    (out_dir / f"h2_{args.split}_{args.norm}_{args.backbone}{gtag}{args.tag}.json").write_text(
        json.dumps(res, indent=1, default=float))
    print(json.dumps(res["G_ZeroShot_2"]), "planning source:", res["planning_source"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
