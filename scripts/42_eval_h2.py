"""H2 evaluation: zero-shot forecasting for genes never seen in training (A.2, A.8, A.10).

For each encoding source with an encoding-mode checkpoint, on the unseen-gene wells of the split
(h2_val for selection and G-ZeroShot, h2_test once after freezing) that the source covers:

  true        the predictor fed the gene's own encoding
  shuffled    the same predictor fed the encoding of another evaluated gene (a derangement of
              genes; 5 derangements averaged) -- the H2 null
  knn_gene    no model: z_0 + mean drift (z_t - z_0) of the training wells of the k = 10
              training genes nearest in this encoding (cosine)
  agnostic    the action-agnostic predictor
Per gene, the forecast MSE (mean over 3/12/24/36 h and the gene's wells); the difference
true - shuffled is bootstrapped over genes (10,000 resamples). G-ZeroShot passes for a source
when the 95% CI of true - shuffled lies below 0. The comparison is repeated on the genes covered
by every source, so that sources are ranked on one gene set.
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

from geneworld import sets  # noqa: E402
from geneworld.data import load  # noqa: E402
from geneworld.eval import h1  # noqa: E402
from geneworld.train.batcher import from_checkpoint  # noqa: E402

K_NN = 10


def gene_means(err: np.ndarray, genes: np.ndarray) -> dict[str, float]:
    return {g: float(np.nanmean(err[genes == g])) for g in np.unique(genes)}


def boot_diff(a: dict, b: dict, n: int = 10_000, seed: int = 0) -> dict:
    gs = sorted(set(a) & set(b))
    d = np.array([a[g] - b[g] for g in gs])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"n_genes": len(gs), "mean": float(d.mean()),
            "relative": float(np.mean([a[g] for g in gs]) / np.mean([b[g] for g in gs]) - 1),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "frac_genes_better": float(np.mean(d < 0))}


def knn_forecast(z, meta, idx, train, enc, k=K_NN):
    act = meta.action.to_numpy()
    tr_genes = sorted({g for g in act[train] if g in enc})
    e_tr = np.stack([enc[g] for g in tr_genes])
    e_tr /= np.linalg.norm(e_tr, axis=1, keepdims=True) + 1e-9
    drift = {}
    for g in tr_genes:
        w = np.flatnonzero(train & (act == g))
        drift[g] = (z[w, 1:13] - z[w, :1]).mean(0)
    out = np.empty((len(idx), 12) + z.shape[2:], np.float32)
    for r, i in enumerate(idx):
        q = enc[act[i]] / (np.linalg.norm(enc[act[i]]) + 1e-9)
        nn_ = np.argsort(-(e_tr @ q))[:k]
        out[r] = z[i, :1] + np.mean([drift[tr_genes[j]] for j in nn_], 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--gene-split", choices=["gene", "complex"], default="gene")
    ap.add_argument("--models", required=True,
                    help="comma list source=checkpoint, e.g. string=h2_string,esm2=h2_esm2")
    ap.add_argument("--agnostic", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--freeze-commit", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.split == "test" and not args.freeze_commit:
        sys.exit("test genes are evaluated once, after freezing: pass --freeze-commit")
    ck = REPO / "checkpoints" / "predictors"
    lat = load(args.backbone, gene_split=args.gene_split)
    m, z = lat.meta, lat.z
    agn, bat_n, rec_n, _ = from_checkpoint(ck / f"{args.agnostic}.pt", lat)
    hcols = [h - 1 for h in h1.HORIZONS]
    res = {"split": args.split, "backbone": args.backbone, "agnostic": rec_n,
           "design_commit": "deb443d", "freeze_commit": args.freeze_commit or None,
           "sources": {}}
    per_source_gene_err = {}
    for item in args.models.split(","):
        src, name = item.split("=")
        model, bat, rec, d = from_checkpoint(ck / f"{name}.pt", lat)
        enc = sets.encoding(src)
        mk = sets.masks(m, set(d["vocab"]) | {"NEG"})
        sel = mk[f"h2_{args.split}"] & m.action.isin(set(enc)).to_numpy()
        idx = np.flatnonzero(sel)
        genes = m.action.to_numpy()[idx]
        e_true = h1.errors(h1.model_forecast(model, z, bat, idx), z, idx)[:, hcols].mean(1)
        # shuffled: derangements of the evaluated genes' encodings
        ug = np.unique(genes)
        e_sh = []
        orig = bat.action.clone()
        for s in range(5):
            rng = np.random.default_rng(100 + s)
            while True:
                perm = rng.permutation(len(ug))
                if not np.any(perm == np.arange(len(ug))):
                    break
            mp = dict(zip(ug, ug[perm], strict=True))
            for r, i in enumerate(idx):
                bat.action[i] = torch.tensor(enc[mp[genes[r]]], device=bat.action.device)
            e_sh.append(h1.errors(h1.model_forecast(model, z, bat, idx), z, idx)[:, hcols]
                        .mean(1))
            bat.action.copy_(orig)
        e_sh = np.mean(e_sh, 0)
        e_knn = h1.errors(knn_forecast(z, m, idx, mk["train"] & m.action.isin(set(enc))
                                       .to_numpy(), enc), z, idx)[:, hcols].mean(1)
        e_agn = h1.errors(h1.model_forecast(agn, z, bat_n, idx), z, idx)[:, hcols].mean(1)
        e_copy = h1.errors(np.repeat(z[idx, :1], 12, 1), z, idx)[:, hcols].mean(1)
        G = {k: gene_means(v, genes) for k, v in
             {"true": e_true, "shuffled": e_sh, "knn_gene": e_knn, "agnostic": e_agn,
              "copy_last": e_copy}.items()}
        per_source_gene_err[src] = G
        r = {"model": rec, "n_wells": int(len(idx)), "n_genes": int(len(ug)),
             "mean_error": {k: float(np.mean(list(v.values()))) for k, v in G.items()},
             "true_vs_shuffled": boot_diff(G["true"], G["shuffled"]),
             "true_vs_knn_gene": boot_diff(G["true"], G["knn_gene"]),
             "true_vs_agnostic": boot_diff(G["true"], G["agnostic"])}
        r["G_ZeroShot_pass"] = bool(r["true_vs_shuffled"]["ci95"][1] < 0)
        res["sources"][src] = r
        t = r["true_vs_shuffled"]
        print(f"{src}: {len(ug)} genes; true-shuffled {t['mean']:+.4f} ({t['relative']:+.1%}) "
              f"CI {t['ci95'][0]:+.4f}..{t['ci95'][1]:+.4f}; vs kNN "
              f"{r['true_vs_knn_gene']['relative']:+.1%}; vs agnostic "
              f"{r['true_vs_agnostic']['relative']:+.1%}")
    common = set.intersection(*[set(G["true"]) for G in per_source_gene_err.values()])
    res["common_genes"] = {
        "n_genes": len(common),
        "by_source": {s: {k: boot_diff({g: G["true"][g] for g in common},
                                       {g: G[k][g] for g in common})
                          for k in ("shuffled", "knn_gene", "agnostic")}
                      for s, G in per_source_gene_err.items()}}
    passing = [s for s, r in res["sources"].items() if r["G_ZeroShot_pass"]]
    best = min(passing, key=lambda s: res["sources"][s]["true_vs_shuffled"]["mean"]) \
        if passing else None
    res["G_ZeroShot"] = {"rule": "true - shuffled 95% CI over genes below 0 for >= 1 source",
                         "pass": bool(passing), "passing_sources": passing,
                         "selected_source": best}
    res["code_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short",
                                         "HEAD"], capture_output=True, text=True).stdout.strip()
    out = REPO / "results" / "h2"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"h2_{args.split}_{args.backbone}{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["G_ZeroShot"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
