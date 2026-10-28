"""Stage-2 robustness controls R1-R4 (docs/STAGE2_ROBUSTNESS_PLAN.md, committed 78c8775 before
any of them ran): post hoc, exploratory, after the test. Nothing here changes a pre-registered
result, gate, threshold or frozen file; outputs go to results/a2/robustness/.

  r1-train --family h1_learned|h1_none --seed S
      h1_learned: trained-shuffled control at the selected h1_learned configuration
      (50M_d0.1_w0.2). One fixed permutation pi over the training siRNAs (siRNAs of training
      wells with a gene action, i.e. library and positive-control reagents; negative controls
      unchanged): default_rng(20261027 + seed).permutation over the sorted list. A well with
      siRNA s receives siRNA token pi(s) and the gene action gene(pi(s)), identically in
      training and evaluation. A well with a gene action whose siRNA is not a training siRNA
      (no primary well; non-primary evaluation wells only) keeps an unknown token and takes
      gene(pi(s*)), s* = the first training siRNA of its gene in sorted order.
      h1_none: action-agnostic at 50M_d0.1_w0.2, the config-matched agnostic reference.
      Checkpoints in checkpoints/a2_robust/r1_<family>/, run records in
      results/a2/robustness/r1/train/.
  r1-eval --split val|test     I1: real h1_learned vs the trained-shuffled models.
  r234 --split val|test        R2 additive memorisation baseline (lambda chosen on validation),
                               R3 reagent dependence (a, b, c), R4 where the gain sits.

Validation uses the cache the validation gates used (data/cache/vjepa2); test uses the test
cache (data/cache/vjepa2_test) and records the freeze commit ce5c3df. Statistics are those of
G-Signal-2: primary wells, mean error at 24 h and 36 h, errors averaged over the 3 seeds,
per-plate means, one-sided Monte Carlo sign flip over plates (a2eval.plate_test). Before any
control number is written, the real model's comparison with the selected agnostic models is
recomputed and checked against the recorded results/a2/h1 file.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import cache  # noqa: E402
from geneworld.data import N_STEPS  # noqa: E402
from geneworld.eval import a2eval as E  # noqa: E402
from geneworld.train import a2  # noqa: E402

LABEL = "exploratory, after the test (post hoc robustness control, docs/STAGE2_ROBUSTNESS_PLAN.md)"
PLAN_COMMIT = "78c8775"
FREEZE = "ce5c3df"
R1_CONFIG = "50M_d0.1_w0.2"
PERM_BASE = 20261027
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT = REPO / "results" / "a2" / "robustness"
CK = REPO / "checkpoints" / "a2_robust"
SUFFIX = {"val": "", "test": "_test"}
CLASSES = None   # set from 61_eval_h1_a2.CLASSES


def script(name: str):
    spec = importlib.util.spec_from_file_location(name[:-3].replace("-", "_"),
                                                  REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def code_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def write(rel: str, obj: dict) -> None:
    f = OUT / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"label": LABEL, "plan_commit": PLAN_COMMIT,
                             "code_commit": code_commit(), **obj}, indent=1, default=float))
    print(f"written {f.relative_to(REPO)} (numbers not printed)")


# ---------------------------------------------------------------- R1 permutation
def permutation(meta, seed: int) -> dict[str, str]:
    """siRNA -> siRNA (pi) over the training siRNAs with a gene action."""
    tr = meta.train.to_numpy() & (meta.action != "NEG").to_numpy() & (meta.sirna != "").to_numpy()
    s = sorted(set(meta.sirna[tr]))
    perm = np.random.default_rng(PERM_BASE + seed).permutation(len(s))
    return {a: s[j] for a, j in zip(s, perm, strict=True)}


def remap(meta, seed: int):
    """(meta with permuted action and sirna, record). Plate, well, split and train columns are
    unchanged, so training and evaluation masks are identical to the real model's."""
    pi = permutation(meta, seed)
    tr = meta.train.to_numpy()
    g = meta[tr & meta.sirna.isin(set(pi)).to_numpy()].groupby("sirna").action
    if (g.nunique() > 1).any():
        raise SystemExit("an siRNA maps to more than one gene on training plates")
    gene_of = g.first().to_dict()
    first = {}
    for s in sorted(pi):
        first.setdefault(gene_of[s], s)
    m = meta.copy()
    act, sir = m.action.to_numpy().copy(), m.sirna.to_numpy().copy()
    n_mapped = n_gene_only = 0
    for i, (a, s) in enumerate(zip(act, sir, strict=True)):
        if a == "NEG":
            continue
        if s in pi:
            sir[i], act[i] = pi[s], gene_of[pi[s]]
            n_mapped += 1
        elif a in first:
            sir[i], act[i] = "", gene_of[pi[first[a]]]
            n_gene_only += 1
    m["action"], m["sirna"] = act, sir
    h = hashlib.sha256("\n".join(f"{a}\t{b}" for a, b in sorted(pi.items())).encode())
    return m, {"rule": f"default_rng({PERM_BASE} + seed).permutation over the sorted training "
                       "siRNAs with a gene action; well with siRNA s -> token pi(s), gene "
                       "gene(pi(s))",
               "seed": seed, "n_sirnas": len(pi), "sha256": h.hexdigest(),
               "n_fixed_points": int(sum(a == b for a, b in pi.items())),
               "n_wells_remapped": n_mapped,
               "n_wells_gene_only (siRNA not a training siRNA; no primary well)": n_gene_only}


def r1_train(family: str, seed: int) -> int:
    tr = script("50_train_a2.py")
    a2.CK_DIR = CK / f"r1_{family}"
    a2.RES_DIR = OUT / "r1"
    perm_rec = None
    if family == "h1_learned":
        orig = cache.load

        def load(*a, **k):
            nonlocal perm_rec
            c = orig(*a, **k)
            meta, perm_rec = remap(c.meta, seed)
            return dataclasses.replace(c, meta=meta)
        cache.load = load
    sys.argv = ["50_train_a2.py", "--family", family, "--config", R1_CONFIG, "--seed",
                str(seed), "--norm", "plate", "--backbone", "vjepa2"]
    rc = tr.main()
    name = a2.run_name("plate", "vjepa2", family, R1_CONFIG, seed)
    f = a2.RES_DIR / "train" / f"{name}.json"
    d = json.loads(f.read_text())
    d = {"label": LABEL, "plan_commit": PLAN_COMMIT,
         "role": "trained-shuffled h1_learned (R1)" if family == "h1_learned"
         else "config-matched action-agnostic reference (R1)",
         "permutation": perm_rec, **d}
    f.write_text(json.dumps(d, indent=1, default=float))
    return rc


# ---------------------------------------------------------------- shared evaluation
def models(family, cfg, c, ck_dir=None, norm="plate"):
    """Three seeds' (model, batcher); `c` may carry a remapped meta (R1)."""
    out, recs = [], []
    for s in a2.protocol()["grid"]["seeds"]:
        d = ck_dir or a2.CK_DIR
        cc = c(s) if callable(c) else c
        m, b, r, _ = a2.load_model(d / f"{a2.run_name(norm, 'vjepa2', family, cfg, s)}.pt", cc)
        out.append((m, b))
        recs.append(r)
    return out, recs


def setup(split: str, norm: str = "plate"):
    c = cache.load("vjepa2", norm, suffix=SUFFIX[split])
    sel = json.loads((REPO / "results/a2/selection.json").read_text())["selected"]
    AC, rac = models("h1_learned", sel["h1_learned"], c, norm=norm)
    AG, rag = models("h1_none", sel["h1_none"], c, norm=norm)
    import torch
    vocab = set(torch.load(
        a2.CK_DIR / f"{a2.run_name(norm, 'vjepa2', 'h1_learned', sel['h1_learned'], 0)}.pt",
        map_location="cpu", weights_only=False)["vocab"]) | {"NEG"}
    m = c.meta
    which = f"h1_{split}"
    prim = a2.eval_mask(m, which, vocab, primary=True)
    allw = a2.eval_mask(m, which, vocab, primary=False)
    idx = np.flatnonzero(allw)
    is_p = prim[idx]
    plates = m.plate.to_numpy()[idx]
    return dict(c=c, sel=sel, AC=AC, AG=AG, rec={"action_conditioned": rac,
                                                  "action_agnostic": rag},
                idx=idx, is_p=is_p, plates=plates, split=split, norm=norm)


def ptest(ea, eb, S, mask=None, steps=a2.PRIMARY_STEPS):
    """G-Signal-2 statistic (model a vs model b) on primary wells, or on `mask` over idx."""
    k = S["is_p"] if mask is None else mask
    if k.sum() == 0 or len(np.unique(S["plates"][k])) < 2:
        return {"n_wells": int(k.sum()), "skipped": "fewer than 2 plates"}
    t = E.plate_test(E.primary(ea[k], steps), E.primary(eb[k], steps), S["plates"][k])
    t["n_wells"] = int(k.sum())
    return t


def check_against_record(S, e_ac, e_ag) -> dict:
    """The recomputed real comparison must equal the recorded results/a2/h1 file."""
    f = REPO / "results/a2/h1" / f"h1_{S['split']}_{S['norm']}_vjepa2.json"
    rec = json.loads(f.read_text())["G_Signal_2"]["vs_action_agnostic"]
    now = ptest(e_ac, e_ag, S)
    ok = (now["n_groups"] == rec["n_groups"] and
          abs(now["relative_change"] - rec["relative_change"]) < 1e-6 and
          abs(now["mean"] - rec["mean"]) < 1e-6)
    if not ok:
        raise SystemExit(f"recomputed real-vs-agnostic differs from {f.name}: {now} vs {rec}")
    return {"file": str(f.relative_to(REPO)), "relative_change": now["relative_change"],
            "mean": now["mean"], "n_groups": now["n_groups"], "matches": True}


def base_record(S) -> dict:
    return {"split": S["split"], "norm": S["norm"],
            "freeze_commit": FREEZE if S["split"] == "test" else None,
            "selected_configs": {k: S["sel"][k] for k in ("h1_learned", "h1_none")},
            "n_wells_primary": int(S["is_p"].sum()),
            "n_plates_primary": int(len(np.unique(S["plates"][S["is_p"]]))),
            "n_wells_all": int(len(S["idx"])), "statistic": (
                "primary wells, mean error at 24 h and 36 h, errors averaged over 3 seeds, "
                "per-plate means, one-sided Monte Carlo sign flip over plates (100,000 draws); "
                "mean < 0 means the first-named model has lower error")}


# ---------------------------------------------------------------- R1 evaluation
def r1_eval(split: str) -> None:
    S = setup(split)
    c, idx, z = S["c"], S["idx"], S["c"].z
    e_ac, _ = E.seed_errors(S["AC"], z, idx)
    e_ag, _ = E.seed_errors(S["AG"], z, idx)
    chk = check_against_record(S, e_ac, e_ag)
    perms = {}

    def shuffled_cache(seed):
        meta, perms[seed] = remap(c.meta, seed)
        return dataclasses.replace(c, meta=meta)
    SH, rsh = models("h1_learned", R1_CONFIG, shuffled_cache, CK / "r1_h1_learned")
    NO, rno = models("h1_none", R1_CONFIG, c, CK / "r1_h1_none")
    e_sh, _ = E.seed_errors(SH, z, idx)
    e_no, _ = E.seed_errors(NO, z, idx)
    tests = {
        "real_vs_trained_shuffled": ptest(e_ac, e_sh, S),
        "trained_shuffled_vs_selected_agnostic": ptest(e_sh, e_ag, S),
        "trained_shuffled_vs_config_matched_agnostic": ptest(e_sh, e_no, S),
        "real_vs_config_matched_agnostic": ptest(e_ac, e_no, S),
        "config_matched_agnostic_vs_selected_agnostic": ptest(e_no, e_ag, S),
        "real_vs_selected_agnostic (recorded gate comparison)": ptest(e_ac, e_ag, S)}
    tests["secondary_all_wells_real_vs_trained_shuffled"] = ptest(
        e_ac, e_sh, S, mask=np.ones(len(idx), bool))
    gain_real = tests["real_vs_selected_agnostic (recorded gate comparison)"]["mean"]
    gain_sh = tests["trained_shuffled_vs_selected_agnostic"]["mean"]
    t = tests["real_vs_trained_shuffled"]
    write(f"r1/r1_{split}.json", {
        **base_record(S), "check_real_vs_agnostic_against_record": chk,
        "trained_shuffled": {"config": R1_CONFIG, "models": rsh, "permutations": perms},
        "config_matched_agnostic": {"config": R1_CONFIG, "models": rno},
        "tests": tests,
        "fraction_of_gain_over_selected_agnostic_reached_by_trained_shuffled":
            gain_sh / gain_real if gain_real else None,
        "I1": {"rule": "real h1_learned beats the R1 trained-shuffled model, p < 0.05 over "
                       "plates", "holds": bool(t["p_one_sided"] < 0.05 and t["mean"] < 0)}})


# ---------------------------------------------------------------- R2-R4
def mean_forecast(models_, z, idx, bs=4096) -> np.ndarray:
    """Seed-mean rollout from clip 0, (n, 12, K, D) float32."""
    out = None
    for mdl, b in models_:
        p = np.concatenate([E.forecast(mdl, z, b, idx[i:i + bs]).astype(np.float32)
                            for i in range(0, len(idx), bs)])
        out = p if out is None else out + p
    return out / len(models_)


def r2_correction(S, ag_eval) -> tuple[np.ndarray, dict]:
    """Per evaluated well: tile-mean residual obs - agnostic (seed-mean), averaged over the
    training-plate wells of its siRNA (else the gene's other siRNAs), (n, 12, 1, D)."""
    m, z, idx = S["c"].meta, S["c"].z, S["idx"]
    train = a2.train_mask(m, "h1_learned", 0) & (m.action != "NEG").to_numpy()
    sir, act = m.sirna.to_numpy(), m.action.to_numpy()
    need_s, need_g = set(sir[idx]), set(act[idx])
    tr = np.flatnonzero(train & (np.isin(sir, list(need_s)) | np.isin(act, list(need_g))))
    print(f"R2: {len(tr)} training-plate wells for the residuals")
    ag_tr = mean_forecast(S["AG"], z, tr)
    res = (z[tr, 1:N_STEPS].astype(np.float32) - ag_tr).mean(2, keepdims=True)
    del ag_tr
    by_s, by_g = {}, {}
    for r, i in enumerate(tr):
        if sir[i]:
            by_s.setdefault(sir[i], []).append(r)
        by_g.setdefault(act[i], []).append(r)
    corr = np.zeros((len(idx), N_STEPS - 1, 1, z.shape[-1]), np.float32)
    used = {"sirna": 0, "gene_other_sirnas": 0, "none": 0}
    for r, i in enumerate(idx):
        w = by_s.get(sir[i], [])
        kind = "sirna"
        if not w:
            w = [k for k in by_g.get(act[i], []) if sir[tr[k]] != sir[i]]
            kind = "gene_other_sirnas"
        if w:
            corr[r] = res[w].mean(0)
        else:
            kind = "none"
        used[kind] += 1
    used["primary_sirna"] = int(sum(bool(by_s.get(sir[i])) for i in idx[S["is_p"]]))
    return corr, used


def r234(split: str) -> None:
    global CLASSES
    CLASSES = script("61_eval_h1_a2.py").CLASSES
    S = setup(split)
    c, m, z, idx, is_p, plates = S["c"], S["c"].meta, S["c"].z, S["idx"], S["is_p"], S["plates"]
    e_ac, P_ac = E.seed_errors(S["AC"], z, idx)
    e_ag, P_ag = E.seed_errors(S["AG"], z, idx)
    chk = check_against_record(S, e_ac, e_ag)
    base = base_record(S)

    # R2
    ag_mean = np.mean([p.astype(np.float32) for p in P_ag], 0)
    corr, used = r2_correction(S, ag_mean)
    e_r2 = {lam: E.errors(ag_mean + lam * corr, z, idx) for lam in LAMBDAS}
    stat = {lam: float(np.mean([np.nanmean(E.primary(e_r2[lam][is_p & (plates == p)]))
                                for p in np.unique(plates[is_p])])) for lam in LAMBDAS}
    if split == "val":
        lam_val = min(LAMBDAS, key=lambda x: (stat[x], x))
    else:
        lam_val = json.loads((OUT / "r2/r2_val.json").read_text())["lambda_validation"]
    rep, _ = E.replicate_baselines(z, m, idx, a2.train_mask(m, "h1_learned", 0))
    e_drift = E.errors(rep["replicate_drift"], z, idx)
    tests = {f"real_vs_R2_lambda_{lam}": ptest(e_ac, e_r2[lam], S) for lam in LAMBDAS}
    tests["R2_lambda_validation_vs_selected_agnostic"] = ptest(e_r2[lam_val], e_ag, S)
    tests["R2_lambda_1_vs_selected_agnostic"] = ptest(e_r2[1.0], e_ag, S)
    tests["R2_lambda_validation_vs_replicate_drift"] = ptest(e_r2[lam_val], e_drift, S)
    tr_lib = np.flatnonzero(a2.train_mask(m, "h1_learned", 0) & (m.control == "").to_numpy())
    feat = lambda x: x.astype(np.float32).mean(1)  # noqa: E731  tile mean
    cls = m.classes.to_numpy()
    r2v = ag_mean + lam_val * corr
    probes = {}
    for k in CLASSES:
        probes[k] = E.probe_with_ci(
            feat(z[tr_lib, 12]), np.array([k in x for x in cls[tr_lib]]),
            {"observed": [feat(z[idx, 12])],
             "action_conditioned": [feat(p[:, 11]) for p in P_ac],
             "action_agnostic": [feat(p[:, 11]) for p in P_ag],
             "R2_lambda_validation": [feat(r2v[:, 11])]},
            np.array([k in x for x in cls[idx]]), plates,
            diffs=(("action_conditioned", "R2_lambda_validation"),
                   ("R2_lambda_validation", "action_agnostic")))
    del r2v
    t = tests[f"real_vs_R2_lambda_{lam_val}"]
    write(f"r2/r2_{split}.json", {
        **base, "check_real_vs_agnostic_against_record": chk,
        "definition": "agnostic(w, h) + lambda * mean over training-plate wells w' of the "
                      "well's siRNA (else the gene's other siRNAs) of [obs(w', h) - "
                      "agnostic(w', h)]; agnostic = seed-mean forecast of the selected h1_none "
                      "models; residuals averaged over tiles (tiles are exchangeable fields of "
                      "view, as for the replicate baselines); R2 errors are those of the "
                      "seed-mean forecast",
        "caveat": "the residuals on training-plate wells are in-sample for the agnostic model",
        "lambda_grid": LAMBDAS,
        "validation_primary_error_by_lambda" if split == "val"
        else "primary_error_by_lambda": stat,
        "lambda_validation": lam_val,
        "lambda_choice": "lowest validation primary error (per-plate mean); applied unchanged "
                         "to test",
        "residual_sources": used, "tests": tests, "phenotype_probe_36h": probes,
        "I2_component": {"rule": "real h1_learned beats R2 at the validation-chosen lambda, "
                                 "p < 0.05 over plates (I2 also requires I1)",
                         "holds": bool(t["p_one_sided"] < 0.05 and t["mean"] < 0)}})
    del corr, e_r2

    # R3 (a) token zeroed, (b) labels permuted within plate, (c) global with token zeroed
    def variant(models_, edit):
        import copy
        out = []
        for mdl, b in models_:
            b2 = copy.copy(b)          # edits clone the tensors they change
            edit(b2)
            out.append((mdl, b2))
        return out

    def zero(b):
        b.sirna = b.sirna.clone()
        b.sirna[:] = 0
    rng = np.random.default_rng(PERM_BASE)
    perm_idx = idx.copy()
    for p in np.unique(plates):
        k = np.flatnonzero(plates == p)
        perm_idx[k] = idx[k][rng.permutation(len(k))]

    def permute(b):
        import torch
        b.action, b.sirna = b.action.clone(), b.sirna.clone()
        src = torch.tensor(perm_idx, device=b.action.device)
        dst = torch.tensor(idx, device=b.action.device)
        b.action[dst], b.sirna[dst] = b.action[src].clone(), b.sirna[src].clone()
    e_zero, _ = E.seed_errors(variant(S["AC"], zero), z, idx)
    e_perm, _ = E.seed_errors(variant(S["AC"], permute), z, idx)
    full = ptest(e_ac, e_ag, S)
    za = ptest(e_zero, e_ag, S)
    r3 = {"a_token_zeroed": {"vs_selected_agnostic": za,
                             "real_vs_token_zeroed": ptest(e_ac, e_zero, S),
                             "fraction_of_gain_retained": za["mean"] / full["mean"]},
          "b_labels_permuted_within_plate": {
              "rule": f"gene and siRNA labels permuted across the evaluated library wells of "
                      f"each plate (default_rng({PERM_BASE})), at evaluation time",
              "vs_selected_agnostic": ptest(e_perm, e_ag, S),
              "real_vs_permuted": ptest(e_ac, e_perm, S)},
          "full_gain_vs_selected_agnostic": full}
    del e_zero, e_perm, P_ac, P_ag
    G = setup(split, norm="global")
    ge_ac, _ = E.seed_errors(G["AC"], G["c"].z, G["idx"])
    ge_ag, _ = E.seed_errors(G["AG"], G["c"].z, G["idx"])
    gchk = check_against_record(G, ge_ac, ge_ag)
    ge_zero, _ = E.seed_errors(variant(G["AC"], zero), G["c"].z, G["idx"])
    gfull = ptest(ge_ac, ge_ag, G)
    gz = ptest(ge_zero, ge_ag, G)
    r3["c_global_token_zeroed"] = {"check_real_vs_agnostic_against_record": gchk,
                                   "vs_selected_agnostic": gz,
                                   "full_gain_vs_selected_agnostic": gfull,
                                   "fraction_of_gain_retained": gz["mean"] / gfull["mean"],
                                   "n_wells_primary": int(G["is_p"].sum())}
    del G, ge_ac, ge_ag, ge_zero
    a = r3["a_token_zeroed"]
    write(f"r3/r3_{split}.json", {
        **base, "check_real_vs_agnostic_against_record": chk, **r3,
        "I3_component": {"rule": "under R3(a) the gain over agnostic keeps >= 50% of its size "
                                 "and stays p < 0.05 (I3 also requires I1)",
                         "holds": bool(a["fraction_of_gain_retained"] >= 0.5 and
                                       a["vs_selected_agnostic"]["p_one_sided"] < 0.05 and
                                       a["vs_selected_agnostic"]["mean"] < 0)}})

    # R4
    train = a2.train_mask(m, "h1_learned", 0)
    sir = m.sirna.to_numpy()
    cls_of = {}
    for i in np.flatnonzero(train):
        if sir[i]:
            cls_of.setdefault(sir[i], set()).update(cls[i])
    wcls = [cls_of.get(sir[i], set()) for i in idx]
    per_class = {}
    for k in CLASSES + ["none"]:
        mk = np.array([(k in s) if k != "none" else not s for s in wcls]) & is_p
        per_class[k] = ptest(e_ac, e_ag, S, mask=mk)
    write(f"r4/r4_{split}.json", {
        **base, "description": "gain of the real h1_learned over the selected agnostic models "
                               "(descriptive)",
        "per_horizon_primary": {f"{h * 3}h": ptest(e_ac, e_ag, S, steps=[h])
                                for h in (1, 4, 8, 12)},
        "per_training_plate_class_primary": per_class,
        "class_definition": "classes of the training-plate wells of the well's siRNA "
                            "(reproducible-phenotype annotation); a well can be in several",
        "non_primary_library_wells": ptest(e_ac, e_ag, S, mask=~is_p),
        "primary_wells": ptest(e_ac, e_ag, S)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["r1-train", "r1-eval", "r234"])
    ap.add_argument("--family", choices=["h1_learned", "h1_none"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--split", choices=["val", "test"])
    args = ap.parse_args()
    if args.cmd == "r1-train":
        return r1_train(args.family, args.seed)
    if args.cmd == "r1-eval":
        r1_eval(args.split)
    else:
        r234(args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
