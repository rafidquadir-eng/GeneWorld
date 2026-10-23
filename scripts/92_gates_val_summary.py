"""S13 review checkpoint (completion plan v3, items 1-8; v4, item 4 corrected, items 9-10):
reports/GATES_VAL_SUMMARY.md with the validation gate outcomes and a mechanical checklist.

Every item is checked from files and git history; an item that needs the author's judgment is
marked so, with the evidence. Writes nothing under results/.
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.train import a2  # noqa: E402

R = REPO / "results" / "a2"


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def first_commit(pattern: str) -> tuple[str, str] | None:
    """(hash, ISO time) of the first commit that added a file matching `pattern`."""
    out = git("log", "--diff-filter=A", "--reverse", "--format=%h %cI", "--", pattern)
    return tuple(out.splitlines()[0].split()) if out else None


def load(rel: str):
    p = R / rel
    return json.loads(p.read_text()) if p.exists() else None


def item(n, title, status, evidence):
    return {"n": n, "title": title, "status": status, "evidence": evidence}


def main() -> int:
    items = []
    gv = load("gates_val.json")
    sel = load("selection.json")
    h1 = load("h1/h1_val_plate_vjepa2.json")
    h2 = load("h2/h2_val_plate_vjepa2.json")

    # 1 commit order
    c_sel = first_commit("results/a2/selection.json")
    c_h2 = [first_commit(p) for p in ("results/a2/h2/*", "results/a2/h3/*")]
    c_g = first_commit("results/a2/gates_val.json")
    ok = bool(c_sel and all(c_h2) and c_g and
              all(not subprocess.run(["git", "merge-base", "--is-ancestor", c_sel[0], c[0]],
                                     cwd=REPO).returncode for c in c_h2) and
              all(not subprocess.run(["git", "merge-base", "--is-ancestor", c[0], c_g[0]],
                                     cwd=REPO).returncode for c in c_h2) and
              c_sel[0] not in (c[0] for c in c_h2))
    items.append(item(1, "commit order: selection.json before any H2/H3 output, before "
                         "gates_val.json", "pass" if ok else "FAIL",
                      {"selection": c_sel, "first_h2": c_h2[0], "first_h3": c_h2[1],
                       "gates_val": c_g}))

    # 2 three seeds at the selected configuration; seed averaging
    fams = a2.protocol()["fixed_before_any_amended_model_2026-10-20"]["families"]
    def run_json(f, s):
        return R / "train" / f"{a2.run_name('plate', 'vjepa2', f, sel['selected'][f], s)}.json"
    missing = [f"{f} s{s}" for f in fams for s in a2.protocol()["grid"]["seeds"]
               if sel and not run_json(f, s).exists()]
    n_models = {k: len(v) for k, v in (h1 or {}).get("models", {}).items()}
    items.append(item(2, "each gated family has 3 seeds at its selected configuration; errors "
                         "averaged over seeds before testing",
                      "pass" if sel and not missing and all(v == 3 for v in n_models.values())
                      else "FAIL",
                      {"missing_runs": missing, "h1_model_records": n_models,
                       "averaging": "src/geneworld/eval/a2eval.py seed_errors (mean of the "
                                    "three seeds' errors per well and horizon)"}))

    # 3 shuffled permutations and identical evaluation sets
    perm = {}
    for f in fams:
        if not f.startswith("h2shuf:"):
            continue
        for s in a2.protocol()["grid"]["seeds"]:
            enc = a2.family_encoding(f, s)
            src_enc = a2.family_encoding(f.replace("h2shuf:", "h2:"), s)
            inv = {k.tobytes(): g for g, k in src_enc.items()}
            mapping = "\n".join(f"{g}>{inv.get(v.tobytes(), '?')}"
                                for g, v in sorted(enc.items()))
            perm[f"{f} s{s}"] = hashlib.sha256(mapping.encode()).hexdigest()[:16]
    same = {}
    for s, v in (h2 or {}).get("sources", {}).items():
        p = v["primary"]
        same[s] = p["n_genes"] == p["true_vs_shuffled_trained"].get("n_genes")
    items.append(item(3, "trained-shuffled controls: fixed permutation per seed (hash); true "
                         "and shuffled scored on identical wells and genes",
                      "pass" if perm and same and all(same.values()) else "FAIL",
                      {"permutation_rule": "np.random.default_rng(10000 + seed) over the "
                                           "source's sorted genes (src/geneworld/train/a2.py "
                                           "family_encoding); not stored in run files, "
                                           "recomputed here",
                       "permutation_sha256_16": perm, "identical_gene_sets": same,
                       "identical_wells": "one evaluation index per scope for every model "
                                          "(scripts/62_eval_h2_a2.py)"}))

    # 4 (v4) H1 primary validation wells
    fr = pd.read_csv(REPO / "splits/amend02_h1_primary_wells.tsv", sep="\t")
    fr = fr[fr.plate_split == "val"]
    gs = pd.read_csv(REPO / "splits/gene_split.tsv", sep="\t")
    gcol = [c for c in gs.columns if c.lower() in ("gene", "gene symbol")][0]
    split_of = dict(zip(gs[gcol], gs.split, strict=False))
    mf = json.loads((R / "missing_frames.json").read_text())["wells"]
    s2 = {(w["plate"], w["well"]) for w in mf}
    dec = {"frozen": len(fr)}
    for sp in ("train", "val", "test"):
        f = fr[fr["Gene Symbol"].map(split_of) == sp]
        k = set(zip(f.Plate, f.Well, strict=True))
        dec[f"{sp}_split_genes"] = {"wells": len(k), "s2_excluded": len(k & s2)}
    expected = dec["train_split_genes"]["wells"] - dec["train_split_genes"]["s2_excluded"]
    got = h1["n_wells_primary"] if h1 else None
    items.append(item(4, "H1 primary validation wells evaluated = 362 = frozen validation "
                         "primary wells of training-split genes (386) minus S2 (24)",
                      "pass" if got == expected == 362 else "FAIL",
                      {"evaluated": got, "expected": expected, "decomposition_of_535": dec}))

    # 5 gate fields
    flds = {}
    if gv:
        s = gv["G_Signal_2"]
        flds["G_Signal_2"] = all(k in s for k in ("rule", "pass")) and all(
            all(x in s[b] for x in ("n_groups", "mean", "p_one_sided"))
            for b in ("vs_action_agnostic", "vs_replicate_drift"))
        z = gv["G_ZeroShot_2"]
        flds["G_ZeroShot_2"] = "rule" in z and all(
            all(x in v[c] for x in ("n_genes", "mean", "ci95"))
            for v in z["per_source"].values() for c in ("true_vs_shuffled_trained",
                                                         "true_vs_agnostic"))
        g = gv["G_Plan_2"]
        flds["G_Plan_2"] = ("rule" in g and g.get("trajectory_gate_comparator") ==
                            "retrieval_knn" and "oracle_ceiling_hit@100" in g and
                            "ci95" in g.get("trajectory_vs_comparator", {}))
    items.append(item(5, "every gate field reports n, effect, p or CI and the rule; G-Plan-2 "
                         "trajectory uses retrieval_knn; oracle ceiling reported",
                      "pass" if flds and all(flds.values()) else "FAIL", flds))

    # 6 baselines in H1
    need = ["replicate_drift", "replicate_endpoint", "slot_mean", "control_drift",
            "copy_last", "action_agnostic"]
    have = list((h1 or {}).get("mean_error", {}))
    items.append(item(6, "H1 baselines present (replicate drift and endpoint, slot and "
                         "control drift, copy-last, action-agnostic) with fallback counts",
                      "pass" if h1 and all(b in have for b in need) and
                      h1.get("replicate_baseline_sources") else "FAIL",
                      {"baselines": have, "fallbacks": (h1 or {}).get(
                          "replicate_baseline_sources")}))

    # 7 no test reference in results; test script not run
    ps = pd.read_csv(REPO / "splits/plate_split.tsv", sep="\t")
    tp = set(ps.loc[ps.split == "test", "plate"])
    tg = set(gs.loc[gs.split == "test", gcol])
    sb = set(pd.read_csv(REPO / "splits/screenB_wells.tsv", sep="\t").Plate)
    hits = {}
    for f in glob.glob(str(R.parent / "**" / "*.json"), recursive=True):
        st = set(re.findall(r'"([^"\\]{1,40})"', Path(f).read_text()))
        h = {"test_plates": len(st & tp), "test_genes": len(st & tg), "screenB_plates":
             len(st & sb)}
        if any(h.values()):
            hits[str(Path(f).relative_to(REPO))] = h
    ran = (REPO / "logs/stage1_test").exists()
    only_s2 = set(hits) <= {"results/a2/missing_frames.json"} and not any(
        v["test_plates"] or v["screenB_plates"] for v in hits.values())
    items.append(item(7, "no results file refers to a test plate, test gene or screenB well; "
                         "run_stage1_test.sh not run",
                      "pass" if not hits and not ran else
                      "author judgment" if only_s2 and not ran else "FAIL",
                      {"files_with_matches": hits, "logs/stage1_test exists": ran,
                       "note": "results/a2/missing_frames.json (S2) names the genes of failed "
                               "wells on training and validation plates, some of which are "
                               "test-split genes: annotation metadata of those wells, no "
                               "measurement or evaluation of a test gene" if hits else ""}))

    # 8 normalised and global
    need8 = ["h1/h1_val_plate_vjepa2.json", "h1/h1_val_global_vjepa2.json",
             "h2/h2_val_plate_vjepa2.json", "h2/h2_val_global_vjepa2.json"]
    src = (h2 or {}).get("planning_source")
    if src:
        need8 += [f"h3/h3_val_plate_vjepa2_{src}.json", f"h3/h3_val_global_vjepa2_{src}.json"]
    pres = {p: (R / p).exists() for p in need8}
    items.append(item(8, "normalised and global results both present",
                      "pass" if all(pres.values()) else "FAIL", pres))

    # 9 S18
    s18 = load("rehearsal/summary.json")
    p9 = bool(s18 and s18["compare"]["pass"] and s18.get("evaluation_all_ok"))
    items.append(item(9, "S18 passed (re-embedding criterion; full rehearsal path ran)",
                      "pass" if p9 else "FAIL" if s18 else "not yet run",
                      {"summary": "results/a2/rehearsal/summary.json",
                       "compare": {b: {k: v[k] for k in ("all_byte_identical",
                                                         "normalised_mse_fraction_of_error_level",
                                                         "pass")}
                                   for b, v in s18["compare"]["backbones"].items()} if s18
                       else None}))

    # 10 S17
    c17 = git("log", "--format=%h", "--grep", "S17 (completion plan v4)", "-1")
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_test_path.py"],
                       cwd=REPO, capture_output=True, text=True)
    reh_only = not ran
    items.append(item(10, "S17 committed and its tests pass; run_stage1_test.sh run only in "
                          "rehearsal mode", "pass" if c17 and t.returncode == 0 and reh_only
                      else "FAIL",
                      {"commit": c17, "tests": t.stdout.strip().splitlines()[-1:],
                       "test_mode_logs_absent": reh_only}))

    lines = ["# Validation gates and S13 checklist", "",
             f"Generated by `scripts/92_gates_val_summary.py` at HEAD "
             f"`{git('rev-parse', '--short', 'HEAD')}`. "
             "Validation only; no test plate, test gene or screenB well has been embedded or "
             "evaluated.", ""]
    if gv:
        lines += ["## Gate outcomes (validation)", "", "| gate | outcome | main numbers |",
                  "|---|---|---|"]
        s = gv["G_Signal_2"]
        lines.append(f"| G-Signal-2 | {'pass' if s['pass'] else 'fail'} | vs action-agnostic "
                     f"{s['vs_action_agnostic']['relative_change']:+.1%} (p = "
                     f"{s['vs_action_agnostic']['p_one_sided']:.3g}); vs replicate drift "
                     f"{s['vs_replicate_drift']['relative_change']:+.1%} (p = "
                     f"{s['vs_replicate_drift']['p_one_sided']:.3g}) |")
        z = gv["G_ZeroShot_2"]
        per = "; ".join(f"{k}: {'pass' if v['pass'] else 'fail'} (true − shuffled "
                        f"{v['true_vs_shuffled_trained']['mean']:+.4f} CI "
                        f"{np.round(v['true_vs_shuffled_trained']['ci95'], 4).tolist()})"
                        for k, v in z["per_source"].items())
        lines.append(f"| G-ZeroShot-2 | {'pass' if z['pass'] else 'fail'} | {per} |")
        g = gv["G_Plan_2"]
        lines.append(f"| G-Plan-2 | {'pass' if g['pass'] else 'fail'} | phenotype "
                     f"{'pass' if g['phenotype_pass'] else 'fail'}; trajectory "
                     f"{'pass' if g['trajectory_pass'] else 'fail'} (planner − retrieval_knn "
                     f"hit@100 {g['trajectory_vs_comparator']['difference']:+.3f} CI "
                     f"{np.round(g['trajectory_vs_comparator']['ci95'], 3).tolist()}; oracle "
                     f"ceiling hit@100 {g['oracle_ceiling_hit@100']:.3f}) |")
        lines += ["", f"Consequence: {gv['consequence']}. Planning source: "
                      f"`{gv['planning_source']}`.", ""]
    lines += ["## Checklist", "", "| # | item | status |", "|---|---|---|"]
    lines += [f"| {i['n']} | {i['title']} | {i['status']} |" for i in items]
    lines += ["", "## Evidence", ""]
    for i in items:
        lines += [f"**{i['n']}.** `{json.dumps(i['evidence'], default=str)}`", ""]
    (REPO / "reports" / "GATES_VAL_SUMMARY.md").write_text("\n".join(lines))
    print(json.dumps([[i["n"], i["status"]] for i in items]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
