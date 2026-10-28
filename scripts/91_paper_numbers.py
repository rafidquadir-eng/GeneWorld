"""Numbers for paper/geneworld/: every number in the manuscript is a LaTeX macro written here
from a results file, a frozen manifest or the protocol, never typed by hand (rule 1.1.1).

Writes paper/geneworld/numbers.tex. Results of the gates and of the test are added to this
script only after they exist; until then the manuscript's results sections are placeholders.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "geneworld" / "numbers.tex"


def j(rel: str) -> dict:
    return json.loads((REPO / rel).read_text())


def n(x: int) -> str:
    return f"{x:,}".replace(",", "{,}")


def pv(x: float) -> str:
    """p value for math mode."""
    if x < 0.001:
        a, e = f"{x:.1e}".split("e")
        return f"{a}\\times10^{{{int(e)}}}"
    return f"{x:.3f}"


def rel(x: float) -> str:
    """|relative change| in percent, one decimal (direction is stated in the text)."""
    return f"{abs(100 * x):.1f}"


def fx(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def word(b: bool) -> str:
    return "holds" if b else "does not hold"


def results_macros() -> dict[str, str]:
    """Gates (pre-registered), sensitivity analyses, screenB, exploratory E1-E3 and the post hoc
    robustness controls; each block is added only if its results files exist."""
    m: dict[str, str] = {}
    A = REPO / "results" / "a2"
    for sp, S in (("val", "Val"), ("test", "Test")):
        f = A / f"gates_{sp}.json"
        if not f.exists():
            continue
        g = json.loads(f.read_text())
        s = g["G_Signal_2"]
        m |= {f"Signal{S}Pass": "passes" if s["pass"] else "fails",
              f"Signal{S}Plates": n(s["vs_action_agnostic"]["n_groups"]),
              f"Signal{S}Ag": rel(s["vs_action_agnostic"]["relative_change"]),
              f"Signal{S}AgP": pv(s["vs_action_agnostic"]["p_one_sided"]),
              f"Signal{S}Drift": rel(s["vs_replicate_drift"]["relative_change"]),
              f"Signal{S}DriftP": pv(s["vs_replicate_drift"]["p_one_sided"])}
        z = g["G_ZeroShot_2"]
        m |= {f"ZeroShot{S}Pass": "passes" if z["pass"] else "fails",
              f"ZeroShot{S}NPassing": n(len(z["passing_sources"]))}
        st = z["per_source"]["string"]
        for k, kk in (("true_vs_shuffled_trained", "Shuf"), ("true_vs_agnostic", "Ag")):
            t = st[k]
            m |= {f"ZeroShot{S}String{kk}": fx(t["mean"], 4),
                  f"ZeroShot{S}String{kk}Lo": fx(t["ci95"][0], 4),
                  f"ZeroShot{S}String{kk}Hi": fx(t["ci95"][1], 4)}
        m[f"ZeroShot{S}StringGenes"] = n(st["true_vs_shuffled_trained"]["n_genes"])
        p = g["G_Plan_2"]
        t = p["trajectory_vs_comparator"]
        m |= {f"Plan{S}Pass": "passes" if p["pass"] else "fails",
              f"Plan{S}Diff": fx(t["difference"]), f"Plan{S}Lo": fx(t["ci95"][0]),
              f"Plan{S}Hi": fx(t["ci95"][1]), f"Plan{S}Oracle": fx(p["oracle_ceiling_hit@100"])}
        h1 = json.loads((A / "h1" / f"h1_{sp}_plate_vjepa2.json").read_text())
        m[f"HOne{S}Wells"] = n(h1["n_wells_primary"])
        for f2, k in ((f"h1_{sp}_global_vjepa2.json", "Global"),
                      (f"h1_{sp}_plate_dinov2.json", "Dino"),
                      (f"h1_{sp}_plate_tijepa1.json", "Tijepa")):
            q = A / "h1" / f2
            if q.exists():
                t = json.loads(q.read_text())["G_Signal_2"]["vs_action_agnostic"]
                m |= {f"HOne{S}{k}Ag": rel(t["relative_change"]),
                      f"HOne{S}{k}AgP": pv(t["p_one_sided"]),
                      f"HOne{S}{k}AgDir": "lower" if t["mean"] < 0 else "higher"}
    q = A / "h1" / "h1_screenB_global_vjepa2.json"
    if q.exists():
        s = json.loads(q.read_text())["G_Signal_2"]
        m |= {"ScreenBPlates": n(s["vs_action_agnostic"]["n_groups"]),
              "ScreenBAg": rel(s["vs_action_agnostic"]["relative_change"]),
              "ScreenBAgP": pv(s["vs_action_agnostic"]["p_one_sided"]),
              "ScreenBDrift": rel(s["vs_replicate_drift"]["relative_change"]),
              "ScreenBDriftP": pv(s["vs_replicate_drift"]["p_one_sided"])}
    X = A / "exploratory"
    if (X / "e1/h1/h1_test_plate_vjepa2.json").exists():
        for sp, S in (("val", "Val"), ("test", "Test")):
            t = json.loads((X / f"e1/h1/h1_{sp}_plate_vjepa2.json").read_text())["G_Signal_2"]
            m |= {f"EOne{S}Ag": rel(t["vs_action_agnostic"]["relative_change"]),
                  f"EOne{S}AgP": pv(t["vs_action_agnostic"]["p_one_sided"])}
    R = A / "robustness"
    if all((R / f"{k}/{k}_{sp}.json").exists() for k in ("r1", "r2", "r3", "r4")
           for sp in ("val", "test")):
        out = {}
        for sp, S in (("val", "Val"), ("test", "Test")):
            r1, r2, r3, r4 = (json.loads((R / f"{k}/{k}_{sp}.json").read_text())
                              for k in ("r1", "r2", "r3", "r4"))
            t1 = r1["tests"]["real_vs_trained_shuffled"]
            sh = r1["tests"]["trained_shuffled_vs_selected_agnostic"]
            lam = r2["lambda_validation"]
            t2 = r2["tests"][f"real_vs_R2_lambda_{lam}"]
            r2ag = r2["tests"]["R2_lambda_validation_vs_selected_agnostic"]
            na = r1["tests"]["config_matched_agnostic_vs_selected_agnostic"]
            m |= {f"RobNoneAg{S}": f"{abs(100 * na['relative_change']):.2f}",
                  f"RobNoneAgP{S}": pv(na["p_one_sided"])}
            a = r3["a_token_zeroed"]
            c = r3["c_global_token_zeroed"]
            b = r3["b_labels_permuted_within_plate"]["vs_selected_agnostic"]
            ok = lambda t: t["p_one_sided"] < 0.05 and t["mean"] < 0  # noqa: E731
            i1 = ok(t1)
            out[S] = {"I1": i1, "I2": i1 and ok(t2),
                      "I3": i1 and a["fraction_of_gain_retained"] >= 0.5
                      and ok(a["vs_selected_agnostic"])}
            m |= {f"RobOneRel{S}": rel(t1["relative_change"]),
                  f"RobOneDir{S}": "lower" if t1["mean"] < 0 else "higher",
                  f"RobOneP{S}": pv(t1["p_one_sided"]),
                  f"RobShufAg{S}": rel(sh["relative_change"]),
                  f"RobShufAgDir{S}": "lower" if sh["mean"] < 0 else "higher",
                  f"RobShufAgP{S}": pv(sh["p_one_sided"]),
                  f"RobShufFrac{S}": f"{100 * r1['fraction_of_gain_over_selected_agnostic_reached_by_trained_shuffled']:.0f}",
                  f"RobTwoLambda": f"{lam:g}",
                  f"RobTwoRel{S}": rel(t2["relative_change"]),
                  f"RobTwoDir{S}": "lower" if t2["mean"] < 0 else "higher",
                  f"RobTwoP{S}": pv(t2["p_one_sided"]),
                  f"RobTwoAg{S}": rel(r2ag["relative_change"]),
                  f"RobTwoAgDir{S}": "lower" if r2ag["mean"] < 0 else "higher",
                  f"RobTwoAgP{S}": pv(r2ag["p_one_sided"]),
                  f"RobThreeFrac{S}": f"{100 * a['fraction_of_gain_retained']:.0f}",
                  f"RobThreeP{S}": pv(a["vs_selected_agnostic"]["p_one_sided"]),
                  f"RobThreeGlobalFrac{S}": f"{100 * c['fraction_of_gain_retained']:.0f}",
                  f"RobThreeGlobalP{S}": pv(c["vs_selected_agnostic"]["p_one_sided"]),
                  f"RobThreePermRel{S}": rel(b["relative_change"]),
                  f"RobThreePermDir{S}": "lower" if b["mean"] < 0 else "higher",
                  f"RobThreePermP{S}": pv(b["p_one_sided"]),
                  f"RobFourNonPrimRel{S}": rel(r4["non_primary_library_wells"]["relative_change"]),
                  f"RobFourNonPrimP{S}": pv(r4["non_primary_library_wells"]["p_one_sided"])}
            for h, t in r4["per_horizon_primary"].items():
                k = {"3h": "Three", "12h": "Twelve", "24h": "TwentyFour", "36h": "ThirtySix"}[h]
                m |= {f"RobFour{k}h{S}": rel(t["relative_change"]),
                      f"RobFour{k}hP{S}": pv(t["p_one_sided"])}
            m |= {f"RobIOne{S}": word(out[S]["I1"]), f"RobITwo{S}": word(out[S]["I2"]),
                  f"RobIThree{S}": word(out[S]["I3"])}
    return m


def main() -> int:
    m: dict[str, str] = {}
    ps = pd.read_csv(REPO / "splits" / "plate_split.tsv", sep="\t").split.value_counts()
    m |= {"NPlatesTrain": n(ps["train"]), "NPlatesVal": n(ps["val"]),
          "NPlatesTest": n(ps["test"])}
    cov = j("results/actions_coverage.json")
    mg = cov["mitocheck_genes"]
    m |= {"NGenesTrain": n(mg["train"]), "NGenesVal": n(mg["val"]),
          "NGenesTest": n(mg["test"])}
    names = {"perturbseq": "Perturbseq", "jump_crispr": "JumpCrispr", "jump_orf": "JumpOrf",
             "string": "String", "esmtwo": "EsmTwo"}
    for src, v in cov.items():
        if src == "mitocheck_genes":
            continue
        k = names["esmtwo" if src == "esm2" else src]
        m |= {f"Cov{k}Val": n(v["val"]), f"Cov{k}Test": n(v["test"])}
    gd = j("results/gates/G_Data.json")
    m |= {"NMoviesStageOne": n(gd["movies_embedded"]), "NPlatesStageOne": n(gd["plates"])}
    s2 = j("results/a2/stage2_data.json")
    m |= {"NWellsStageTwoSelected": n(s2["wells_selected"]),
          "NWellsStageTwoEmbedded": n(s2["wells_embedded"]),
          "NPlatesStageTwoSelected": n(s2["plates_selected"]),
          "NPlatesStageTwoEmbedded": n(s2["plates_embedded"])}
    mf = j("results/a2/missing_frames.json")
    m |= {"NWellsMissingFrames": n(mf["wells_failed"]),
          "NWellsMissingOnlyLate": n(sum(v["all_missing_ge_88"]
                                         for v in mf["by_split"].values())),
          "NWholePlatesMissing": n(len(mf["whole_plates_missing"])),
          "NPrimaryValMissing": n(mf["by_split"]["val"]["h1_primary_wells"])}
    cs = j("results/a2/cache_stats_vjepa2.json")
    m |= {"NCacheWells": n(cs["n_wells"]), "NCachePlates": n(cs["n_plates"]),
          "NNegPerPlateMin": n(cs["plate_norm"]["neg_wells_per_plate"]["min"]),
          "NNegPerPlateMax": n(cs["plate_norm"]["neg_wells_per_plate"]["max"])}
    fz = j("manifests/amend02_freeze.json")
    m |= {"NPrimaryWellsVal": n(fz["h1_primary_wells"]["by_plate_split"]["val"]),
          "NPrimaryWellsTest": n(fz["h1_primary_wells"]["by_plate_split"]["test"]),
          "NPrimaryGenesVal": n(fz["h2_primary_genes"]["gene_split"]["val"]),
          "NPrimaryGenesTest": n(fz["h2_primary_genes"]["gene_split"]["test"])}
    for p, k in (("mitotic_delay", "MitoticDelay"), ("polylobed", "Polylobed"),
                 ("binuclear", "Binuclear")):
        u = fz["truth"][p]["unseen_counts"]
        m |= {f"NTruth{k}Val": n(u["gene_split_val"]),
              f"NTruth{k}Test": n(u["gene_split_test"])}
    pr = j("manifests/amend02_protocol.json")
    g, fx = pr["grid"], pr["fixed_not_tuned"]
    fams = pr["fixed_before_any_amended_model_2026-10-20"]["families"]
    ncfg = len(g["predictor"]) * len(g["dropout"]) * len(g["weight_decay"])
    m |= {"NFamilies": n(len(fams)), "NConfigs": n(ncfg), "NSeeds": n(len(g["seeds"])),
          "NGridRuns": n(len(fams) * ncfg * len(g["seeds"])),
          "WidthSmall": n(g["predictor"]["22M"]["width"]),
          "DepthSmall": n(g["predictor"]["22M"]["depth"]),
          "ParamsSmall": f"{g['predictor']['22M']['core_params_M']:.1f}",
          "WidthLarge": n(g["predictor"]["50M"]["width"]),
          "DepthLarge": n(g["predictor"]["50M"]["depth"]),
          "ParamsLarge": f"{g['predictor']['50M']['core_params_M']:.1f}",
          "Dropouts": ", ".join(str(x) for x in g["dropout"]),
          "WeightDecays": ", ".join(str(x) for x in g["weight_decay"]),
          "LearningRate": f"{fx['lr']:g}", "BatchWells": n(fx["batch_wells"]),
          "MaxSteps": n(fx["max_steps"]), "EvalEvery": n(fx["eval_every"]),
          "RolloutSteps": n(fx["rollout_steps"]),
          "SirnaDropout": f"{fx['sirna_token_dropout']:g}",
          "Patience": n(int(re.search(r"not improved for (\d+) steps",
                                      fx["early_stopping"])[1]))}
    m |= results_macros()
    lines = ["% generated by scripts/91_paper_numbers.py from results/, manifests/ and "
             "splits/; do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(m)} macros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
