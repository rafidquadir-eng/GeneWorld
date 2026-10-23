"""Build reports/RUN_REPORT_STAGE2.md and its figures from the results files only.

Every number is read from a file under results/ (or a frozen manifest); nothing is typed by
hand. Directional words come from the tests' p values and CIs (`verdict`, `ci_word`). A section
whose input does not exist yet says so in one sentence (test results before the test is run;
exploratory results before the test gates exist). Exploratory numbers (E1-E3) are printed only
when results/a2/gates_test.json exists, so they cannot appear before the test.

  python scripts/90_run_report_stage2.py [--out reports/RUN_REPORT_STAGE2.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results" / "a2"
FIG = REPO / "reports" / "figures" / "stage2"
# reference palette slots 1-3 (dataviz skill), light mode; validated all-pairs; slot 3 is
# below 3:1 contrast, so every figure has its numbers in an adjacent table
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load(rel: str):
    p = R / rel
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "pass" if x else "fail"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:.{nd}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{nd}f}".rstrip("0").rstrip(".")


def pct(x):
    return "–" if x is None else f"{100 * x:+.1f}%"


def ci(c):
    return "–" if not c else f"[{f(c[0])}, {f(c[1])}]"


def verdict(t: dict) -> str:
    """One-sided plate test of model - baseline: words from p and sign."""
    if not t or t.get("p_one_sided") is None:
        return "not tested"
    better = t["p_one_sided"] < 0.05 and t["mean"] < 0
    return f"{'beats' if better else 'does not beat'} (p = {f(t['p_one_sided'], 3)})"


def ci_word(c, below="lower", above="higher") -> str:
    if not c:
        return "not estimable"
    return below if c[1] < 0 else above if c[0] > 0 else "not distinguishable"


def table(head: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


class Doc:
    def __init__(self):
        self.lines: list[str] = []

    def h(self, level: int, text: str):
        self.lines += [f"{'#' * level} {text}", ""]

    def p(self, text: str):
        self.lines += [text, ""]

    def t(self, head, rows):
        self.lines += table(head, rows)

    def img(self, path: Path, alt: str):
        rel = (path.relative_to(REPO / "reports").as_posix()
               if path.is_relative_to(REPO / "reports") else path.as_posix())
        self.lines += [f"![{alt}]({rel})", ""]


def _axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)


def fig_h1(h1v, h1t) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not h1v:
        return None
    methods = list(h1v["mean_error"])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(methods) + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(methods))
    for res, col, lab, dy in ((h1v, C1, "validation", -0.12), (h1t, C2, "test", 0.12)):
        if res:
            v = [res["mean_error"].get(m, {}).get("primary_wells_24_36h") for m in methods]
            ax.scatter(v, [i + dy for i in y], s=36, color=col, label=lab, zorder=3,
                       edgecolors=SURFACE, linewidths=2)
    ax.set_yticks(list(y), [m.replace("_", " ") for m in methods], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("mean squared error, primary wells, 24/36 h (normalised space)", color=INK2)
    _axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=3, borderaxespad=0, handletextpad=0.4, columnspacing=1.2)
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "h1_primary_error.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_h2(h2v) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not h2v:
        return None
    src = list(h2v["sources"])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(src) + 1.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for k, (key, col, lab) in enumerate((("true_vs_shuffled_trained", C1, "true − shuffled"),
                                         ("true_vs_agnostic", C2, "true − agnostic"))):
        for i, s in enumerate(src):
            g = h2v["sources"][s]["primary"][key]
            if g.get("ci95") is None:
                continue
            yy = i + (k - 0.5) * 0.24
            ax.plot(g["ci95"], [yy, yy], color=col, lw=2, solid_capstyle="round")
            ax.scatter([g["mean"]], [yy], s=36, color=col, zorder=3, edgecolors=SURFACE,
                       linewidths=2, label=lab if i == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(src)), src, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("difference in mean error over primary genes (95% CI); < 0 favours the "
                  "true encoding", color=INK2)
    _axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=3, borderaxespad=0, handletextpad=0.4, columnspacing=1.2)
    out = FIG / "h2_sources.png"
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_probe(pp) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not pp:
        return None
    rows = [("main, validation plates", pp["main_validation_plates"], "accuracy"),
            ("(a) within series", pp["within_series_replicates"], "mean_accuracy_")]
    fig, ax = plt.subplots(figsize=(7, 2.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for k, (v, col) in enumerate((("global", C1), ("plate", C2), ("centre_only", C3))):
        vals = [r[v]["accuracy"] if key == "accuracy" else r[f"{key}{v}"]
                for _, r, key in rows]
        ax.barh([i + (k - 1) * 0.26 for i in range(len(rows))], vals, height=0.26, color=col,
                edgecolor=SURFACE, linewidth=2,
                label={"plate": "normalised", "centre_only": "centre only"}.get(v, v))
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("plate decoding accuracy, clip 12 (36 h)", color=INK2)
    _axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=3, borderaxespad=0, handletextpad=0.4, columnspacing=1.2)
    out = FIG / "plate_probe.png"
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def section_gates(d: Doc, gv, gt):
    d.h(2, "1. Gates")
    rows = []
    for g, key in (("G-Signal-2", "G_Signal_2"), ("G-ZeroShot-2", "G_ZeroShot_2"),
                   ("G-Plan-2", "G_Plan_2")):
        rows.append([g, f(gv[key]["pass"]) if gv else "not yet computed",
                     f(gt[key]["pass"]) if gt else "test not yet run"])
    d.t(["gate", "validation", "test"], rows)
    if gv:
        d.p(f"Consequence on validation: {gv['consequence']}. Planning source: "
            f"`{gv['planning_source']}`. Errata applied: {'; '.join(gv['errata'])}.")
    if not gt:
        d.p("The test split is evaluated once, after the freeze; its results are not yet "
            "available.")


def section_h1(d: Doc, h1v, h1t, extra: dict):
    d.h(2, "2. H1: action conditioning on seen genes (G-Signal-2)")
    if not h1v:
        d.p("Validation H1 results are not yet available.")
        return
    for name, r in (("validation", h1v), ("test", h1t)):
        if not r:
            d.p(f"H1 {name}: not yet available.")
            continue
        g = r["G_Signal_2"]
        d.p(f"**{name.capitalize()}:** {r['n_wells_primary']:,} primary wells on "
            f"{r['n_plates_primary']} plates. The action-conditioned predictor "
            f"{verdict(g['vs_action_agnostic'])} the action-agnostic predictor and "
            f"{verdict(g['vs_replicate_drift'])} replicate drift; G-Signal-2 "
            f"{f(g['pass'])}. Rule: {g['rule']}.")
        rows = []
        for m, e in r["mean_error"].items():
            t = r["tests"].get(m, {})
            rows.append([m, f(e["primary_wells_24_36h"]), f(e["all_wells_24_36h"]),
                         pct(t.get("primary", {}).get("relative_change")) if t else "",
                         f(t.get("primary", {}).get("p_one_sided"), 3) if t else "",
                         f(t.get("secondary_well_weighted_primary", {}).get("p_one_sided"), 3)
                         if t else ""])
        d.t(["method", "primary 24/36 h", "all wells 24/36 h",
             "action-conditioned vs this (primary)", "p (plates)", "p well-weighted (F6)"],
            rows)
        fb = r.get("replicate_baseline_sources")
        if fb:
            d.p(f"Replicate-baseline sources (fallback counts): `{json.dumps(fb)}`.")
        pr = r.get("phenotype_probe_36h", {})
        prow = []
        for k, v in pr.items():
            if "auroc" not in v:
                prow.append([k, v.get("n_pos_eval"), v.get("n_neg_eval"), "skipped", "", ""])
                continue
            a = v["auroc"]
            dd = v.get("difference", {})
            prow.append([k, v["n_pos_eval"], v["n_neg_eval"],
                         " · ".join(f"{m} {f(x['value'], 3)} {ci(x['ci95'])}"
                                    for m, x in a.items()),
                         " · ".join(f"{m}: {f(x['value'], 3)} {ci(x['ci95'])} "
                                    f"({ci_word(x['ci95'])})" for m, x in dd.items()), ""])
        if prow:
            d.p("Phenotype probe at 36 h (AUROC, 95% cluster-bootstrap CI over plates):")
            d.t(["class", "n pos", "n neg", "AUROC by representation", "differences", ""],
                prow)
    for label, r in extra.items():
        if r:
            g = r["G_Signal_2"]
            ag = g["vs_action_agnostic"]
            d.p(f"*{label}:* vs action-agnostic {pct(ag['relative_change'])}"
                f" ({verdict(ag)}); vs replicate drift "
                f"{pct(g['vs_replicate_drift']['relative_change'])} "
                f"({verdict(g['vs_replicate_drift'])}).")
        else:
            d.p(f"*{label}:* not yet available.")


def section_h2(d: Doc, h2v, h2t, extra: dict):
    d.h(2, "3. H2: zero-shot transfer to unseen genes (G-ZeroShot-2)")
    if not h2v:
        d.p("Validation H2 results are not yet available.")
        return
    for name, r in (("validation", h2v), ("test", h2t)):
        if not r:
            d.p(f"H2 {name}: not yet available.")
            continue
        rows = []
        for s, v in r["sources"].items():
            p = v["primary"]
            a, b = p["true_vs_shuffled_trained"], p["true_vs_agnostic"]
            rows.append([s, p["n_genes"], f(a.get("mean")), ci(a.get("ci95")),
                         ci_word(a.get("ci95")), f(b.get("mean")), ci(b.get("ci95")),
                         ci_word(b.get("ci95")), f(v["G_ZeroShot_2_pass"])])
        d.p(f"**{name.capitalize()}:** G-ZeroShot-2 {f(r['G_ZeroShot_2']['pass'])} "
            f"(passing sources: {', '.join(r['G_ZeroShot_2']['passing_sources']) or 'none'}). "
            f"Rule: {r['G_ZeroShot_2']['rule']}.")
        d.t(["source", "primary genes", "true − shuffled", "95% CI", "", "true − agnostic",
             "95% CI", "", "pass"], rows)
    for label, r in extra.items():
        d.p(f"*{label}:* " + (f"G-ZeroShot-2 rule {f(r['G_ZeroShot_2']['pass'])}"
                               if r else "not yet available."))


def section_h3(d: Doc, h3v, h3t, extra: dict):
    d.h(2, "4. H3: planning (G-Plan-2)")
    if not h3v:
        d.p("Validation H3 results are not yet available.")
        return
    for name, r in (("validation", h3v), ("test", h3t)):
        if not r:
            d.p(f"H3 {name}: not yet available.")
            continue
        g = r["G_Plan_2"]
        tr = r["trajectory_targets"]
        d.p(f"**{name.capitalize()}:** phenotype half {f(g['phenotype_pass'])}, trajectory "
            f"half {f(g['trajectory_pass'])}; G-Plan-2 {f(g['pass'])}. Rule: {g['rule']}.")
        rows = [[m, f(v["hit@1"], 3), f(v["hit@10"], 3), f(v["hit@100"], 3), f(v["mrr"], 3),
                 f(v["median_rank"])] for m, v in tr["methods"].items()]
        rows.append(["random", f(tr["random"]["hit@1"], 3), f(tr["random"]["hit@10"], 3),
                     f(tr["random"]["hit@100"], 3), "", ""])
        d.t(["trajectory method", "hit@1", "hit@10", "hit@100", "MRR", "median rank"], rows)
        cmp_ = tr[f"planner_vs_{tr['gate_comparator']}_hit@100"]
        d.p(f"{tr['n_targets']} targets ({tr['n_target_genes']} genes), {tr['n_candidates']} "
            f"candidates. Planner vs random hit@100: binomial p = "
            f"{f(tr['planner_vs_random_hit@100_p'], 3)}. Planner − `{tr['gate_comparator']}` "
            f"(gate comparator, erratum E-02.1): {f(cmp_['difference'], 3)} "
            f"{ci(cmp_['ci95'])} "
            f"({ci_word(cmp_['ci95'], 'lower', 'higher')}). "
            f"`oracle_ceiling_replicate_endpoint` = {tr['oracle_ceiling_label']}.")
        ph = r.get("phenotype", r.get("sets"))
        if isinstance(ph, dict):
            d.p(f"Phenotype-planning details: see `{name}` results file (structure: "
                f"{', '.join(sorted(ph))[:120]}).")
    for label, r in extra.items():
        d.p(f"*{label}:* " + (f"G-Plan-2 rule {f(r['G_Plan_2']['pass'])}" if r
                               else "not yet available."))


def section_data(d: Doc):
    d.h(2, "5. Data, exclusions and missing frames")
    s2 = load("stage2_data.json")
    mf = load("missing_frames.json")
    if s2:
        d.p(f"Stage 2 embedded {s2['wells_embedded']:,} of {s2['wells_selected']:,} selected "
            f"wells on {s2['plates_embedded']} of {s2['plates_selected']} plates.")
    if mf:
        rows = [[s, v["wells"], v["plates"], v["h1_primary_wells"], v["h2_primary_gene_wells"],
                 v["all_missing_ge_88"]] for s, v in mf["by_split"].items()]
        d.p(f"S2 ({mf['rule']}): {mf['wells_failed']:,} wells failed; "
            f"{mf['frames_missing']:,} of {mf['frames_requested']:,} frames missing; "
            f"{mf['wells_recovered']} recovered; "
            f"whole plates lost: {', '.join(mf['whole_plates_missing'])}.")
        d.t(["split", "wells lost", "plates", "H1 primary wells", "primary-gene wells",
             "missing only frames ≥ 88 (F9)"], rows)
    for sp in ("test", "screenB"):
        acc = load(f"missing_frames_{sp}.json")
        d.p(f"S3 accounting, {sp}: " + (f"see `results/a2/missing_frames_{sp}.json`" if acc
                                        else "not yet available (embedded after the freeze)."))


def section_probe(d: Doc):
    d.h(2, "6. Plate-identity probes and R1")
    pp = load("plate_probe.json")
    if not pp:
        d.p("Plate probes are not available.")
        return
    rows = [["main: validation plates", f(pp["main_validation_plates"]["global"]["chance"], 3)]
            + [f(pp["main_validation_plates"][v]["accuracy"], 3)
               for v in ("global", "plate", "centre_only")],
            ["(a) within-series replicates",
             f(pp["within_series_replicates"]["mean_chance"], 3)]
            + [f(pp["within_series_replicates"][f"mean_accuracy_{v}"], 3)
               for v in ("global", "plate", "centre_only")]]
    nc = pp.get("negative_controls_loo")
    if nc:
        rows.append(["(b) negative controls, leave-one-out (uninformative after "
                     "normalisation; global value only)", f(nc["global"].get("chance"), 3),
                     f(nc["global"]["accuracy"], 3), "uninformative", "uninformative"])
    d.t(["probe", "chance", "global", "normalised", "centre only"], rows)
    out = fig_probe(pp)
    if out:
        d.img(out, "plate-identity probe accuracy by preprocessing")


def section_grid(d: Doc):
    d.h(2, "7. Grid, selection and hardware")
    g = load("grid_runs.json")
    sel = load("selection.json")
    if g:
        d.p(f"{g['n_runs']} grid runs: {json.dumps(g['status_counts'])}; completed on "
            f"{json.dumps(g['completed_on'])}; failed attempts by incident "
            f"{json.dumps(g['failed_attempts_by_incident'])} "
            f"(`results/a2/grid_runs.json`; reconstructed values are labelled there).")
        for n in g.get("notes", []):
            d.p(f"Note: {n}.")
    if sel:
        d.t(["family", "selected configuration"], [[k, v] for k, v in sel["selected"].items()])
    ej = load("eval_jobs.json")
    if ej:
        d.t(["validation job", "GPU", "minutes", "exit"],
            [[k, ",".join(v["gpus"]) or "CPU", v.get("minutes"), v.get("exit_code")]
             for k, v in ej["jobs"].items()])


def section_exploratory(d: Doc, gt):
    d.h(2, "Exploratory, after the test")
    if not gt:
        d.p("E1, E2 and E3 run only after the test; they have not been run.")
        return
    d.p("Every number in this section is exploratory, after the test, and is never used to "
        "revise a gate or claim.")
    e1 = {s: load(f"exploratory/e1/h1/h1_{s}_plate_vjepa2.json") for s in ("val", "test")}
    e2 = load("exploratory/e2.json")
    e3 = load("exploratory/e3/compare.json")
    for s, r in e1.items():
        g = r["G_Signal_2"] if r else None
        d.p(f"E1 ({s}): " + (f"vs action-agnostic {verdict(g['vs_action_agnostic'])}; vs "
                             f"replicate drift {verdict(g['vs_replicate_drift'])}" if g
                             else "not run."))
    if e2:
        d.t(["E2 representation", "series", "mean accuracy", "chance"],
            [[k, v["n_series"], f(v["mean_accuracy"], 3), f(v["mean_chance"], 3)]
             for k, v in e2.items() if isinstance(v, dict) and "mean_accuracy" in v])
    else:
        d.p("E2: not run.")
    d.p("E3: " + (json.dumps({k: v for k, v in e3.items() if k != "label"}) if e3
                  else "not run."))


def build() -> str:
    d = Doc()
    gv, gt = load("gates_val.json"), load("gates_test.json")
    d.h(1, "GeneWorld stage 2 (amendment 02): run report")
    d.p("Generated by `scripts/90_run_report_stage2.py` from the results files; every number "
        "below is read from a file under `results/`. Test results: " +
        ("present." if gt else "not yet run (placeholders state this)."))
    section_gates(d, gv, gt)
    h1v, h1t = load("h1/h1_val_plate_vjepa2.json"), load("h1/h1_test_plate_vjepa2.json")
    section_h1(d, h1v, h1t, {
        "Global preprocessing (sensitivity), validation": load("h1/h1_val_global_vjepa2.json"),
        "DINOv2, validation": load("h1/h1_val_plate_dinov2.json"),
        "TiJepa-1, validation": load("h1/h1_val_plate_tijepa1.json"),
        "ScreenB (global, outside the gates)": load("h1/h1_screenB_global_vjepa2.json")})
    out = fig_h1(h1v, h1t)
    if out:
        d.img(out, "H1 mean primary error by method, validation and test")
    h2v, h2t = load("h2/h2_val_plate_vjepa2.json"), load("h2/h2_test_plate_vjepa2.json")
    section_h2(d, h2v, h2t, {
        "Global preprocessing, validation": load("h2/h2_val_global_vjepa2.json"),
        "Complex split, validation": load("h2/h2_val_plate_vjepa2_complex.json")})
    out = fig_h2(h2v)
    if out:
        d.img(out, "H2 differences per encoding source with 95% CIs")
    src = h2v["planning_source"] if h2v else None
    h3v = load(f"h3/h3_val_plate_vjepa2_{src}.json") if src else None
    h3t = load(f"h3/h3_test_plate_vjepa2_{src}.json") if src else None
    section_h3(d, h3v, h3t, {
        "Global preprocessing, validation": load(f"h3/h3_val_global_vjepa2_{src}.json")
        if src else None,
        "Complex split, validation": load(f"h3/h3_val_plate_vjepa2_{src}_complex.json")
        if src else None})
    section_data(d)
    section_probe(d)
    section_grid(d)
    d.h(2, "8. Controls")
    for n in ("plate", "global"):
        c = load(f"controls/controls_val_{n}_vjepa2.json")
        st = c["static"]["real_vs_static_over_between_wells_median"] if c else None
        d.p(f"{n}: " + (f"static-clip ratio {f(st, 3)}; "
                        f"leakage max r {f(c['leakage']['max_r'], 3)}, "
                        f"{c['leakage']['n_match_r>=0.98']} matches r ≥ 0.98" if c
                        else "not yet available."))
    section_exploratory(d, gt)
    return "\n".join(d.lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/RUN_REPORT_STAGE2.md")
    ap.add_argument("--fig-dir", default="", help="default reports/figures/stage2")
    args = ap.parse_args()
    global FIG
    if args.fig_dir:
        FIG = Path(args.fig_dir).resolve()
    out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
