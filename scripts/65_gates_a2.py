"""Collect the amended gates (amendment 02 section 4) into one committed results file.

  --split val   results/a2/gates_val.json, written before the freeze (step 6)
  --split test  results/a2/gates_test.json (after the one-time test evaluation)
Each gate records its rule, outcome, the source results file and its SHA-256, and the
consequence named by the amendment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.train import a2  # noqa: E402

# Erratum E-02.1 (docs/DESIGN_AMENDMENT_02.md section 10): trajectory half of G-Plan-2 is
# judged against kNN-imputed endpoint retrieval, not the replicate-endpoint oracle.
TRAJECTORY_GATE_COMPARATOR = "retrieval_knn"


def rec(path: Path) -> dict:
    return {"file": str(path.relative_to(REPO)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()
    sp = args.split
    h1 = a2.RES_DIR / "h1" / f"h1_{sp}_plate_vjepa2.json"
    h2 = a2.RES_DIR / "h2" / f"h2_{sp}_plate_vjepa2.json"
    src = json.loads((a2.RES_DIR / "h2" / "h2_val_plate_vjepa2.json").read_text())[
        "planning_source"]
    h3 = a2.RES_DIR / "h3" / f"h3_{sp}_plate_vjepa2_{src}.json"
    r1, r2, r3 = (json.loads(p.read_text()) for p in (h1, h2, h3))
    comp = r3["trajectory_targets"].get("gate_comparator")
    if comp != TRAJECTORY_GATE_COMPARATOR:
        sys.exit(f"{h3.name}: trajectory gate comparator is {comp!r}, erratum E-02.1 requires "
                 f"{TRAJECTORY_GATE_COMPARATOR!r}")
    sig = r1["G_Signal_2"]
    out = {"split": sp, "design_commit": a2.DESIGN_COMMIT,
           "freeze_commit": r1.get("freeze_commit"),
           "selection": rec(a2.RES_DIR / "selection.json"),
           "planning_source": src,
           "G_Signal_2": {**sig, "source": rec(h1)},
           "G_ZeroShot_2": {**r2["G_ZeroShot_2"], "source": rec(h2),
                            "per_source": {s: {"pass": v["G_ZeroShot_2_pass"],
                                               "true_vs_shuffled_trained":
                                                   v["primary"]["true_vs_shuffled_trained"],
                                               "true_vs_agnostic":
                                                   v["primary"]["true_vs_agnostic"]}
                                           for s, v in r2["sources"].items()}},
           "G_Plan_2": {**r3["G_Plan_2"], "source": rec(h3),
                        "trajectory_gate_comparator": comp,
                        "trajectory_vs_comparator": r3["trajectory_targets"][
                            f"planner_vs_{comp}_hit@100"],
                        "oracle_ceiling_hit@100": r3["trajectory_targets"]["methods"][
                            "oracle_ceiling_replicate_endpoint"]["hit@100"]},
           "errata": ["E-02.1 (2026-10-22): G-Plan-2 trajectory comparator = kNN-imputed "
                      "endpoint retrieval"],
           "consequence": ("G-Signal-2 passed: H2/H3 are confirmatory" if sig["pass"] else
                           "G-Signal-2 failed: H2/H3 are exploratory (amendment 02 section 4)"),
           "code_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short",
                                          "HEAD"], capture_output=True, text=True).stdout.strip()}
    f = a2.RES_DIR / f"gates_{sp}.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    print(f"{sp}: G-Signal-2 {sig['pass']}, G-ZeroShot-2 {r2['G_ZeroShot_2']['pass']}, "
          f"G-Plan-2 {r3['G_Plan_2']['pass']} (source {src})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
