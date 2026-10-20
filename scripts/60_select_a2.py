"""Select one grid configuration per family on validation (amendment 02 section 5).

For every family, the configuration with the lowest mean over its three seeds of the best
validation primary error (per-plate normalisation, V-JEPA 2, gene split). All 8 x 3 runs of a
family must exist. Writes results/a2/selection.json (to be committed before any gate).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.train import a2  # noqa: E402


def main() -> int:
    fams = a2.protocol()["fixed_before_any_amended_model_2026-10-20"]["families"]
    seeds = a2.protocol()["grid"]["seeds"]
    table, selected, missing = {}, {}, []
    for f in fams:
        table[f] = {}
        for cfg in a2.configs():
            vals = []
            for s in seeds:
                p = a2.RES_DIR / "train" / f"{a2.run_name('plate', 'vjepa2', f, cfg['id'], s)}.json"
                if not p.exists():
                    missing.append(p.name)
                    continue
                vals.append(json.loads(p.read_text())["best_val_primary"])
            table[f][cfg["id"]] = {"per_seed": vals, "mean": float(np.mean(vals)) if
                                   len(vals) == len(seeds) else None}
        ok = {k: v["mean"] for k, v in table[f].items() if v["mean"] is not None}
        if len(ok) == len(a2.configs()):
            selected[f] = min(ok, key=ok.get)
    if missing:
        sys.exit(f"{len(missing)} runs missing, e.g. {missing[:3]}")
    out = {"design_commit": a2.DESIGN_COMMIT, "rule": a2.protocol()[
        "fixed_before_any_amended_model_2026-10-20"]["selection"], "selected": selected,
        "table": table}
    (a2.RES_DIR / "selection.json").write_text(json.dumps(out, indent=1))
    for f, cid in selected.items():
        print(f"{f:22s} {cid:18s} {table[f][cid]['mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
