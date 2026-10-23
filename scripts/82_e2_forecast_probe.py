"""E2, batch carried by the forecast (completion plan v2 S7, v3 S15): exploratory, after the
test.

The within-series replicate probe of scripts/53_plate_probe.py (a) (training series with >= 3
training plates, well positions present on every replicate plate, cross-validation grouped by
well position) on clip-12 (36 h) tile-mean latents of the probe's wells:
  observed_clip12      the observed latent (= probe (a) in the plate space)
  observed_clip0       the context the forecasts start from
  forecast_learned     the selected h1_learned models' rollout from clip 0, mean over 3 seeds
  forecast_agnostic    the same for the selected h1_none models
Writes results/a2/exploratory/e2.json, labelled "exploratory, after the test".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import cache, explore  # noqa: E402
from geneworld.eval import a2eval as E  # noqa: E402
from geneworld.train import a2  # noqa: E402


def models(family: str, cfg: str, c) -> list:
    out = []
    for s in a2.protocol()["grid"]["seeds"]:
        name = a2.run_name("plate", "vjepa2", family, cfg, s)
        m, b, _, _ = a2.load_model(a2.CK_DIR / f"{name}.pt", c)
        out.append((m, b))
    return out


def main() -> int:
    sel = json.loads((a2.RES_DIR / "selection.json").read_text())["selected"]
    c = cache.load("vjepa2", "plate")
    m, z = c.meta, c.z
    rows = explore.probe_wells(m)
    feats = {"observed_clip12": z[rows, 12].astype(np.float32).mean(1),
             "observed_clip0": z[rows, 0].astype(np.float32).mean(1)}
    for name, fam in (("forecast_learned", "h1_learned"), ("forecast_agnostic", "h1_none")):
        _, preds = E.seed_errors(models(fam, sel[fam], c), z, rows)
        feats[name] = np.mean([p[:, 11].astype(np.float32).mean(1) for p in preds], 0)
    res = {"label": explore.LABEL, "n_wells": int(len(rows)), "clip": 12,
           "selected": {"h1_learned": sel["h1_learned"], "h1_none": sel["h1_none"]}}
    for k, f in feats.items():
        x = np.zeros((len(m), f.shape[1]), np.float32)
        x[rows] = f
        res[k] = explore.within_series_probe(x, m, rows=rows)
    out = a2.RES_DIR / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e2.json").write_text(json.dumps(res, indent=1))
    print("written results/a2/exploratory/e2.json (numbers not printed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
