"""E1, spatial batch correction (completion plan v2 S7, v3 S15): exploratory, after the test.

  cache   data/cache/<base>_e1bscore/: the section-1 (plate) cache with a per-plate, per-clip
          row/column median polish of the library wells' tile-mean latents subtracted from
          every well (B-score style; src/geneworld/explore.py). Base `vjepa2` for validation,
          `vjepa2_test` (built by scripts/run_stage1_test.sh) for test.
  probes  plate probes of scripts/53_plate_probe.py, main (validation plates) and (a) (within
          training series), on the corrected and on the uncorrected plate cache.
  train   one model: --family h1_learned|h1_none --seed S, its selected configuration
          (results/a2/selection.json), trained by scripts/50_train_a2.py on the corrected
          cache; checkpoints in checkpoints/a2_e1/, summaries in
          results/a2/exploratory/e1/train/.
  eval    scripts/61_eval_h1_a2.py with those models on the corrected cache (replicate drift
          and every baseline recomputed in that space); results/a2/exploratory/e1/h1/.
No gated file, checkpoint or frozen list is read for writing; every output carries the label
"exploratory, after the test".
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import cache, explore  # noqa: E402
from geneworld.train import a2  # noqa: E402

TAG = "_e1bscore"
OUT = REPO / "results" / "a2" / "exploratory" / "e1"
CK = REPO / "checkpoints" / "a2_e1"


def script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_cache(base: str) -> Path:
    src, dst = cache.CACHE / base, cache.CACHE / f"{base}{TAG}"
    dst.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(src / "meta.tsv", sep="\t", low_memory=False)
    lib = meta.control.fillna("").eq("").to_numpy() & meta.plate_norm_ok.to_numpy()
    z = np.load(src / "plate.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(dst / "plate.npy", "w+", z.dtype, z.shape)
    out[:] = explore.bscore_correct(z, meta, lib)
    out.flush()
    for f in ("meta.tsv", "plate_stats.npz", "fingerprint.npy"):
        shutil.copy(src / f, dst / f)
    st = json.loads((src / "stats.json").read_text())
    st["e1"] = {"label": explore.LABEL, "base_cache": base,
                "correction": "per plate, clip and dimension: Tukey median polish (10 "
                              "iterations) of the library wells' tile-mean latents after "
                              "section-1 normalisation; overall + row + column effects "
                              "subtracted from every well of the plate (all tiles)"}
    (dst / "stats.json").write_text(json.dumps(st, indent=1))
    return dst


def base_suffix(base: str) -> str:
    """'vjepa2' -> '', 'vjepa2_test' -> '_test'"""
    if not base.startswith("vjepa2"):
        raise SystemExit("E1 uses the V-JEPA 2 caches")
    return base[len("vjepa2"):]


def probes(base: str) -> dict:
    pp = script("53_plate_probe.py")
    res = {"label": explore.LABEL, "clip": 12, "base_cache": base}
    bs = base_suffix(base)
    for name, suffix in (("plate", bs), ("e1_bscore", bs + TAG)):
        c = cache.load("vjepa2", "plate", suffix=suffix)
        m = c.meta
        x = c.z[:, 12].astype(np.float32).mean(1)
        val = (m.plate_split == "val").to_numpy()
        res[name] = {"main_validation_plates": pp.stratified(x[val], m.plate.to_numpy()[val]),
                     "within_series_replicates": explore.within_series_probe(x, m)}
    return res


def patch_dirs() -> None:
    """Point the trainer and the H1 evaluator at the exploratory directories; the selection
    is read from a copy of results/a2/selection.json (hash recorded)."""
    OUT.mkdir(parents=True, exist_ok=True)
    sel = REPO / "results" / "a2" / "selection.json"
    shutil.copy(sel, OUT / "selection.json")
    (OUT / "selection_source.json").write_text(json.dumps(
        {"copied_from": "results/a2/selection.json",
         "sha256": hashlib.sha256(sel.read_bytes()).hexdigest()}, indent=1))
    a2.CK_DIR, a2.RES_DIR = CK, OUT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "probes", "train", "eval"])
    ap.add_argument("--base", default="vjepa2", help="vjepa2 (validation) or vjepa2_test")
    ap.add_argument("--family", choices=["h1_learned", "h1_none"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--freeze-commit", default="")
    args = ap.parse_args()
    suffix = base_suffix(args.base) + TAG
    if args.cmd == "cache":
        print(build_cache(args.base))
    elif args.cmd == "probes":
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"probes{base_suffix(args.base)}.json").write_text(
            json.dumps(probes(args.base), indent=1))
        print("written (numbers not printed)")
    elif args.cmd == "train":
        patch_dirs()
        sel = json.loads((OUT / "selection.json").read_text())["selected"][args.family]
        tr = script("50_train_a2.py")
        sys.argv = ["50_train_a2.py", "--family", args.family, "--config", sel, "--seed",
                    str(args.seed), "--norm", "plate", "--backbone", "vjepa2",
                    "--suffix", suffix]
        return tr.main()
    else:
        patch_dirs()
        ev = script("61_eval_h1_a2.py")
        sys.argv = ["61_eval_h1_a2.py", "--split", args.split, "--suffix", suffix]
        if args.freeze_commit:
            sys.argv += ["--freeze-commit", args.freeze_commit]
        rc = ev.main()
        f = OUT / "h1" / f"h1_{args.split}_plate_vjepa2.json"
        d = json.loads(f.read_text())
        d = {"label": explore.LABEL, "e1_cache": f"vjepa2{suffix}", **d}
        f.write_text(json.dumps(d, indent=1, default=float))
        return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
