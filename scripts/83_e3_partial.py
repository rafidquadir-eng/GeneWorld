"""E3, wells excluded only for missing frames >= 88 (completion plan v3 S15): exploratory,
after the test.

  candidates  per split, the excluded wells whose missing frames are all >= 88: train / val
              from results/a2/missing_frames.json (S2); test / screenB from the embed failure
              lists (results/a2/embed_failures_<split>.json, S3 rule), re-requested frame by
              frame as in S2 (HEAD, 3 attempts). Writes
              results/a2/exploratory/e3/candidates_<split>.json.
  embed       frames 0-87 of each candidate (level 1), clips 0-12 embedded with the three
              backbones; clip 13 (frames 77-92) is NaN. Intensity quantiles use the real frames
              (src/geneworld/explore.py partial_clips). Shards in data/latents_e3/<bb>/, each
              backed up with SHA-256 and added to manifests/cache_shards.tsv.
  cache       data/cache/<base>_e3/: the plate cache <base> (vjepa2 for validation, vjepa2_test
              for test) plus the E3 wells of its splits, normalised with the stored per-plate
              statistics (existing wells unchanged; new training-plate wells enter no training
              set or replicate baseline).
  eval        scripts/61_eval_h1_a2.py with the frozen selected models on that cache;
              results/a2/exploratory/e3/h1/.
  compare     whether G-Signal-2's conclusion differs with the E3 wells added
              (results/a2/exploratory/e3/compare.json).
Every output carries the label "exploratory, after the test".
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import backup, cache, explore  # noqa: E402
from geneworld.train import a2  # noqa: E402

OUT = REPO / "results" / "a2" / "exploratory" / "e3"
SHARDS = REPO / "data" / "latents_e3"
TABLES = {"train": "splits/stage2_wells.tsv", "val": "splits/stage2_wells.tsv",
          "test": "splits/test_wells_to_embed.tsv", "screenB": "splits/screenB_wells.tsv"}
SPLITS_OF_BASE = {"vjepa2": ("train", "val"), "vjepa2_test": ("train", "val", "test")}
BACKBONES = ("vjepa2", "dinov2", "tijepa1")


def script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""),
                                                  REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidates(split: str) -> list[dict]:
    if split in ("train", "val"):
        w = json.loads((REPO / "results/a2/missing_frames.json").read_text())["wells"]
        return [{"plate": x["plate"], "well": x["well"], "missing_frames": x["missing_frames"]}
                for x in w if x["plate_split"] == split and x["all_missing_ge_88"]]
    fails = json.loads((REPO / f"results/a2/embed_failures_{split}.json").read_text())
    tab = pd.read_csv(REPO / TABLES[split], sep="\t").set_index(["Plate", "Well"])
    mf = script("72_missing_frames.py")
    from geneworld.io.mitocheck import N_FRAMES, chunk_url
    out = []
    for f in fails:
        plate, well = f.split(":")[:2]
        uuid = tab.loc[(plate, well), "uuid"]
        with ThreadPoolExecutor(16) as ex:
            codes = list(ex.map(mf.head, [chunk_url(uuid, well, 1, t)
                                          for t in range(N_FRAMES)]))
        miss = [t for t, c in enumerate(codes) if c != 200]
        if miss and min(miss) >= explore.VALID_FRAMES_E3:
            out.append({"plate": plate, "well": well, "missing_frames": miss})
    return out


def embed(split: str, streams: int) -> None:
    from geneworld.io import mitocheck as MC
    cand = json.loads((OUT / f"candidates_{split}.json").read_text())["wells"]
    tab = pd.read_csv(REPO / TABLES[split], sep="\t").set_index(["Plate", "Well"])
    emb = script("20_stream_embed.py")
    bb = emb.Backbones(list(BACKBONES))
    by_plate: dict[str, list] = {}
    for x in cand:
        by_plate.setdefault(x["plate"], []).append(x["well"])
    pool = ThreadPoolExecutor(streams)
    man = REPO / "manifests" / "cache_shards.tsv"
    for plate, wells in sorted(by_plate.items()):
        lat = {b: [] for b in BACKBONES}
        fms = []
        for well in wells:
            frames = MC.read_well(tab.loc[(plate, well), "uuid"], well, 1,
                                  frames=range(explore.VALID_FRAMES_E3), pool=pool)
            clips, valid, fm = explore.partial_clips(frames)
            e = bb(clips)
            for b in BACKBONES:
                x = e[b].reshape(explore.N_CLIPS, 4, 16, -1).astype(np.float16)
                x[~valid] = np.nan
                lat[b].append(x)
            fms.append(fm.astype(np.float32))
        for b in BACKBONES:
            d = SHARDS / b
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{plate}.npz"
            np.savez(path, latents=np.stack(lat[b]), wells=np.array(wells),
                     frame_mean=np.stack(fms), clip_starts=np.array(explore.CLIP_STARTS),
                     valid_clips=np.arange(explore.N_CLIPS) < 13, backbone=b, partial=True,
                     n_frames=explore.VALID_FRAMES_E3, label=explore.LABEL)
            row = backup.backup(path)
            with open(man, "a") as fh:
                fh.write(f"{b}\t{path.relative_to(REPO)}\t{row['sha256']}\n")
        print(f"{plate}: {len(wells)} wells", flush=True)


def build(base: str) -> dict:
    rows, raws = [], []
    for split in SPLITS_OF_BASE[base]:
        tab = pd.read_csv(REPO / TABLES[split], sep="\t").set_index(["Plate", "Well"])
        for f in sorted((SHARDS / "vjepa2").glob("*.npz")):
            d = np.load(f)
            for i, w in enumerate(d["wells"]):
                key = (f.stem, str(w))
                if key not in tab.index or tab.loc[key, "plate_split"] != split:
                    continue
                r = tab.loc[key]
                rows.append({"plate": key[0], "well": key[1], "Gene Symbol": r["Gene Symbol"],
                             "siRNA Identifier": r["siRNA Identifier"],
                             "control": r["control"], "series": r["series"],
                             "plate_split": r["plate_split"], "gene_split": r["gene_split"]})
                raws.append(d["latents"][i].astype(np.float32).mean(2))   # (14, 4, D)
    meta_new = pd.DataFrame(rows)
    return explore.extend_cache(cache.CACHE / base, cache.CACHE / f"{base}_e3",
                                np.stack(raws), meta_new,
                                {"splits": list(SPLITS_OF_BASE[base]),
                                 "shards": "data/latents_e3/vjepa2"})


def patch_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sel = REPO / "results" / "a2" / "selection.json"
    shutil.copy(sel, OUT / "selection.json")
    (OUT / "selection_source.json").write_text(json.dumps(
        {"copied_from": "results/a2/selection.json",
         "sha256": hashlib.sha256(sel.read_bytes()).hexdigest()}, indent=1))
    a2.RES_DIR = OUT                       # the frozen models stay in checkpoints/a2/


def compare() -> dict:
    out = {"label": explore.LABEL}
    for split in ("val", "test"):
        f0 = REPO / f"results/a2/h1/h1_{split}_plate_vjepa2.json"
        f1 = OUT / "h1" / f"h1_{split}_plate_vjepa2.json"
        if not (f0.exists() and f1.exists()):
            out[split] = "not available"
            continue
        a, b = json.loads(f0.read_text()), json.loads(f1.read_text())
        out[split] = {"n_wells_primary": [a["n_wells_primary"], b["n_wells_primary"]],
                      "G_Signal_2_pass": [a["G_Signal_2"]["pass"], b["G_Signal_2"]["pass"]],
                      "conclusion_differs": a["G_Signal_2"]["pass"] != b["G_Signal_2"]["pass"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["candidates", "embed", "cache", "eval", "compare"])
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "screenB"])
    ap.add_argument("--base", default="vjepa2", choices=sorted(SPLITS_OF_BASE))
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--freeze-commit", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "candidates":
        c = candidates(args.split)
        (OUT / f"candidates_{args.split}.json").write_text(json.dumps(
            {"label": explore.LABEL, "split": args.split, "n": len(c),
             "utc": time.strftime("%FT%TZ", time.gmtime()), "wells": c}, indent=1))
        print(f"{args.split}: {len(c)} candidate wells")
    elif args.cmd == "embed":
        embed(args.split, args.streams)
    elif args.cmd == "cache":
        print(json.dumps(build(args.base), indent=1))
    elif args.cmd == "eval":
        patch_dirs()
        ev = script("61_eval_h1_a2.py")
        suffix = args.base[len("vjepa2"):] + "_e3"
        sys.argv = ["61_eval_h1_a2.py", "--split", args.split, "--suffix", suffix]
        if args.freeze_commit:
            sys.argv += ["--freeze-commit", args.freeze_commit]
        rc = ev.main()
        f = OUT / "h1" / f"h1_{args.split}_plate_vjepa2.json"
        f.write_text(json.dumps({"label": explore.LABEL, "e3_cache": f"vjepa2{suffix}",
                                 **json.loads(f.read_text())}, indent=1, default=float))
        return rc
    else:
        (OUT / "compare.json").write_text(json.dumps(compare(), indent=1))
        print("written results/a2/exploratory/e3/compare.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
