"""Test-time embedding with one embedder per GPU (completion plan v4 S17, decision D2).

Partitions a wells list by plate over the GPUs (largest plates first, each to the least-loaded
GPU) and runs `scripts/20_stream_embed.py --fetcher native` once per GPU
(CUDA_VISIBLE_DEVICES), each with `--streams-total / n_gpus` BIA streams, so all fetchers
together use at most `--streams-total` (128) streams. Shards go to `--out-root`; each embedder
backs up its shards with SHA-256 as it writes them. Afterwards the per-GPU failure lists are
merged into `--failures`, and the plate -> GPU assignment, exit codes and times go to
`logs/<tag>/embed_record.json`. Exit status 1 if an embedder failed or a plate has neither a
shard nor a failure for each of its wells.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PY = str(Path.home() / "venv" / "bin" / "python")
BACKBONES = ("vjepa2", "dinov2", "tijepa1")


def partition(wells: pd.DataFrame, n: int) -> list[list[str]]:
    sizes = wells.groupby("Plate").size().sort_values(ascending=False, kind="stable")
    parts: list[list[str]] = [[] for _ in range(n)]
    load = [0] * n
    for plate, k in sizes.items():
        i = min(range(n), key=lambda g: (load[g], g))
        parts[i].append(plate)
        load[i] += int(k)
    return parts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--failures", required=True, help="merged failure list (JSON)")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--streams-total", type=int, default=128)
    args = ap.parse_args()
    gpus = args.gpus.split(",")
    streams = args.streams_total // len(gpus)
    assert streams * len(gpus) <= 128
    wells = pd.read_csv(REPO / args.wells, sep="\t")
    log = REPO / "logs" / args.tag
    log.mkdir(parents=True, exist_ok=True)
    rec = {"wells": args.wells, "n_wells": int(len(wells)),
           "n_plates": int(wells.Plate.nunique()),
           "out_root": args.out_root, "streams_per_gpu": streams,
           "streams_total": streams * len(gpus), "fetcher": "native", "gpus": {}}
    procs = []
    for g, plates in zip(gpus, partition(wells, len(gpus)), strict=True):
        if not plates:
            continue
        part = log / f"{args.tag}_part{g}.tsv"
        wells[wells.Plate.isin(plates)].to_csv(part, sep="\t", index=False)
        cmd = [PY, str(REPO / "scripts" / "20_stream_embed.py"), "--wells",
               str(part.relative_to(REPO)), "--fetcher", "native", "--streams", str(streams),
               "--out-root", args.out_root]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        fh = open(log / f"embed_gpu{g}.log", "w")
        procs.append((g, part, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh,
                                                stderr=subprocess.STDOUT), time.time()))
        rec["gpus"][g] = {"plates": plates, "n_wells": int(wells.Plate.isin(plates).sum()),
                          "start_utc": time.strftime("%FT%TZ", time.gmtime())}
        print(f"GPU {g}: {len(plates)} plates, {rec['gpus'][g]['n_wells']} wells", flush=True)
    failures, bad = [], []
    for g, part, p, t0 in procs:
        rc = p.wait()
        rec["gpus"][g].update(exit_code=rc, minutes=round((time.time() - t0) / 60, 1),
                              end_utc=time.strftime("%FT%TZ", time.gmtime()))
        f = REPO / "logs" / f"embed_failures_{part.stem}.json"
        failures += json.loads(f.read_text()) if f.exists() else []
        if rc != 0:
            bad.append(f"embedder on GPU {g} exited {rc}")
    # every well is either in a shard of every backbone or in the failure list
    failed = {tuple(x.split(":")[:2]) for x in failures}
    for plate, w in wells.groupby("Plate"):
        want = set(w.Well.astype(str))
        for b in BACKBONES:
            sh = REPO / args.out_root / b / f"{plate}.npz"
            got = set(np.load(sh)["wells"].astype(str)) if sh.exists() else set()
            miss = {x for x in want - got if (plate, x) not in failed}
            if miss:
                bad.append(f"{plate} {b}: {len(miss)} wells neither embedded nor failed")
    out = REPO / args.failures
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(failures), indent=1))
    rec.update(n_failures=len(failures), problems=bad)
    (log / "embed_record.json").write_text(json.dumps(rec, indent=1))
    print(f"{len(failures)} wells failed; problems: {bad or 'none'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
