"""Run amendment-02 training jobs on every GPU, scheduled by GPU memory.

Each GPU has `--units` memory units (4 per 40 GB GPU; 22 M = 1, 50 M = 2); each job runs with
CUDA_VISIBLE_DEVICES set to one GPU, placed on the GPU with the most free units.

  --phase grid       every family x 8 configurations x 3 seeds (per-plate normalisation)
  --phase secondary  selected configurations: global preprocessing (all families), DINOv2 and
                     TiJepa-1 (h1 families), complex split (planning source true / shuffled,
                     h1_none); needs results/a2/selection.json and, for the complex split,
                     results/a2/h2/h2_val_plate_vjepa2.json (planning source)
A job whose results JSON exists is skipped (resumable; completed runs are never rerun or
overwritten). Each job logs to logs/a2/<run>.log. `--check-inputs` only loads the inputs of
every pending job's family (scripts/50_train_a2.py --check-inputs) and exits.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.train import a2  # noqa: E402

PY = str(Path.home() / "venv" / "bin" / "python")


def jobs(phase: str) -> list[dict]:
    P = a2.protocol()
    fams = P["fixed_before_any_amended_model_2026-10-20"]["families"]
    seeds = P["grid"]["seeds"]
    out = []
    if phase == "grid":
        # interleave families so partial progress covers every family
        for s in seeds:
            for cfg in a2.configs():
                for f in fams:
                    out.append(dict(family=f, config=cfg["id"], seed=s, norm="plate",
                                    backbone="vjepa2", gene_split="gene"))
        return out
    sel = json.loads((a2.RES_DIR / "selection.json").read_text())["selected"]
    for s in seeds:
        for f in fams:
            out.append(dict(family=f, config=sel[f], seed=s, norm="global", backbone="vjepa2",
                            gene_split="gene"))
        for bb in ("dinov2", "tijepa1"):
            for f in ("h1_learned", "h1_none"):
                out.append(dict(family=f, config=sel[f], seed=s, norm="plate", backbone=bb,
                                gene_split="gene"))
    h2 = a2.RES_DIR / "h2" / "h2_val_plate_vjepa2.json"
    if h2.exists():
        src = json.loads(h2.read_text())["planning_source"]
        for s in seeds:
            for f in (f"h2:{src}", f"h2shuf:{src}", "h1_none"):
                out.append(dict(family=f, config=sel[f], seed=s, norm="plate", backbone="vjepa2",
                                gene_split="complex"))
    return out


def units_of(name: str) -> int:
    """GPU memory units: 22 M ~8.5 GB peak, 50 M ~14.6 GB peak (measured), capacity 4 = 40 GB."""
    return 2 if "_50M_" in name else 1


def gpu_ids() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.split()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    return [g for g in out if vis is None or g in vis.split(",")]


def external_live(mine: set[str]) -> dict[str, str | None]:
    """Runs holding a live lock that this runner did not start (e.g. after a restart), with
    the GPU recorded in the lock (None if the lock predates per-GPU slots)."""
    out = {}
    for lk in a2.CK_DIR.glob("*.lock"):
        try:
            info = json.loads(lk.read_text())
        except (OSError, ValueError):
            continue
        if lk.stem not in mine and Path(f"/proc/{info['pid']}").exists():
            out[lk.stem] = info.get("gpu")
    return out


def train_cmd(j: dict) -> list[str]:
    return [PY, str(REPO / "scripts" / "50_train_a2.py"), "--family", j["family"],
            "--config", j["config"], "--seed", str(j["seed"]), "--norm", j["norm"],
            "--backbone", j["backbone"], "--gene-split", j["gene_split"]]


def check_inputs(todo: list[tuple[str, dict]], gpus: list[str]) -> int:
    """One input check per pending (family, norm, backbone, gene split), over the GPUs."""
    combos = {}
    for name, j in todo:
        combos.setdefault((j["family"], j["norm"], j["backbone"], j["gene_split"]), (name, j))

    def run(k_item):
        k, (name, j) = k_item
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[k % len(gpus)])
        r = subprocess.run(train_cmd(j) + ["--check-inputs"], cwd=REPO, env=env,
                           capture_output=True, text=True)
        return name, r.returncode, (r.stdout + r.stderr).strip().splitlines()[-3:]

    with ThreadPoolExecutor(len(gpus)) as ex:
        res = list(ex.map(run, enumerate(combos.values())))
    bad = [r for r in res if r[1] != 0]
    for name, rc, tail in res:
        print(("ok   " if rc == 0 else "FAIL ") + name)
        for t in (tail if rc else tail[-1:]):
            print("       " + t)
    print(f"input check: {len(res) - len(bad)} of {len(res)} family inputs ok", flush=True)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["grid", "secondary"], required=True)
    ap.add_argument("--workers", type=int, default=3, help="unused; see --units")
    ap.add_argument("--units", type=int, default=4,
                    help="GPU memory units per GPU (40 GB: 4; 22M = 1, 50M = 2)")
    ap.add_argument("--check-inputs", action="store_true")
    args = ap.parse_args()
    gpus = gpu_ids()
    todo = []
    for j in jobs(args.phase):
        name = a2.run_name(j["norm"], j["backbone"], j["family"], j["config"], j["seed"],
                           j["gene_split"])
        if not (a2.RES_DIR / "train" / f"{name}.json").exists():
            todo.append((name, j))
    print(f"{args.phase}: {len(todo)} jobs to run; GPUs {','.join(gpus)}, "
          f"{args.units} memory units each", flush=True)
    if args.check_inputs:
        return check_inputs(todo, gpus)
    log = REPO / "logs" / "a2"
    log.mkdir(parents=True, exist_ok=True)
    running: list[tuple[str, subprocess.Popen, float, int, str]] = []
    failed = []
    while todo or running or external_live(set()):
        used = {g: 0 for g in gpus}
        for *_, u, g in running:
            used[g] += u
        mine = {n for n, *_ in running}
        ext = external_live(mine)
        for n, g in ext.items():
            for gg in ([g] if g in used else gpus):     # unknown GPU: count it on every GPU
                used[gg] += units_of(n)
        for name, j in list(todo):
            u = units_of(name)
            if name in ext:
                continue
            g = min(gpus, key=lambda x: (used[x], int(x)))
            if used[g] + u > args.units:
                continue
            todo.remove((name, j))
            fh = open(log / f"{name}.log", "w")
            env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                       CUDA_VISIBLE_DEVICES=g)
            running.append((name, subprocess.Popen(train_cmd(j), stdout=fh,
                                                   stderr=subprocess.STDOUT, cwd=REPO,
                                                   env=env), time.time(), u, g))
            used[g] += u
        time.sleep(10)
        still = []
        for name, p, t0, u, g in running:
            rc = p.poll()
            if rc is None:
                still.append((name, p, t0, u, g))
            else:
                ok = rc == 0 and (a2.RES_DIR / "train" / f"{name}.json").exists()
                if not ok:
                    failed.append(name)
                print(f"{time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED'} {name} "
                      f"(GPU {g}, {(time.time() - t0) / 60:.1f} min); {len(todo)} queued",
                      flush=True)
        running = still
        if not todo and not running:
            if not external_live(set()):
                break
    print(f"{args.phase} finished; {len(failed)} failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
