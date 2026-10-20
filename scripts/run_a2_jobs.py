"""Run amendment-02 training jobs, `--workers` at a time on the one GPU.

  --phase grid       every family x 8 configurations x 3 seeds (per-plate normalisation)
  --phase secondary  selected configurations: global preprocessing (all families), DINOv2 and
                     TiJepa-1 (h1 families), complex split (planning source true / shuffled,
                     h1_none); needs results/a2/selection.json and, for the complex split,
                     results/a2/h2/h2_val_plate_vjepa2.json (planning source)
A job whose results JSON exists is skipped (resumable). Each job logs to logs/a2/<run>.log.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["grid", "secondary"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    todo = []
    for j in jobs(args.phase):
        name = a2.run_name(j["norm"], j["backbone"], j["family"], j["config"], j["seed"],
                           j["gene_split"])
        if not (a2.RES_DIR / "train" / f"{name}.json").exists():
            todo.append((name, j))
    log = REPO / "logs" / "a2"
    log.mkdir(parents=True, exist_ok=True)
    print(f"{args.phase}: {len(todo)} jobs to run", flush=True)
    running: list[tuple[str, subprocess.Popen, float]] = []
    failed = []
    while todo or running:
        while todo and len(running) < args.workers:
            name, j = todo.pop(0)
            cmd = [PY, str(REPO / "scripts" / "50_train_a2.py"), "--family", j["family"],
                   "--config", j["config"], "--seed", str(j["seed"]), "--norm", j["norm"],
                   "--backbone", j["backbone"], "--gene-split", j["gene_split"]]
            fh = open(log / f"{name}.log", "w")
            running.append((name, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                                   cwd=REPO), time.time()))
        time.sleep(10)
        still = []
        for name, p, t0 in running:
            rc = p.poll()
            if rc is None:
                still.append((name, p, t0))
            else:
                ok = rc == 0 and (a2.RES_DIR / "train" / f"{name}.json").exists()
                if not ok:
                    failed.append(name)
                print(f"{time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED'} {name} "
                      f"({(time.time() - t0) / 60:.1f} min); {len(todo)} queued", flush=True)
        running = still
    print(f"{args.phase} finished; {len(failed)} failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
