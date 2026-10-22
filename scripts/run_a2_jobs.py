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


def units_of(name: str) -> int:
    """GPU memory units: 22 M ~8.5 GB peak, 50 M ~14.6 GB peak (measured), capacity 4 = 40 GB."""
    return 2 if "_50M_" in name else 1


def external_live(mine: set[str]) -> set[str]:
    """Runs holding a live lock that this runner did not start (e.g. after a restart)."""
    out = set()
    for lk in a2.CK_DIR.glob("*.lock"):
        try:
            info = json.loads(lk.read_text())
        except (OSError, ValueError):
            continue
        if lk.stem not in mine and Path(f"/proc/{info['pid']}").exists():
            out.add(lk.stem)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["grid", "secondary"], required=True)
    ap.add_argument("--workers", type=int, default=3, help="unused; see --units")
    ap.add_argument("--units", type=int, default=4, help="GPU memory units (22M = 1, 50M = 2)")
    args = ap.parse_args()
    todo = []
    for j in jobs(args.phase):
        name = a2.run_name(j["norm"], j["backbone"], j["family"], j["config"], j["seed"],
                           j["gene_split"])
        if not (a2.RES_DIR / "train" / f"{name}.json").exists():
            todo.append((name, j))
    log = REPO / "logs" / "a2"
    log.mkdir(parents=True, exist_ok=True)
    print(f"{args.phase}: {len(todo)} jobs to run; GPU memory units {args.units}", flush=True)
    running: list[tuple[str, subprocess.Popen, float, int]] = []
    failed = []
    while todo or running or external_live(set()):
        used = sum(u for *_, u in running)
        mine = {n for n, *_ in running}
        ext = external_live(mine)
        used += sum(units_of(n) for n in ext)
        for k, (name, j) in enumerate(list(todo)):
            u = units_of(name)
            if name in ext or used + u > args.units:
                continue
            todo.remove((name, j))
            cmd = [PY, str(REPO / "scripts" / "50_train_a2.py"), "--family", j["family"],
                   "--config", j["config"], "--seed", str(j["seed"]), "--norm", j["norm"],
                   "--backbone", j["backbone"], "--gene-split", j["gene_split"]]
            fh = open(log / f"{name}.log", "w")
            env = dict(__import__("os").environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            running.append((name, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                                   cwd=REPO, env=env), time.time(), u))
            used += u
        time.sleep(10)
        still = []
        for name, p, t0, u in running:
            rc = p.poll()
            if rc is None:
                still.append((name, p, t0, u))
            else:
                ok = rc == 0 and (a2.RES_DIR / "train" / f"{name}.json").exists()
                if not ok:
                    failed.append(name)
                print(f"{time.strftime('%H:%M:%S')} {'done' if ok else 'FAILED'} {name} "
                      f"({(time.time() - t0) / 60:.1f} min); {len(todo)} queued", flush=True)
        running = still
        if not todo and not running:
            if not external_live(set()):
                break
    print(f"{args.phase} finished; {len(failed)} failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
