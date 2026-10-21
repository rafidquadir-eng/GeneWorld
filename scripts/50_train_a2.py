"""Train one amendment-02 model: one family x one grid configuration x one seed.

Fixed values come from `manifests/amend02_protocol.json` (rollout 6, lr 3e-4, batch 256 wells,
OneCycle over 12,000 steps, siRNA-token dropout 0.2). The checkpoint kept is the one with the
lowest validation primary error (24 h and 36 h, primary wells or primary genes), evaluated every
250 steps; training stops 3,000 steps after the last improvement. Seeds vary initialisation,
batch order, rollout-context draws and (shuffled families) the gene-to-encoding permutation.

Single writer: `checkpoints/a2/<run>.lock` (hostname, PID, start) is taken exclusively and the
run refuses to start if a live lock exists; checkpoints are written atomically (tmp + fsync +
os.replace) and backed up with SHA-256 when the run ends.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import backup, cache  # noqa: E402
from geneworld.data import BACKBONE_KEY, N_STEPS  # noqa: E402
from geneworld.eval.provenance import sha256_file, verified_backbone_record  # noqa: E402
from geneworld.models.predictor import ActionPredictor  # noqa: E402
from geneworld.train import a2  # noqa: E402
from geneworld.train.batcher import DEV, Batcher  # noqa: E402


def split_sha() -> str:
    h = hashlib.sha256()
    for f in ("gene_split.tsv", "gene_split_complex.tsv", "plate_split.tsv", "stage1_wells.tsv",
              "stage2_wells.tsv", "amend02_h1_primary_wells.tsv",
              "amend02_h2_primary_genes.tsv"):
        h.update((REPO / "splits" / f).read_bytes())
    return h.hexdigest()


def take_lock(path: Path) -> None:
    if path.exists():
        info = json.loads(path.read_text())
        alive = info["host"] == socket.gethostname() and Path(f"/proc/{info['pid']}").exists()
        if alive:
            sys.exit(f"{path} is held by live PID {info['pid']}; refusing to start")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.write(fd, json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                             "start": time.time()}).encode())
    os.close(fd)


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@torch.no_grad()
def primary_error(model, z, bat, idx, steps=a2.PRIMARY_STEPS, bs=256) -> float:
    model.eval()
    errs = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        a, c, s = bat.get(torch.tensor(j, device=DEV))
        zz = torch.from_numpy(z[j, :N_STEPS].astype(np.float32)).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.rollout(zz[:, :1], a, c, s, max(steps)).float()
        sel = [h - 1 for h in steps]
        errs.append(((pred[:, sel] - zz[:, steps]) ** 2).mean((1, 2, 3)).cpu())
    model.train()
    return float(torch.cat(errs).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--config", required=True, help="grid id, e.g. 22M_d0.1_w0.05")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--norm", default="plate", choices=["plate", "global"])
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--gene-split", default="gene", choices=["gene", "complex"])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--max-steps", type=int, default=0, help="override (smoke tests only)")
    args = ap.parse_args()
    P = a2.protocol()["fixed_not_tuned"]
    cfg = a2.config_by_id(args.config)
    max_steps = args.max_steps or P["max_steps"]
    name = a2.run_name(args.norm, args.backbone, args.family, args.config, args.seed,
                       args.gene_split)
    a2.CK_DIR.mkdir(parents=True, exist_ok=True)
    lock = a2.CK_DIR / f"{name}.lock"
    take_lock(lock)
    ck = a2.CK_DIR / f"{name}.pt"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    c = cache.load(args.backbone, args.norm, gene_split=args.gene_split, suffix=args.suffix)
    m = c.meta
    mode, src, shuf = a2.family_mode(args.family)
    enc = a2.family_encoding(args.family, args.seed)
    train = a2.train_mask(m, args.family, args.seed)
    vocab_list = sorted(set(m.action[train]) - {"NEG"})
    vocab = {g: i for i, g in enumerate(vocab_list)}
    sirna_list = sorted(set(m.sirna[train]) - {""})
    sirnas = {s: i + 1 for i, s in enumerate(sirna_list)}
    which = "h2_val" if mode == "encoding" else "h1_val"
    val = a2.eval_mask(m, which, set(vocab) | {"NEG"}, enc if mode == "encoding" else None)
    tr_idx = np.flatnonzero(train)
    va_idx = np.flatnonzero(val)
    print(f"{name}: train wells {len(tr_idx)}, val primary wells {len(va_idx)}, "
          f"actions {len(vocab)}")
    z = c.z                                                   # (N, 14, 4, D) float16, host
    bat = Batcher(c, mode, vocab, sirnas, enc)
    enc_dim = bat.action.shape[1] if mode == "encoding" else 0
    model = ActionPredictor(z.shape[-1], mode, n_actions=len(vocab), n_sirna=len(sirnas),
                            enc_dim=enc_dim, width=cfg["width"], depth=cfg["depth"],
                            drop=cfg["dropout"]).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=P["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, P["lr"], total_steps=max_steps,
                                                pct_start=0.05)
    zt = torch.from_numpy(z[:, :N_STEPS])
    best, best_step, curve, t0 = np.inf, 0, [], time.time()
    R, bs, ev, pat = P["rollout_steps"], P["batch_wells"], P["eval_every"], 3000
    for step in range(1, max_steps + 1):
        j = tr_idx[torch.randint(len(tr_idx), (bs,), generator=gen).numpy()]
        a, cc, s = bat.get(torch.tensor(j, device=DEV), sirna_drop=P["sirna_token_dropout"])
        zz = zt[j].to(DEV, non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(zz[:, :-1], a, cc, s)
            loss_tf = (pred.float() - zz[:, 1:]).abs().mean()
            ctx = int(rng.integers(1, 5))
            r = min(R, N_STEPS - ctx)
            roll = model.rollout(zz[:, :ctx], a, cc, s, r)
            loss_ro = (roll.float() - zz[:, ctx:ctx + r]).abs().mean()
            loss = loss_tf + loss_ro
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % ev == 0 or step == max_steps:
            score = primary_error(model, z, bat, va_idx)
            curve.append({"step": step, "val_primary": score, "loss_tf": float(loss_tf),
                          "loss_ro": float(loss_ro)})
            if score < best:
                best, best_step = score, step
                atomic_save({"model": model.state_dict(), "step": step, "val_primary": score,
                             "args": {**vars(args), "config": cfg["id"], **cfg,
                                      "mode": mode, "source": src, "shuffled": shuf},
                             "vocab": vocab_list, "sirnas": sirna_list,
                             "dim": int(z.shape[-1]), "enc_dim": enc_dim, "n_params": n_par,
                             "split_sha256": split_sha(),
                             "cache_shards_sha256": c.stats["shards_sha256"],
                             "train_wells_sha256": a2.train_wells_sha256(m, args.family,
                                                                         args.seed),
                             "n_train_wells": int(len(tr_idx)),
                             "design_commit": a2.DESIGN_COMMIT}, ck)
            if step - best_step >= pat:
                break
    d = torch.load(ck, map_location="cpu", weights_only=False)
    d["curve"] = curve
    d["stopped_at"] = step
    atomic_save(d, ck)
    if not a2._DEV:
        backup.backup(ck)
    rec = {"name": name, "family": args.family, "config": cfg["id"], "seed": args.seed,
           "norm": args.norm, "backbone_key": args.backbone, "gene_split": args.gene_split,
           "best_val_primary": best, "best_step": best_step, "stopped_at": step,
           "n_params": n_par, "n_train_wells": int(len(tr_idx)), "n_val_wells": int(len(va_idx)),
           "minutes": (time.time() - t0) / 60, "curve": curve, "checkpoint": ck.name,
           "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
           "checkpoint_sha256": sha256_file(ck),
           "backbone": verified_backbone_record(BACKBONE_KEY[args.backbone]),
           "design_commit": a2.DESIGN_COMMIT}
    out = a2.RES_DIR / "train"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1))
    lock.unlink(missing_ok=True)
    print(f"{name}: best {best:.4f} at {best_step} (stopped {step}); "
          f"{(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
