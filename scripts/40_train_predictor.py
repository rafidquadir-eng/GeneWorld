"""Train an action-conditioned latent predictor on cached latents (design A.6).

Teacher-forced next-clip L1 loss on all steps plus an autoregressive rollout L1 loss (context of
1–4 clips, `--rollout` steps), as in V-JEPA 2-AC. Selection only on validation wells: every
`--eval-every` steps the 12-step rollout error from clip 0 is measured on the validation set of
the run's hypothesis (h1_val for learned / none, h2_val for encoding), and the best checkpoint
is kept. The checkpoint stores its config, vocabularies, step, validation curve and the SHA-256
of the split files; it is backed up to the persistent filesystem.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import backup, sets  # noqa: E402
from geneworld.data import N_STEPS, load  # noqa: E402
from geneworld.models.predictor import ActionPredictor  # noqa: E402
from geneworld.train.batcher import Batcher  # noqa: E402

DEV = "cuda"
EVAL_H = [1, 4, 8, 12]


def split_sha() -> str:
    h = hashlib.sha256()
    for f in ("gene_split.tsv", "plate_split.tsv", "stage1_wells.tsv"):
        h.update((REPO / "splits" / f).read_bytes())
    return h.hexdigest()


@torch.no_grad()
def rollout_error(model, z, bat, idx, steps=N_STEPS - 1, bs=512):
    """Per-well MSE at each rollout step from clip 0: (n, steps)."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        a, c, s = bat.get(j)
        zz = z[j].float()
        pred = model.rollout(zz[:, :1], a, c, s, steps)
        out.append(((pred - zz[:, 1:1 + steps]) ** 2).mean((2, 3)).cpu())
    model.train()
    return torch.cat(out).numpy() if out else np.full((1, steps), np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vjepa2")
    ap.add_argument("--gene-split", choices=["gene", "complex"], default="gene")
    ap.add_argument("--mode", choices=["learned", "encoding", "none"], default="learned")
    ap.add_argument("--source", default="", help="encoding source for --mode encoding")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--rollout", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    lat = load(args.backbone, gene_split=args.gene_split)
    m = lat.meta
    enc = sets.encoding(args.source) if args.mode == "encoding" else {}
    train = m.train.to_numpy().copy()
    if args.mode == "encoding":          # genes without this encoding cannot be trained on
        train &= m.action.isin(set(enc) | {"NEG"}).to_numpy()
    vocab_list = sorted(set(m.action[train]) - {"NEG"})
    vocab = {a: i for i, a in enumerate(vocab_list)}
    sirna_list = sorted(set(m.sirna[train]) - {""})
    sirnas = {s: i + 1 for i, s in enumerate(sirna_list)}
    mk = sets.masks(m, set(vocab) | {"NEG"})
    val = mk["h2_val"] if args.mode == "encoding" else mk["h1_val"]
    if args.mode == "encoding":
        val &= m.action.isin(set(enc)).to_numpy()
    tr_idx = torch.tensor(np.flatnonzero(train), device=DEV)
    va_idx = torch.tensor(np.flatnonzero(val), device=DEV)
    print(f"{args.name}: backbone {args.backbone}, mode {args.mode} {args.source}; "
          f"train wells {len(tr_idx)}, val wells {len(va_idx)}, actions {len(vocab)}")

    z = torch.tensor(lat.z[:, :N_STEPS], device=DEV, dtype=torch.float16)
    bat = Batcher(lat, args.mode, vocab, sirnas, enc)
    enc_dim = bat.action.shape[1] if args.mode == "encoding" else 0
    model = ActionPredictor(z.shape[-1], args.mode, n_actions=len(vocab), n_sirna=len(sirnas),
                            enc_dim=enc_dim, width=args.width, depth=args.depth).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    ck_dir = REPO / "checkpoints" / "predictors"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck = ck_dir / f"{args.name}.pt"
    best, curve, t0 = np.inf, [], time.time()
    for step in range(1, args.steps + 1):
        j = tr_idx[torch.randint(len(tr_idx), (args.bs,), device=DEV)]
        a, c, s = bat.get(j, sirna_drop=0.2)
        zz = z[j].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(zz[:, :-1], a, c, s)
            loss_tf = (pred.float() - zz[:, 1:]).abs().mean()
            ctx = int(rng.integers(1, 5))
            r = min(args.rollout, N_STEPS - ctx)
            roll = model.rollout(zz[:, :ctx], a, c, s, r)
            loss_ro = (roll.float() - zz[:, ctx:ctx + r]).abs().mean()
            loss = loss_tf + loss_ro
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            err = rollout_error(model, z, bat, va_idx)
            score = float(err[:, [h - 1 for h in EVAL_H]].mean())
            curve.append({"step": step, "val": score, "loss_tf": float(loss_tf),
                          "loss_ro": float(loss_ro)})
            print(f"step {step}: tf {loss_tf.item():.4f} ro {loss_ro.item():.4f} "
                  f"val {score:.4f} ({(time.time() - t0) / 60:.1f} min)")
            if score < best:
                best = score
                torch.save({"model": model.state_dict(), "args": vars(args), "step": step,
                            "val": score, "vocab": vocab_list, "sirnas": sirna_list,
                            "dim": int(z.shape[-1]), "enc_dim": enc_dim, "n_params": n_par,
                            "split_sha256": split_sha(), "latent_mean": lat.mean,
                            "latent_std": lat.std, "n_train_wells": int(len(tr_idx))}, ck)
    d = torch.load(ck, weights_only=False)
    d["curve"] = curve
    torch.save(d, ck)
    backup.backup(ck)
    rec = {"name": args.name, "best_val": best, "best_step": d["step"], "n_params": n_par,
           "curve": curve}
    (REPO / "results" / "train").mkdir(parents=True, exist_ok=True)
    (REPO / "results" / "train" / f"{args.name}.json").write_text(json.dumps(rec, indent=1))
    print(f"best val {best:.4f} at step {d['step']}; {n_par / 1e6:.1f} M parameters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
