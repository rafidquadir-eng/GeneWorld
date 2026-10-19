"""Stream MitoCheck wells from EBI S3 and embed them with frozen backbones (no raw storage).

For each well in the selection (default: `splits/stage1_wells.tsv`):
  1. fetch the 93 level-1 frames (512 x 672) into memory (EBI S3, 128 parallel GETs);
  2. scale intensities per well to [0, 1] by the 0.5th and 99.8th percentiles of the whole
     movie (one scaling for all frames, so temporal changes are preserved; plate-to-plate
     exposure differences are removed);
  3. cut four 256 x 256 tiles from the central 512 x 512 region and 14 clips of 16 frames
     starting at frames 0, 6, ..., 72 and 77 (8 h clips, one every 3 h);
  4. embed every (clip, tile) with
       * V-JEPA 2 ViT-L (frozen; pinned revision; 16 x 256 x 256 input, ImageNet
         normalisation, gray replicated to RGB),
       * DINOv2 ViT-S/14 (frozen; each of the 16 frames at 224 x 224),
       * TiJepa-1 (frozen target encoder; 16 x 96 x 96 single-channel input);
     and keep, per (clip, tile), the patch tokens averaged over time and pooled to a 4 x 4
     spatial grid (16 tokens);
  5. drop the raw frames.
Per plate, one shard `data/latents/<backbone>/<plate>.npz` holds latents (wells, 14, 4, 16, D)
in float16, the well list and the per-frame mean intensity (QC). Each shard is backed up to the
persistent filesystem with SHA-256 when written. Re-running resumes: finished plates are skipped.
"""
from __future__ import annotations

import argparse
import functools
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

print = functools.partial(print, flush=True)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import backup  # noqa: E402
from geneworld.io import mitocheck as MC  # noqa: E402

OUT = REPO / "data" / "latents"
CLIP_STARTS = list(range(0, 73, 6)) + [77]
NF, TILE = 16, 256
VJ_REPO, VJ_REV = "facebook/vjepa2-vitl-fpc64-256", "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
TIJEPA_CKPT = REPO / "checkpoints" / "backbones" / "tijepa1_T1_seed0.pt"
DEV = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 1, 3, 1, 1)


def tiles_and_clips(frames: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """(93, 512, 672) uint8 -> (14*4, 16, 256, 256) float on the GPU in [0, 1]; frame means."""
    f = torch.from_numpy(frames).to(DEV, non_blocking=True).float()
    sub = f[::4, ::4, ::4].flatten()
    lo, hi = torch.quantile(sub, torch.tensor([0.005, 0.998], device=DEV))
    f = ((f - lo) / torch.clamp(hi - lo, min=1.0)).clamp_(0, 1)
    x0 = (f.shape[2] - 512) // 2
    f = f[:, :, x0:x0 + 512]
    tiles = [f[:, :TILE, :TILE], f[:, :TILE, TILE:], f[:, TILE:, :TILE], f[:, TILE:, TILE:]]
    clips = torch.stack([t[s:s + NF] for s in CLIP_STARTS for t in tiles])  # (56, 16, 256, 256)
    return clips, frames.reshape(len(frames), -1).mean(1)


def pool_grid(tok: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
    """(B, t*h*w, D) -> time-mean, 4 x 4 adaptive average pool -> (B, 16, D)."""
    b, _, d = tok.shape
    g = tok.reshape(b, t, h, w, d).mean(1).permute(0, 3, 1, 2)
    g = torch.nn.functional.adaptive_avg_pool2d(g.float(), 4)
    return g.flatten(2).transpose(1, 2)


class Backbones:
    def __init__(self, which: list[str]):
        self.which = which
        if "vjepa2" in which:
            from transformers import VJEPA2Model
            self.vj = VJEPA2Model.from_pretrained(VJ_REPO, revision=VJ_REV, dtype=torch.bfloat16,
                                                  attn_implementation="sdpa").eval().to(DEV)
        if "dinov2" in which:
            import timm
            self.dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                                          num_classes=0, img_size=224).eval().to(DEV)
        if "tijepa1" in which:
            from geneworld.models.video_jepa import VideoJEPA
            sd = torch.load(TIJEPA_CKPT, map_location="cpu", weights_only=False)
            m = VideoJEPA(clip=(16, 96, 96))
            m.load_state_dict(sd["model"])
            self.tj = m.eval().to(DEV)

    @torch.no_grad()
    def __call__(self, clips: torch.Tensor) -> dict[str, np.ndarray]:
        out = {}
        x = clips.to(DEV, non_blocking=True)                       # (B, 16, 256, 256)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if "vjepa2" in self.which:
                v = ((x[:, :, None].expand(-1, -1, 3, -1, -1) - MEAN) / STD)
                toks = []
                for i in range(0, len(v), 32):
                    toks.append(self.vj(pixel_values_videos=v[i:i + 32]).last_hidden_state)
                out["vjepa2"] = pool_grid(torch.cat(toks), 8, 16, 16)
            if "dinov2" in self.which:
                fr = torch.nn.functional.interpolate(x.flatten(0, 1)[:, None], size=224,
                                                     mode="bilinear", align_corners=False)
                fr = (fr.expand(-1, 3, -1, -1) - MEAN[0]) / STD[0]
                toks = []
                for i in range(0, len(fr), 512):
                    t = self.dino.forward_features(fr[i:i + 512])[:, self.dino.num_prefix_tokens:]
                    toks.append(t)
                t = torch.cat(toks).reshape(x.shape[0], NF * 16 * 16, -1)
                out["dinov2"] = pool_grid(t, NF, 16, 16)
            if "tijepa1" in self.which:
                s = torch.nn.functional.interpolate(x, size=96, mode="bilinear",
                                                    align_corners=False)
                s = (s - s.mean(dim=(1, 2, 3), keepdim=True)) / (
                    s.std(dim=(1, 2, 3), keepdim=True) + 1e-6)
                t = self.tj.target(s[:, None])
                out["tijepa1"] = pool_grid(t, 8, 12, 12)
        return {k: v.half().cpu().numpy() for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", default=str(REPO / "splits" / "stage1_wells.tsv"))
    ap.add_argument("--backbones", default="vjepa2,dinov2,tijepa1")
    ap.add_argument("--fetchers", type=int, default=4, help="wells fetched concurrently")
    ap.add_argument("--streams", type=int, default=128)
    ap.add_argument("--max-plates", type=int, default=0)
    ap.add_argument("--inflight", type=int, default=16, help="max wells in memory (~32 MB each)")
    args = ap.parse_args()
    which = args.backbones.split(",")
    wells = pd.read_csv(args.wells, sep="\t")
    plates = list(dict.fromkeys(wells.Plate))
    done = [p for p in plates if all((OUT / b / f"{p}.npz").exists() for b in which)]
    todo = [p for p in plates if p not in done][: args.max_plates or None]
    print(f"{len(plates)} plates ({len(done)} done), {len(todo)} to embed; backbones {which}")
    bb = Backbones(which)
    pool = ThreadPoolExecutor(args.streams)
    q: queue.Queue = queue.Queue()
    # at most `inflight` wells fetched or waiting at any time (bounded rolling buffer)
    inflight = threading.BoundedSemaphore(args.inflight)

    def producer():
        jobs = [(p, r.Well, r.uuid) for p in todo for r in wells[wells.Plate == p].itertuples()]
        def fetch(j):
            try:
                return j, MC.read_well(j[2], j[1], 1, pool=pool), None
            except Exception as exc:  # recorded, not fatal
                return j, None, f"{type(exc).__name__}: {exc}"[:300]
        pending: queue.Queue = queue.Queue()
        with ThreadPoolExecutor(args.fetchers) as ex:
            def feed():
                for j in jobs:
                    inflight.acquire()
                    pending.put(ex.submit(fetch, j))
                pending.put(None)
            threading.Thread(target=feed, daemon=True).start()
            while (fut := pending.get()) is not None:
                q.put(fut.result())
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    store: dict[str, dict] = {}
    expected = {p: int((wells.Plate == p).sum()) for p in todo}
    failures: list[str] = []
    t0, n = time.time(), 0

    def flush(plate: str) -> None:
        s = store.pop(plate, None)
        if not s or not s["wells"]:
            return
        for b in which:
            d = OUT / b
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{plate}.npz"
            np.savez(path, latents=np.stack(s[b]), wells=np.array(s["wells"]),
                     frame_mean=np.stack(s["fmean"]), clip_starts=np.array(CLIP_STARTS))
            backup.backup(path)
        el = time.time() - t0
        print(f"{plate}: {len(s['wells'])} wells; {n} wells in {el / 60:.1f} min "
              f"({n / el:.2f} wells/s)")

    while True:
        item = q.get()
        if item is None:
            break
        job, frames, err = item
        inflight.release()
        plate, well, _ = job
        if err is not None:
            failures.append(f"{plate}:{well}: {err}")
            expected[plate] -= 1
        else:
            clips, fmean = tiles_and_clips(frames)
            emb = bb(clips)
            s = store.setdefault(plate, {"wells": [], "fmean": [], **{b: [] for b in which}})
            s["wells"].append(well)
            s["fmean"].append(fmean.astype(np.float32))
            for b in which:
                s[b].append(emb[b].reshape(len(CLIP_STARTS), 4, 16, -1))
            n += 1
        if plate in store and len(store[plate]["wells"]) >= expected[plate]:
            flush(plate)
    for plate in list(store):
        flush(plate)
    (REPO / "logs").mkdir(exist_ok=True)
    (REPO / "logs" / "embed_failures.json").write_text(json.dumps(failures, indent=1))
    print(f"done: {n} wells, {len(failures)} failures, {(time.time() - t0) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
