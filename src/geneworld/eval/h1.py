"""H1 forecasting metrics, baselines and controls on cached latents.

Every method forecasts clips 1..12 (3–36 h ahead) from clip 0 of a well. Error = mean squared
error in standardised latent space, averaged over tiles and dimensions.

Baselines (design A.8 / A.7(1)):
  copy_last       z_0 repeated
  slot_mean       mean latent of training wells at each clip (no action, no history)
  slot_drift      z_0 + (slot_mean_t - slot_mean_0)
  control_mean    mean of the same plate's other negative-control wells at clip t
  control_drift   z_0 + (control_mean_t - control_mean_0)
  action_agnostic the predictor trained with the action removed
"""
from __future__ import annotations

import numpy as np
import torch

DEV = "cuda"
HORIZONS = [1, 4, 8, 12]


def sign_flip_mc(d: np.ndarray, n: int = 100_000, seed: int = 0) -> dict:
    """One-sided Monte Carlo sign-flip test of mean(d) < 0 over groups (d = model - baseline)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"n_groups": int(len(d)), "mean": None, "p_one_sided": None}
    rng = np.random.default_rng(seed)
    obs = d.mean()
    flips = (rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(1)
    p = (1 + np.sum(flips <= obs)) / (n + 1)
    return {"n_groups": int(len(d)), "mean": float(obs), "p_one_sided": float(p),
            "frac_groups_better": float(np.mean(d < 0))}


def baselines(z: np.ndarray, meta, idx: np.ndarray, train: np.ndarray) -> dict[str, np.ndarray]:
    """Forecasts (n, 12, K, D) for each non-learned baseline on wells `idx`."""
    zt = z[train][:, :13]
    slot = zt.mean(0)                                         # (13, K, D) -> tile-mean
    slot = np.broadcast_to(slot.mean(1, keepdims=True), slot.shape)
    z0 = z[idx, :1]
    out = {"copy_last": np.repeat(z0, 12, 1),
           "slot_mean": np.broadcast_to(slot[1:13], (len(idx),) + slot[1:13].shape),
           "slot_drift": z0 + (slot[1:13] - slot[:1])[None]}
    neg = (meta.action == "NEG").to_numpy()
    plates = meta.plate.to_numpy()
    cm = np.full((len(idx), 13) + z.shape[2:], np.nan, np.float32)
    for p in np.unique(plates[idx]):
        pn = np.flatnonzero(neg & (plates == p))
        for r, i in enumerate(idx):
            if plates[i] != p:
                continue
            others = pn[pn != i]
            if len(others):
                cm[r] = z[others, :13].mean((0, 2), keepdims=False)[:, None, :]
    out["control_mean"] = cm[:, 1:13]
    out["control_drift"] = z0 + cm[:, 1:13] - cm[:, :1]
    return out


@torch.no_grad()
def model_forecast(model, z: np.ndarray, bat, idx: np.ndarray, ctx_from: np.ndarray | None = None,
                   bs: int = 512) -> np.ndarray:
    """(n, 12, K, D) rollout from clip 0; `ctx_from` substitutes another well's context."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        j = torch.tensor(idx[i:i + bs], device=DEV)
        src = j if ctx_from is None else torch.tensor(ctx_from[i:i + bs], device=DEV)
        a, c, s = bat.get(j)
        z0 = torch.tensor(z[src.cpu().numpy(), :1], device=DEV)
        out.append(model.rollout(z0, a, c, s, 12).float().cpu().numpy())
    return np.concatenate(out)


def errors(pred: np.ndarray, z: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(n, 12) MSE per well and horizon."""
    return ((pred - z[idx, 1:13]) ** 2).mean((2, 3))
