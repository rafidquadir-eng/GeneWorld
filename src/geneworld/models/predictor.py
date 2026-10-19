"""Action-conditioned latent predictor (design A.6; after V-JEPA 2-AC).

Input per well: latents z (B, T, K, D) for T clip steps and K = 4 tile tokens, an action token
and a reagent (siRNA) token. Tokens at step t attend to the two conditioning tokens and to all
tile tokens at steps <= t (block-causal); tile tokens carry a step embedding but no tile-position
embedding (fields of view are exchangeable). The output at step t is the predicted latent at
step t + 1, parameterised as z_t + delta.

Action modes:
  learned   a per-gene embedding (seen genes only; the H1 model / upper bound for H2)
  encoding  an MLP adapter of a frozen gene encoding from another modality (H2 / H3)
  none      a constant token (action-agnostic baseline); the reagent token is also blanked
Negative-control wells use a learned "control" action in every mode except none.
"""
from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, width: int, heads: int, mlp: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, int(width * mlp)), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(int(width * mlp), width))

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ActionPredictor(nn.Module):
    def __init__(self, dim: int, mode: str, n_actions: int = 0, n_sirna: int = 0,
                 enc_dim: int = 0, width: int = 512, depth: int = 6, heads: int = 8,
                 n_steps: int = 13, drop: float = 0.1):
        super().__init__()
        self.mode, self.n_steps = mode, n_steps
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
        self.step = nn.Parameter(torch.zeros(n_steps, width))
        nn.init.normal_(self.step, std=0.02)
        self.null = nn.Parameter(torch.zeros(width))          # action-agnostic token
        self.ctrl = nn.Parameter(torch.randn(width) * 0.02)   # negative-control action
        if mode == "learned":
            self.act = nn.Embedding(n_actions, width)
            nn.init.normal_(self.act.weight, std=0.02)
        elif mode == "encoding":
            self.adapter = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, width),
                                         nn.GELU(), nn.Linear(width, width))
        self.sirna = nn.Embedding(max(n_sirna, 1) + 1, width)  # index 0 = unknown reagent
        nn.init.normal_(self.sirna.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(width, heads, drop=drop) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def action_token(self, action, is_ctrl):
        b = is_ctrl.shape[0]
        if self.mode == "none":
            return self.null.expand(b, -1)
        a = self.act(action) if self.mode == "learned" else self.adapter(action)
        return torch.where(is_ctrl[:, None], self.ctrl.expand(b, -1), a)

    @staticmethod
    def mask(t: int, k: int, device) -> torch.Tensor:
        step = torch.arange(t, device=device).repeat_interleave(k)
        tile = step[:, None] >= step[None, :]                      # (tk, tk)
        m = torch.ones(2 + t * k, 2 + t * k, dtype=torch.bool, device=device)
        m[:2, :2] = True
        m[:2, 2:] = False
        m[2:, :2] = True
        m[2:, 2:] = tile
        return ~m                                                  # True = blocked

    def forward(self, z, action, is_ctrl, sirna):
        """z (B, T, K, D) -> predicted z at steps 1..T (B, T, K, D)."""
        b, t, k, _ = z.shape
        x = self.inp(z) + self.step[:t, None, :]
        x = x.reshape(b, t * k, -1)
        s = self.sirna(sirna if self.mode != "none" else torch.zeros_like(sirna))
        x = torch.cat([self.action_token(action, is_ctrl)[:, None], s[:, None], x], 1)
        m = self.mask(t, k, z.device)
        for blk in self.blocks:
            x = blk(x, m)
        d = self.out(self.norm(x[:, 2:])).reshape(b, t, k, -1)
        return z + d

    def rollout(self, z0, action, is_ctrl, sirna, steps: int):
        """Autoregressive forecast from context z0 (B, C, K, D) for `steps` more clips
        (differentiable; wrap in torch.no_grad() for evaluation)."""
        z = z0
        for _ in range(steps):
            nxt = self(z, action, is_ctrl, sirna)[:, -1:]
            z = torch.cat([z, nxt], 1)
        return z[:, z0.shape[1]:]
