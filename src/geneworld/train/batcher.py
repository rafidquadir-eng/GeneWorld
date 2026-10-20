"""Action / reagent tensors for the wells of a Latents object."""
from __future__ import annotations

import numpy as np
import torch

DEV = "cuda"


class Batcher:
    """Action / reagent tensors for a set of wells."""

    def __init__(self, lat, mode, vocab, sirnas, enc):
        m = lat.meta
        self.is_ctrl = torch.tensor((m.action == "NEG").to_numpy(), device=DEV)
        self.sirna = torch.tensor([sirnas.get(s, 0) for s in m.sirna], device=DEV)
        if mode == "learned":
            self.action = torch.tensor([vocab.get(a, 0) for a in m.action], device=DEV)
        elif mode == "encoding":
            dim = len(next(iter(enc.values())))
            a = np.stack([enc.get(g, np.zeros(dim, np.float32)) for g in m.action])
            self.action = torch.tensor(a, device=DEV)
        else:
            self.action = torch.zeros(len(m), device=DEV)

    def get(self, idx, sirna_drop: float = 0.0):
        s = self.sirna[idx]
        if sirna_drop:
            s = torch.where(torch.rand(len(idx), device=DEV) < sirna_drop, 0, s)
        return self.action[idx], self.is_ctrl[idx], s


def from_checkpoint(path, lat):
    """(model, batcher, provenance record) for a saved predictor applied to `lat`."""
    from geneworld import sets
    from geneworld.data import BACKBONE_KEY
    from geneworld.eval.provenance import predictor_record, verified_backbone_record
    from geneworld.models.predictor import ActionPredictor
    d = torch.load(path, map_location="cpu", weights_only=False)
    a = d["args"]
    if a["backbone"] != lat.backbone:
        raise ValueError(f"{path} was trained on {a['backbone']} latents, not {lat.backbone}")
    if not (np.allclose(d["latent_mean"], lat.mean) and np.allclose(d["latent_std"], lat.std)):
        raise ValueError(f"{path}: latent standardisation differs from the loaded latents")
    enc = sets.encoding(a["source"]) if a["mode"] == "encoding" else {}
    vocab = {g: i for i, g in enumerate(d["vocab"])}
    sirnas = {s: i + 1 for i, s in enumerate(d["sirnas"])}
    model = ActionPredictor(d["dim"], a["mode"], n_actions=len(vocab), n_sirna=len(sirnas),
                            enc_dim=d["enc_dim"], width=a["width"], depth=a["depth"]).to(DEV)
    model.load_state_dict(d["model"])
    model.eval()
    bat = Batcher(lat, a["mode"], vocab, sirnas, enc)
    rec = predictor_record(path, d["step"], d["split_sha256"],
                           verified_backbone_record(BACKBONE_KEY[a["backbone"]]),
                           a["mode"] + (f":{a['source']}" if a["source"] else ""))
    return model, bat, rec, d
