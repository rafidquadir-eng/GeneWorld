"""Provenance for every GeneWorld result (TiJepa rule: labels come from records).

A result records which frozen backbone produced the latents (name, source, pinned revision or
checkpoint SHA-256) and which predictor checkpoint (file, step, SHA-256, training split
manifest hash) produced the prediction. Every label shown in a figure or manuscript is derived
from these records by ``label_for``, so a result can never be shown under another model's name.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

#: frozen backbones: key -> (display label, source)
BACKBONES = {
    "vjepa2_vitl": ("V-JEPA 2 ViT-L", "huggingface:facebook/vjepa2-vitl-fpc64-256"),
    "dinov2_vits14": ("DINOv2 ViT-S/14", "timm:vit_small_patch14_dinov2.lvd142m"),
    "tijepa1": ("TiJepa-1", "TiJepa checkpoint tijepa1_T1_seed0.pt"),
}
#: non-learned baselines: key -> label
BASELINES = {
    "copy_last": "copy-last", "slot_mean": "time-slot mean", "control_mean": "control mean",
    "action_agnostic": "action-agnostic predictor", "knn_gene": "nearest-gene lookup",
    "endpoint_retrieval": "endpoint retrieval", "shuffled_action": "shuffled action",
}


class ProvenanceError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def backbone_record(key: str, revision: str | None = None, sha256: str | None = None) -> dict:
    if key not in BACKBONES:
        raise ProvenanceError(f"unknown backbone {key!r}")
    return {"kind": "backbone", "key": key, "source": BACKBONES[key][1],
            "revision": revision, "sha256": sha256}


def predictor_record(ckpt: Path, step: int, split_manifest_sha256: str,
                     backbone: dict, action_source: str) -> dict:
    return {"kind": "predictor", "file": Path(ckpt).name, "step": int(step),
            "sha256": sha256_file(ckpt), "split_manifest_sha256": split_manifest_sha256,
            "backbone": backbone, "action_source": action_source}


def label_for(rec: dict | None, key: str | None = None) -> str:
    """Display label derived from a record (or a baseline key without a record)."""
    if rec is None:
        if key in BASELINES:
            return BASELINES[key]
        raise ProvenanceError(f"{key!r} has no record and is not a baseline")
    if rec.get("kind") == "backbone":
        return BACKBONES[rec["key"]][0]
    if rec.get("kind") == "predictor":
        return (f"{BACKBONES[rec['backbone']['key']][0]} + predictor "
                f"({rec['action_source']} actions)")
    raise ProvenanceError(f"unrecognised record {rec!r}")


def check_labels(labels: dict) -> list[str]:
    """`labels`: shown label -> {"record": ..., "key": ...}; returns mismatches."""
    bad = []
    for shown, entry in labels.items():
        try:
            derived = label_for(entry.get("record"), entry.get("key"))
        except ProvenanceError as exc:
            bad.append(f"{shown!r}: {exc}")
            continue
        if derived != shown:
            bad.append(f"{shown!r} shown but its record says {derived!r}")
    return bad


#: manifests/models.yaml key and local weights file for each backbone
_WEIGHTS = {
    "vjepa2_vitl": ("vjepa2_vitl", "hf:facebook/vjepa2-vitl-fpc64-256", "model.safetensors"),
    "dinov2_vits14": ("dinov2_vits14", "hf:timm/vit_small_patch14_dinov2.lvd142m",
                      "model.safetensors"),
    "tijepa1": ("tijepa1", "file:checkpoints/backbones/tijepa1_T1_seed0.pt", None),
}
_VERIFIED: dict[str, dict] = {}


def _weights_path(key: str) -> Path:
    from geneworld.data import REPO
    _, where, fname = _WEIGHTS[key]
    kind, ref = where.split(":", 1)
    if kind == "file":
        return REPO / ref
    snap = (Path.home() / ".cache" / "huggingface" / "hub" /
            ("models--" + ref.replace("/", "--")) / "snapshots")
    m = manifest()[_WEIGHTS[key][0]]
    return snap / m["revision"] / fname


def manifest() -> dict:
    import yaml
    from geneworld.data import REPO
    return yaml.safe_load((REPO / "manifests" / "models.yaml").read_text())


def verified_backbone_record(key: str) -> dict:
    """Backbone record with the pinned revision and the weights SHA-256, verified on disk
    against manifests/models.yaml (amendment 02 section 9). Mismatch is a hard error."""
    if key in _VERIFIED:
        return _VERIFIED[key]
    m = manifest()[_WEIGHTS[key][0]]
    path = _weights_path(key)
    got = sha256_file(path)
    if got != m["file_sha256"]:
        raise ProvenanceError(f"{key}: weights {path} hash {got} != manifest {m['file_sha256']}")
    rec = backbone_record(key, revision=m.get("revision"), sha256=got)
    _VERIFIED[key] = rec
    return rec
