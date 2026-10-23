"""Shard-manifest guard: cache.build refuses stray or missing latent shards."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import cache  # noqa: E402


@pytest.fixture
def synthetic_repo(tmp_path, monkeypatch):
    """A synthetic data/ tree with one manifest shard per root, and its manifest."""
    data = tmp_path / "data"
    rows = []
    for root in ("latents", "latents_s2"):
        d = data / root / "synthetic_bb"
        d.mkdir(parents=True)
        np.savez(d / "synthetic_plate.npz", latents=np.zeros(1))
        rows.append(f"synthetic_bb\tdata/{root}/synthetic_bb/synthetic_plate.npz\t0")
    man = tmp_path / "synthetic_cache_shards.tsv"
    man.write_text("backbone\tpath\tsha256\n" + "\n".join(rows) + "\n")
    monkeypatch.setattr(cache, "DATA", data)
    monkeypatch.setattr(cache, "SHARD_MANIFEST", man)
    monkeypatch.setattr(cache, "CACHE", tmp_path / "synthetic_cache")
    return data


def test_manifest_matches(synthetic_repo):
    cache.check_shard_manifest()


def test_stray_shard_makes_build_fail(synthetic_repo):
    stray = synthetic_repo / "latents" / "synthetic_bb" / "synthetic_stray.npz"
    np.savez(stray, latents=np.zeros(1))
    with pytest.raises(cache.ShardManifestError, match="synthetic_stray.npz"):
        cache.build("synthetic_bb")
    assert not (cache.CACHE / "synthetic_bb").exists()      # refused before writing anything


def test_missing_shard_makes_build_fail(synthetic_repo):
    (synthetic_repo / "latents_s2" / "synthetic_bb" / "synthetic_plate.npz").unlink()
    with pytest.raises(cache.ShardManifestError, match="1 missing"):
        cache.build("synthetic_bb")


def test_unused_dir_excluded_by_name(synthetic_repo):
    d = synthetic_repo / "latents_unused_1019" / "synthetic_bb"
    d.mkdir(parents=True)
    np.savez(d / "synthetic_old.npz", latents=np.zeros(1))
    cache.check_shard_manifest()
