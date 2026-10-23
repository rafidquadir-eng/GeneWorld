"""Test path (completion plan v4 S17): allowed-paths check, refusal conditions, rehearsal
manifest and cache exclusion, embed partition and S3 tables. Synthetic data only."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import cache  # noqa: E402


def script(name):
    spec = importlib.util.spec_from_file_location(name[:-3].replace("-", "_"),
                                                  REPO / "scripts" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EC = script("79_embed_commit.py")


# ---------------------------------------------------------------- allowed paths
@pytest.mark.parametrize("split,path,ok", [
    ("test", "manifests/cache_shards.tsv", True),
    ("test", "results/a2/embed_failures_test.json", True),
    ("test", "results/a2/missing_frames_test.json", True),
    ("test", "logs/test_embed/embed_gpu0.log", True),
    ("test", "results/a2/h1/h1_test_plate_vjepa2.json", False),
    ("test", "results/a2/embed_failures_screenB.json", False),
    ("test", "manifests/cache_shards_rehearsal.tsv", False),
    ("test", "scripts/61_eval_h1_a2.py", False),
    ("screenB", "results/a2/missing_frames_screenB.json", True),
    ("screenB", "results/a2/gates_test.json", False),
    ("rehearsal", "manifests/cache_shards_rehearsal.tsv", True),
    ("rehearsal", "results/a2/rehearsal/embed_failures_rehearsal.json", True),
    ("rehearsal", "manifests/cache_shards.tsv", False),
    ("rehearsal", "results/a2/rehearsal/h1/h1_test_plate_vjepa2.json", False),
])
def test_allowed_paths(split, path, ok):
    assert (EC.offending([path], split) == []) is ok


def sh(cwd, *a):
    r = subprocess.run(a, cwd=cwd, check=True, capture_output=True, text=True)
    return r.stdout.strip()


@pytest.fixture
def synthetic_git(tmp_path, monkeypatch):
    origin, work = tmp_path / "synthetic_origin.git", tmp_path / "synthetic_work"
    sh(tmp_path, "git", "init", "-q", "--bare", str(origin))
    sh(tmp_path, "git", "init", "-q", "-b", "stage1", str(work))
    for k, v in (("user.name", "synthetic"), ("user.email", "synthetic@example.invalid")):
        sh(work, "git", "config", k, v)
    sh(work, "git", "remote", "add", "origin", str(origin))
    (work / "manifests").mkdir()
    (work / "manifests" / "cache_shards.tsv").write_text("backbone\tpath\tsha256\n")
    sh(work, "git", "add", "-A")
    sh(work, "git", "commit", "-qm", "synthetic freeze")
    sh(work, "git", "push", "-q", "origin", "stage1")
    monkeypatch.setattr(EC, "REPO", work)
    return work


def commit(work, files: dict, push=True):
    for f, text in files.items():
        (work / f).parent.mkdir(parents=True, exist_ok=True)
        (work / f).write_text(text)
    sh(work, "git", "add", "-A")
    sh(work, "git", "commit", "-qm", "synthetic embed")
    if push:
        sh(work, "git", "push", "-q", "origin", "stage1")
    return sh(work, "git", "rev-parse", "HEAD")


def test_check_passes_for_allowed_embed_commit(synthetic_git):
    base = sh(synthetic_git, "git", "rev-parse", "HEAD")
    head = commit(synthetic_git, {"manifests/cache_shards.tsv": "x\n",
                                  "results/a2/embed_failures_test.json": "[]",
                                  "logs/test_embed/a.log": "ok"})
    assert EC.check("test", base, head) == []


def test_check_refuses_evaluation_output(synthetic_git):
    base = sh(synthetic_git, "git", "rev-parse", "HEAD")
    head = commit(synthetic_git, {"results/a2/h1/h1_test_plate_vjepa2.json": "{}"})
    assert any("outside the allowed paths" in p for p in EC.check("test", base, head))


def test_check_refuses_unpushed_and_non_ancestor(synthetic_git):
    base = sh(synthetic_git, "git", "rev-parse", "HEAD")
    head = commit(synthetic_git, {"results/a2/embed_failures_test.json": "[]"}, push=False)
    assert any("not pushed" in p for p in EC.check("test", base, head))
    assert any("not an ancestor" in p for p in EC.check("test", head, base))


# ---------------------------------------------------------------- refusal conditions
@pytest.fixture
def synthetic_root(tmp_path):
    (tmp_path / "splits").mkdir()
    pd.DataFrame({"plate": ["synthetic_V1", "synthetic_T1"], "series": "S",
                  "split": ["val", "test"]}).to_csv(tmp_path / "splits/plate_split.tsv",
                                                    sep="\t", index=False)
    (tmp_path / "results/a2").mkdir(parents=True)
    (tmp_path / "results/a2/selection.json").write_text("{}")
    return tmp_path


def test_rehearsal_preflight(synthetic_root):
    assert EC.preflight("rehearsal", ["synthetic_V1"], synthetic_root) == []
    assert EC.preflight("rehearsal", ["synthetic_T1"], synthetic_root)       # not validation
    assert EC.preflight("rehearsal", [], synthetic_root)
    shard = synthetic_root / "data/latents_test/vjepa2/synthetic_T1.npz"
    shard.parent.mkdir(parents=True)
    np.savez(shard, x=np.zeros(1))
    assert any("data/latents_test" in p
               for p in EC.preflight("rehearsal", ["synthetic_V1"], synthetic_root))


def test_test_preflight(synthetic_root, monkeypatch):
    monkeypatch.delenv("GENEWORLD_REHEARSAL", raising=False)
    assert any("gates" in p for p in EC.preflight("test", [], synthetic_root))
    (synthetic_root / "results/a2/gates_val.json").write_text("{}")
    assert EC.preflight("test", [], synthetic_root) == []
    monkeypatch.setenv("GENEWORLD_REHEARSAL", "1")
    assert any("rehearsal" in p for p in EC.preflight("test", [], synthetic_root))
    monkeypatch.delenv("GENEWORLD_REHEARSAL")
    (synthetic_root / "results/a2/h1").mkdir()
    (synthetic_root / "results/a2/h1/h1_test_plate_vjepa2.json").write_text("{}")
    assert any("evaluated once" in p for p in EC.preflight("test", [], synthetic_root))


# ---------------------------------------------------------------- guard and exclusion
def test_guard_rehearsal_manifest(tmp_path, monkeypatch):
    data = tmp_path / "data"
    real = data / "latents" / "vjepa2" / "synthetic_P.npz"
    reh = data / "latents_rehearsal" / "plates" / "vjepa2" / "synthetic_P.npz"
    for f in (real, reh):
        f.parent.mkdir(parents=True)
        np.savez(f, x=np.zeros(1))
    man, rman = tmp_path / "synthetic_m.tsv", tmp_path / "synthetic_rm.tsv"
    man.write_text(f"backbone\tpath\tsha256\nvjepa2\t{real.relative_to(tmp_path)}\t0\n")
    rman.write_text(f"backbone\tpath\tsha256\nvjepa2\t{reh.relative_to(tmp_path)}\t0\n")
    monkeypatch.setattr(cache, "DATA", data)
    monkeypatch.setattr(cache, "SHARD_MANIFEST", man)
    monkeypatch.setattr(cache, "REHEARSAL_MANIFEST", rman)
    monkeypatch.delenv("GENEWORLD_REHEARSAL", raising=False)
    cache.check_shard_manifest()                     # rehearsal dir ignored outside rehearsal
    monkeypatch.setenv("GENEWORLD_REHEARSAL", "1")
    cache.check_shard_manifest()                     # listed in the rehearsal manifest
    rman.write_text("backbone\tpath\tsha256\n")
    with pytest.raises(cache.ShardManifestError, match="synthetic_P"):
        cache.check_shard_manifest()


def _shard(path: Path, wells: list[str], value: float, D=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(wells)
    rng = np.random.default_rng(0)
    lat = value + 0.01 * rng.standard_normal((n, 14, 4, 16, D))
    np.savez(path, latents=lat.astype(np.float16),
             wells=np.array(wells), fingerprint=np.zeros((n, 3), np.float32))


def test_build_exclude_uses_rehearsal_shard(tmp_path, monkeypatch):
    wells = [f"A{i}" for i in range(1, 8)]
    _shard(tmp_path / "data/latents/vjepa2/synthetic_P.npz", wells, 1.0)
    _shard(tmp_path / "data/latents_rehearsal/plates/vjepa2/synthetic_P.npz", wells, 2.0)
    (tmp_path / "splits").mkdir()
    pd.DataFrame({"Plate": "synthetic_P", "Well": wells, "Gene Symbol": "G",
                  "siRNA Identifier": "s", "control": ["negative control"] * 5 + ["", ""],
                  "series": "S", "plate_split": "train", "gene_split": "train"}).to_csv(
        tmp_path / "splits/synthetic_wells.tsv", sep="\t", index=False)
    man = tmp_path / "synthetic_m.tsv"
    man.write_text("backbone\tpath\tsha256\n" + "".join(
        f"vjepa2\t{p}\t0\n" for p in ("data/latents/vjepa2/synthetic_P.npz",
                                      "data/latents_rehearsal/plates/vjepa2/synthetic_P.npz")))
    for k, v in (("REPO", tmp_path), ("CACHE", tmp_path / "cache"),
                 ("DATA", tmp_path / "data"), ("SHARD_MANIFEST", man),
                 ("REHEARSAL_MANIFEST", tmp_path / "none.tsv")):
        monkeypatch.setattr(cache, k, v)
    monkeypatch.setenv("GENEWORLD_REHEARSAL", "1")
    roots = ["data/latents", "data/latents_rehearsal/plates"]
    tsvs = ["splits/synthetic_wells.tsv"]
    a = cache.build("vjepa2", roots=roots, tsvs=tsvs, suffix="_a")
    b = cache.build("vjepa2", roots=roots, tsvs=tsvs, suffix="_b",
                    exclude={"data/latents/vjepa2/synthetic_P.npz"})
    assert abs(np.load(a / "raw.npy").mean() - 1.0) < 0.01   # first shard wins by default
    assert abs(np.load(b / "raw.npy").mean() - 2.0) < 0.01   # the rehearsal shard replaces it


# ---------------------------------------------------------------- partition and S3 tables
def test_partition_balances_plates():
    pe = script("78_parallel_embed.py")
    w = pd.DataFrame({"Plate": sum(([f"synthetic_P{i}"] * (10 + i) for i in range(16)), [])})
    parts = pe.partition(w, 8)
    assert sorted(p for part in parts for p in part) == sorted(w.Plate.unique())
    loads = [int(w.Plate.isin(part).sum()) for part in parts]
    assert max(loads) - min(loads) <= 15


def test_fisher_tables_and_bh():
    s3 = script("76_s3_accounting.py")
    lost = np.array([True, True, False, False, False, False])
    cls = [["mitotic_delay"], [], ["mitotic_delay"], [], ["polylobed"], []]
    t = s3.fisher_tables(lost, cls)
    assert t["mitotic_delay"]["table_lost_embedded_x_class_not"] == [[1, 1], [1, 3]]
    assert set(t) == {"mitotic_delay", "polylobed", "binuclear", "any_reproducible"}
    assert all(0 <= r["q_bh"] <= 1 for r in t.values())
    np.testing.assert_allclose(s3.bh([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
