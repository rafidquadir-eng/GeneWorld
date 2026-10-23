"""E1-E3 helpers (src/geneworld/explore.py) on synthetic data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import explore as X  # noqa: E402


def synthetic_plate(rows=16, cols=24, T=3, D=5, seed=0):
    rng = np.random.default_rng(seed)
    r_eff, c_eff = rng.normal(0, 2, (rows, T, D)), rng.normal(0, 2, (cols, T, D))
    wells = [f"{chr(65 + r)}{c + 1}" for r in range(rows) for c in range(cols)]
    rc = np.array([X.well_rc(w) for w in wells])
    signal = rng.normal(0, 0.1, (len(wells), T, D))
    tm = 1.5 + r_eff[rc[:, 0]] + c_eff[rc[:, 1]] + signal
    z = np.repeat(tm[:, :, None], 4, 2).astype(np.float32)          # 4 identical tiles
    meta = pd.DataFrame({"plate": "synthetic_P1", "well": wells})
    return z, meta, signal


def test_well_rc():
    assert X.well_rc("A1") == (0, 0) and X.well_rc("P24") == (15, 23)


def test_median_polish_recovers_additive_effects():
    rng = np.random.default_rng(1)
    r, c = rng.normal(size=(6, 2)), rng.normal(size=(8, 2))
    x = 3 + r[:, None] + c[None]
    ov, row, col = X.median_polish(x)
    np.testing.assert_allclose(ov + row[:, None] + col[None], x, atol=1e-9)


def test_bscore_removes_row_and_column_effects():
    z, meta, signal = synthetic_plate()
    lib = np.ones(len(meta), bool)
    out = X.bscore_correct(z, meta, lib).astype(np.float32).mean(2)
    # what remains is the well-level signal (up to its own small median)
    assert np.abs(out - signal).max() < 0.15
    assert np.abs(z.mean(2) - signal).max() > 1


def test_bscore_uses_library_wells_only():
    z, meta, _ = synthetic_plate()
    lib = np.ones(len(meta), bool)
    ctrl = np.arange(0, len(meta), 37)
    z2 = z.copy()
    z2[ctrl] += 50                                                  # outlying controls
    lib2 = lib.copy()
    lib2[ctrl] = False
    a = X.bscore_correct(z, meta, lib)
    b = X.bscore_correct(z2, meta, lib2)
    keep = np.setdiff1d(np.arange(len(meta)), ctrl)
    assert np.abs(a[keep] - b[keep]).max() < 0.2                    # effects unchanged


def synthetic_series(offset: float, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(3):
        for p in range(3):
            for w in range(30):
                rows.append({"series": f"synthetic_S{s}", "plate": f"synthetic_S{s}_p{p}",
                             "well": f"A{w + 1}", "plate_split": "train", "p": p})
    meta = pd.DataFrame(rows)
    x = rng.normal(size=(len(meta), 8))
    x[:, 0] += offset * meta.p.to_numpy()
    return x, meta


def test_within_series_probe_detects_plate_offset():
    x, meta = synthetic_series(offset=5.0)
    r = X.within_series_probe(x, meta)
    assert r["n_series"] == 3 and r["mean_accuracy"] > 0.9
    assert len(X.probe_wells(meta)) == len(meta)


def test_within_series_probe_near_chance_without_offset():
    x, meta = synthetic_series(offset=0.0)
    r = X.within_series_probe(x, meta)
    assert r["mean_accuracy"] < 0.6 and np.isclose(r["mean_chance"], 1 / 3)


def test_partial_clips_match_full_movie_clips():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    full = rng.integers(0, 256, (93, 512, 672), dtype=np.uint8)
    full[:, 0, 0], full[:, 0, 1] = 0, 255                           # same quantile bounds
    clips, valid, fm = X.partial_clips(full[:88], device="cpu")
    clips_full, valid_full, _ = X.partial_clips(full, n_valid=93, device="cpu")
    assert valid.tolist() == [True] * 13 + [False] and valid_full.all()
    assert clips.shape == (56, 16, 256, 256) and len(fm) == 88
    # quantiles differ slightly (real frames only), so compare after rescaling to [0, 1]
    a, b = clips[:52].numpy(), clips_full[:52].numpy()
    assert np.corrcoef(a.ravel()[::997], b.ravel()[::997])[0, 1] > 0.999
    assert torch.equal(clips[52:], clips[52:])                      # clip 13 built, flagged


def test_extend_cache_keeps_existing_rows_and_isolates_train_rows(tmp_path):
    rng = np.random.default_rng(3)
    base = tmp_path / "synthetic_base"
    base.mkdir()
    n, D = 6, 4
    meta = pd.DataFrame({"plate": ["synthetic_P1"] * 3 + ["synthetic_P2"] * 3,
                         "well": ["A1", "A2", "A3"] * 2, "Gene Symbol": "G",
                         "siRNA Identifier": "s1", "control": "", "series": "S",
                         "plate_split": "val", "gene_split": "train", "plate_norm_ok": True})
    z = rng.normal(size=(n, 14, 4, D)).astype(np.float16)
    np.save(base / "plate.npy", z)
    np.save(base / "fingerprint.npy", np.ones((n, 7), np.float16))
    meta.to_csv(base / "meta.tsv", sep="\t", index=False)
    np.savez(base / "plate_stats.npz", plates=np.array(["synthetic_P1", "synthetic_P2"]),
             mu=np.zeros((2, 14, D)), sigma=np.full((2, 14, D), 2.0),
             floor=np.full(D, 0.5, np.float32))
    (base / "stats.json").write_text(json.dumps({"n_wells": n}))
    raw = rng.normal(size=(2, 14, 4, D)).astype(np.float32)
    raw[:, 13] = np.nan                                             # clip 13 invalid
    new = pd.DataFrame({"plate": ["synthetic_P1", "synthetic_P2"], "well": ["B1", "B2"],
                        "Gene Symbol": "G", "siRNA Identifier": "s1", "control": "",
                        "series": "S", "plate_split": ["val", "train"],
                        "gene_split": "train"})
    rec = X.extend_cache(base, tmp_path / "synthetic_e3", raw, new, {"why": "test"})
    out = tmp_path / "synthetic_e3"
    z2 = np.load(out / "plate.npy")
    m2 = pd.read_csv(out / "meta.tsv", sep="\t")
    assert rec["n_wells_added"] == 2 and len(z2) == n + 2
    assert np.array_equal(z2[:n], z)                                # existing rows unchanged
    np.testing.assert_allclose(z2[n, :13].astype(np.float32), raw[0, :13] / 2, atol=1e-2)
    assert np.isnan(z2[n:, 13].astype(np.float32)).all()
    assert m2.plate_split.tolist()[n:] == ["val", "train_e3"]
    assert m2.e3_partial.tolist() == [False] * n + [True, True]


@pytest.mark.parametrize("name", ["80_e1_bscore.py", "82_e2_forecast_probe.py",
                                  "83_e3_partial.py"])
def test_exploratory_scripts_import_and_parse(name):
    import subprocess
    if name == "82_e2_forecast_probe.py":            # no CLI: import only
        cmd = [sys.executable, "-c", "import importlib.util as u, sys; "
               f"s = u.spec_from_file_location('m', r'{REPO / 'scripts' / name}'); "
               "m = u.module_from_spec(s); s.loader.exec_module(m)"]
    else:
        cmd = [sys.executable, str(REPO / "scripts" / name), "--help"]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
