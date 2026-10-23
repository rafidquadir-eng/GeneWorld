"""F6 secondary H1 test: plate sign flip with plates weighted by their number of wells."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.eval import a2eval as E  # noqa: E402


def synthetic_errors(sizes, seed=0):
    rng = np.random.default_rng(seed)
    plates = np.repeat([f"synthetic_p{i}" for i in range(len(sizes))], sizes)
    base = rng.normal(5, 1, len(plates))
    model = base + rng.normal(-0.1, 0.5, len(plates))
    return model, base, plates


def test_equal_sizes_match_plate_test():
    m, b, p = synthetic_errors([4] * 12)
    a, w = E.plate_test(m, b, p), E.well_weighted_plate_test(m, b, p)
    assert np.isclose(a["mean"], w["mean"])
    assert np.isclose(a["p_one_sided"], w["p_one_sided"])


def test_unequal_sizes_give_well_weighted_mean():
    m, b, p = synthetic_errors([1, 8, 2, 8, 1, 5, 3, 8, 1, 6])
    w = E.well_weighted_plate_test(m, b, p)
    assert np.isclose(w["mean"], np.mean(m - b))           # every well counts once
    assert w["n_wells"] == len(p) and w["wells_per_plate"] == {"min": 1, "median": 4.0,
                                                               "max": 8}
    assert w["n_groups"] == 10                              # signs still flip per plate
