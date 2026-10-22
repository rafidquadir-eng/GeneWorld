"""Erratum E-02.1: the G-Plan-2 trajectory gate must reference kNN-imputed retrieval."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _src(name: str) -> str:
    return (REPO / "scripts" / name).read_text()


def test_planner_gate_uses_retrieval_knn():
    src = _src("63_plan_a2.py")
    assert re.search(r'^TRAJECTORY_GATE_COMPARATOR = "retrieval_knn"$', src, re.M)
    # the pass rule reads the comparator constant, and the rank array of the comparator
    assert 'tr[f"planner_vs_{TRAJECTORY_GATE_COMPARATOR}_hit@100"]["ci95"][0] > 0' in src
    assert "hr = np.array(ranks[TRAJECTORY_GATE_COMPARATOR]) <= 100" in src
    # the oracle is still computed and reported, under the oracle-ceiling name
    assert 'ORACLE = "oracle_ceiling_replicate_endpoint"' in src
    assert "sc[ORACLE] = ob" in src
    assert "replicate_endpoint_retrieval" not in src.split('"""', 2)[2]


def test_gate_collector_enforces_retrieval_knn():
    src = _src("65_gates_a2.py")
    assert re.search(r'^TRAJECTORY_GATE_COMPARATOR = "retrieval_knn"$', src, re.M)
    assert "if comp != TRAJECTORY_GATE_COMPARATOR:" in src


def test_constants_importable():
    spec = importlib.util.spec_from_file_location("plan", REPO / "scripts" / "63_plan_a2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TRAJECTORY_GATE_COMPARATOR == "retrieval_knn"
