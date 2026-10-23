"""S12 orchestrator: dependency order, GPU assignment, failures block dependents."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(tmp_path):
    spec = importlib.util.spec_from_file_location("vp",
                                                  REPO / "scripts" / "74_val_parallel.py")
    vp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vp)
    vp.LOG = tmp_path / "synthetic_logs"
    vp.RECORD = tmp_path / "synthetic_eval_jobs.json"
    return vp


def synthetic_jobs(fail: str = ""):
    py = sys.executable
    ok = [py, "-c", "import os, time; time.sleep(0.2); "
                    "print(os.environ.get('CUDA_VISIBLE_DEVICES'))"]
    bad = [py, "-c", "raise SystemExit(3)"]
    J = {"a": dict(cmd=ok, deps=[], gpus=8), "c_a": dict(commit="synthetic a", deps=["a"]),
         "b1": dict(cmd=bad if fail == "b1" else ok, deps=["c_a"], gpus=1),
         "b2": dict(cmd=ok, deps=["c_a"], gpus=5),
         "d": dict(cmd=ok, deps=["b1", "b2"], gpus=1),
         "c_d": dict(commit="synthetic d", deps=["d"])}
    for k, j in J.items():
        j.setdefault("log", f"{k}.log")
    return lambda: J


def test_order_and_gpus(tmp_path):
    vp = load(tmp_path)
    commits = []
    vp.COMMIT, vp.JOBS = commits.append, synthetic_jobs()
    sys.argv = ["x"]
    assert vp.main() == 0
    assert commits == ["synthetic a", "synthetic d"]
    rec = json.loads(vp.RECORD.read_text())["jobs"]
    assert rec["a"]["gpus"] == [str(g) for g in range(8)]
    assert len(rec["b2"]["gpus"]) == 5 and not set(rec["b1"]["gpus"]) & set(rec["b2"]["gpus"])
    assert all(r["exit_code"] == 0 for r in rec.values())
    assert rec["d"]["start_utc"] >= max(rec["b1"]["end_utc"], rec["b2"]["end_utc"])


def test_failure_blocks_dependents(tmp_path):
    vp = load(tmp_path)
    commits = []
    vp.COMMIT, vp.JOBS = commits.append, synthetic_jobs(fail="b1")
    sys.argv = ["x"]
    assert vp.main() == 1
    assert commits == ["synthetic a"]                    # the final commit never happens
    rec = json.loads(vp.RECORD.read_text())["jobs"]
    assert rec["b1"]["exit_code"] == 3 and "d" not in rec
