"""Completion plan v3, S12: finish step 5 with the validation jobs in parallel on the 8 GPUs.

Same jobs and order constraints as `scripts/run_a2_val.sh` (step 5 onwards): grid complete ->
`results/a2/selection.json` committed and pushed before anything else -> H2 validation
(planning source) -> secondary training (global preprocessing, DINOv2 / TiJepa-1, then the
complex split, which needs the planning source) -> every validation evaluation -> commit ->
gates -> commit and push. Independent jobs run at the same time, each evaluation alone on one
GPU (CUDA_VISIBLE_DEVICES), training through `scripts/run_a2_jobs.py` on a GPU set. A job
starts only after its dependencies succeeded; a failed job blocks its dependents and the gates.
Every job's GPU, times and exit code go to `results/a2/eval_jobs.json`. Commits use explicit
paths (rule G4) after every new results file is backed up with SHA-256. Operational only.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld import backup  # noqa: E402

PY = str(Path.home() / "venv" / "bin" / "python")
LOG = REPO / "logs" / "a2_val"
RECORD = REPO / "results" / "a2" / "eval_jobs.json"
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def say(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(LOG / "pipeline.log", "a") as fh:
        fh.write(line + "\n")


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, text=True).stdout


def commit(msg: str, extra: tuple[str, ...] = ()) -> None:
    """Back up new or changed results files, then commit explicit paths and push (G4)."""
    if sh("git", "rev-parse", "--abbrev-ref", "HEAD").strip() == "main":
        raise SystemExit("refusing to commit on main")
    changed = [ln[3:] for ln in sh("git", "status", "--porcelain", "--untracked-files=all",
                                   "results").splitlines()]
    for f in changed:
        if (REPO / f).is_file():
            backup.backup(REPO / f)
    paths = [f for f in changed if (REPO / f).exists()] + ["reports/BACKUP_LEDGER.tsv"]
    sh("git", "add", "--", *paths)
    sh("git", "add", "-f", "--", "logs/a2_val", *extra)
    sh("git", "commit", "-q", "-m", msg + TRAILER)
    sh("git", "push", "-q", "origin", "HEAD:stage1")
    say(f"committed and pushed {sh('git', 'rev-parse', '--short', 'HEAD').strip()}: "
        f"{msg.splitlines()[0]}")


COMMIT = commit     # replaced in tests


def planning_source() -> str:
    return json.loads((REPO / "results/a2/h2/h2_val_plate_vjepa2.json").read_text())[
        "planning_source"]


def jobs() -> dict[str, dict]:
    """name -> cmd (list, or callable returning a list), deps, gpus (0 = CPU only)."""
    P = [PY]
    J = {
        "grid": dict(cmd=P + ["scripts/run_a2_jobs.py", "--phase", "grid"], deps=[], gpus=8,
                     log="grid.log"),
        # S11: final per-run grid record, committed with the selection
        "grid_record": dict(cmd=P + ["scripts/73_grid_record.py"], deps=["grid"], gpus=0),
        "selection": dict(cmd=P + ["scripts/60_select_a2.py"], deps=["grid_record"], gpus=0),
        "commit_selection": dict(commit=(
            "Amendment 02 step 5: grid trained (3 seeds); selected configurations recorded "
            "(results/a2/selection.json) before any gate; final grid run record "
            "(results/a2/grid_runs.json, S11)"), deps=["selection"]),
        "h2_val": dict(cmd=P + ["scripts/62_eval_h2_a2.py", "--split", "val"],
                       deps=["commit_selection"], gpus=1),
        "h1_val": dict(cmd=P + ["scripts/61_eval_h1_a2.py", "--split", "val"],
                       deps=["commit_selection"], gpus=1),
        "controls_plate": dict(cmd=P + ["scripts/64_controls_a2.py", "--norm", "plate"],
                               deps=["commit_selection"], gpus=1),
        # global preprocessing and alternative backbones (the complex split is added once
        # the planning source exists: the runner lists those jobs only if H2 validation exists)
        "secondary": dict(cmd=P + ["scripts/run_a2_jobs.py", "--phase", "secondary"],
                          deps=["commit_selection"], gpus=5, log="secondary.log"),
        "secondary_complex": dict(cmd=P + ["scripts/run_a2_jobs.py", "--phase", "secondary"],
                                  deps=["secondary", "h2_val"], gpus=4, log="secondary.log"),
        "h3_val": dict(cmd=P + ["scripts/63_plan_a2.py", "--split", "val"], deps=["h2_val"],
                       gpus=1),
        "h1_val_global": dict(cmd=P + ["scripts/61_eval_h1_a2.py", "--split", "val", "--norm",
                                       "global"], deps=["secondary"], gpus=1),
        "h2_val_global": dict(cmd=P + ["scripts/62_eval_h2_a2.py", "--split", "val", "--norm",
                                       "global"], deps=["secondary"], gpus=1),
        "h3_val_global": dict(cmd=P + ["scripts/63_plan_a2.py", "--split", "val", "--norm",
                                       "global"], deps=["secondary", "h2_val"], gpus=1),
        "h1_val_dinov2": dict(cmd=P + ["scripts/61_eval_h1_a2.py", "--split", "val",
                                       "--backbone", "dinov2"], deps=["secondary"], gpus=1),
        "h1_val_tijepa1": dict(cmd=P + ["scripts/61_eval_h1_a2.py", "--split", "val",
                                        "--backbone", "tijepa1"], deps=["secondary"], gpus=1),
        "controls_global": dict(cmd=P + ["scripts/64_controls_a2.py", "--norm", "global"],
                                deps=["secondary"], gpus=1),
        "h2_val_complex": dict(cmd=lambda: P + ["scripts/62_eval_h2_a2.py", "--split", "val",
                                                "--gene-split", "complex", "--sources",
                                                planning_source()],
                               deps=["secondary_complex"], gpus=1),
        "h3_val_complex": dict(cmd=P + ["scripts/63_plan_a2.py", "--split", "val",
                                        "--gene-split", "complex"],
                               deps=["secondary_complex"], gpus=1),
    }
    evals = [k for k in J if k.startswith(("h1_", "h2_", "h3_", "controls_"))]
    J["commit_evals"] = dict(commit=(
        "Amendment 02 step 5: validation evaluations (H1 with replicate baselines and "
        "phenotype probe CIs, H2, H3; global sensitivity; alternative backbones; complex "
        "split; controls), run in parallel on 8 GPUs (results/a2/eval_jobs.json)"),
        deps=evals + ["secondary_complex"])
    J["gates"] = dict(cmd=P + ["scripts/65_gates_a2.py", "--split", "val"],
                      deps=["commit_evals"], gpus=0)
    J["commit_gates"] = dict(commit=(
        "Amendment 02 step 6: G-Signal-2, G-ZeroShot-2, G-Plan-2 on validation recorded "
        "(results/a2/gates_val.json) before the freeze"), deps=["gates"])
    for k, j in J.items():
        j.setdefault("log", f"{k}.log")
    return J


JOBS = jobs         # replaced in tests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0, help="grid runner to wait for first")
    args = ap.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    if args.wait_pid:
        say(f"S12: waiting for grid runner {args.wait_pid}")
        while Path(f"/proc/{args.wait_pid}").exists():
            time.sleep(30)
    say("S12: validation jobs in parallel on 8 GPUs (scripts/74_val_parallel.py)")
    J = JOBS()
    free = [str(g) for g in range(8)]
    state = {k: "pending" for k in J}
    running: dict[str, tuple[subprocess.Popen, list[str], float]] = {}
    rec = {"host": socket.gethostname(), "hardware": "8 x A100-SXM4-40GB", "jobs": {}}

    def save():
        RECORD.write_text(json.dumps(rec, indent=1))

    while any(s in ("pending", "running") for s in state.values()):
        for k, j in J.items():
            if state[k] != "pending":
                continue
            if any(state[d] in ("failed", "blocked") for d in j["deps"]):
                state[k] = "blocked"
                say(f"blocked: {k}")
                continue
            if not all(state[d] == "done" for d in j["deps"]):
                continue
            if "commit" in j:
                save()
                try:
                    COMMIT(j["commit"])
                    state[k] = "done"
                except (subprocess.CalledProcessError, OSError) as e:
                    state[k] = "failed"
                    say(f"FAILED {k}: {e}")
                continue
            if len(free) < j["gpus"]:
                continue
            gpus = [free.pop(0) for _ in range(j["gpus"])]
            cmd = j["cmd"]() if callable(j["cmd"]) else j["cmd"]
            env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
            fh = open(LOG / j["log"], "a")
            running[k] = (subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh,
                                           stderr=subprocess.STDOUT), gpus, time.time())
            state[k] = "running"
            rec["jobs"][k] = {"cmd": " ".join(cmd[1:]), "gpus": gpus,
                              "log": f"logs/a2_val/{j['log']}",
                              "start_utc": time.strftime("%FT%TZ", time.gmtime())}
            say(f"start {k} on GPU {','.join(gpus) or '-'}")
        for k, (p, gpus, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[k]
            free = sorted(free + gpus, key=int)
            state[k] = "done" if rc == 0 else "failed"
            r = rec["jobs"][k]
            r.update(end_utc=time.strftime("%FT%TZ", time.gmtime()), exit_code=rc,
                     minutes=round((time.time() - t0) / 60, 1))
            save()
            say(f"{'done' if rc == 0 else 'FAILED'} {k} (exit {rc}, {r['minutes']} min)")
        time.sleep(5)
    bad = {k: s for k, s in state.items() if s != "done"}
    say("S12 finished: all jobs done" if not bad else f"S12 stopped: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
