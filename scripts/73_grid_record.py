"""Completion plan v3, S11: per-run record of the amendment-02 grid (288 runs).

Sources, in order of preference:
  * runner log `logs/a2_val/grid.log`: one section per runner start ("grid: N jobs to run");
    each "HH:MM:SS done|FAILED <run> (<min> min)" line is an attempt end (time of day only;
    the date comes from the section start in `logs/a2_val/pipeline.log`, rolling over at
    midnight); the current runner also logs the GPU. Section 4 (10-23 05:36-14:26 UTC) is
    lost (G3).
  * results files `results/a2/train/<run>.json` (minutes; on the new instance the file time is
    the run's end; restored files carry the restore time, so it is not used for them);
  * `results/a2/grid_status_at_migration.json` (76 runs completed on the old instance);
  * `logs/migration/incident_1023_nvidia_restart.log` (the 27 runs ended by incident 3);
  * live locks `checkpoints/a2/<run>.lock` (runs in flight).
Every attempt carries `source`: "logged" (runner line) or "reconstructed: <how>". Incidents:
1 = 10-21 CUDA out-of-memory (4 processes on one 40 GB GPU), 2 = 10-22 07:27-07:31 UTC
out-of-memory (three 50 M runs at once), 3 = 10-23 06:20 UTC OS upgrade restarted the NVIDIA
services. Writes `results/a2/grid_runs.json`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.train import a2  # noqa: E402

UTC = dt.timezone.utc
OLD = "old instance, 1 x A100-40GB (single GPU)"
NEW = "new instance, 8 x A100-SXM4-40GB"
#: runner sections of logs/a2_val/grid.log: start (logs/a2_val/pipeline.log), host, incident
#: that ended its failed attempts
SECTIONS = [
    ("2026-10-21T20:57:39", OLD, 1),
    ("2026-10-21T23:17:08", OLD, 2),
    ("2026-10-22T07:31:59", OLD, None),
    ("2026-10-23T05:35:47", NEW, 3),       # runner log lost (G3)
    ("2026-10-23T14:26:02", NEW, None),
    ("2026-10-23T16:26:51", NEW, None),    # S12 orchestrator's grid step (0 jobs to run)
]
LINE = re.compile(r"^(\d\d):(\d\d):(\d\d) (done|FAILED) (\S+) "
                  r"\((?:GPU (\d+), )?([\d.]+) min\)")


def iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def grid_names() -> list[dict]:
    P = a2.protocol()
    out = []
    for s in P["grid"]["seeds"]:
        for cfg in a2.configs():
            for f in P["fixed_before_any_amended_model_2026-10-20"]["families"]:
                out.append({"run": a2.run_name("plate", "vjepa2", f, cfg["id"], s),
                            "family": f, "config": cfg["id"], "seed": s})
    return out


def logged_attempts() -> dict[str, list[dict]]:
    att: dict[str, list[dict]] = {}
    sec, day, prev = -1, None, None
    for ln in (REPO / "logs" / "a2_val" / "grid.log").read_text().splitlines():
        if "jobs to run" in ln:
            sec += 1
            if sec >= len(SECTIONS):
                raise SystemExit(f"grid.log has runner section {sec + 1} ({ln!r}) that "
                                 "SECTIONS does not date; add its start time")
            start = dt.datetime.fromisoformat(SECTIONS[sec][0]).replace(tzinfo=UTC)
            day, prev = start.date(), start.time()
            continue
        m = LINE.match(ln)
        if not m:
            continue
        t = dt.time(int(m[1]), int(m[2]), int(m[3]))
        if t < prev:
            day += dt.timedelta(days=1)
        prev = t
        end = dt.datetime.combine(day, t, UTC)
        mins = float(m[7])
        _, host, inc = SECTIONS[sec]
        att.setdefault(m[5], []).append({
            "host": host, "gpu": m[6] if m[6] else ("0" if host == OLD else None),
            "start_utc": iso(end - dt.timedelta(minutes=mins)), "end_utc": iso(end),
            "minutes": mins, "status": "complete" if m[4] == "done" else "failed",
            "incident": inc if m[4] == "FAILED" else None, "runner_section": sec + 1,
            "source": "logged (runner line; date from the section start)"})
    return att


def main() -> int:
    runs = grid_names()
    att = logged_attempts()
    mig = set(json.loads((REPO / "results/a2/grid_status_at_migration.json").read_text())
              ["completed"])
    inc3_log = (REPO / "logs/migration/incident_1023_nvidia_restart.log").read_text()
    inc3 = set(inc3_log.split("## grid runs whose log shows the CUDA error")[1].split("##")[0]
               .split())
    live = {p.stem for p in (REPO / "checkpoints" / "a2").glob("*.lock")
            if Path(f"/proc/{json.loads(p.read_text())['pid']}").exists()}
    out_runs, notes = {}, []
    for r in runs:
        n = r["run"]
        res = REPO / "results" / "a2" / "train" / f"{n}.json"
        a = list(att.get(n, []))
        logged_done = any(x["status"] == "complete" for x in a)
        if n in inc3:
            a.append({"host": NEW, "gpu": None, "start_utc": None,
                      "end_utc": "2026-10-23T06:20:17Z..06:23:31Z", "minutes": None,
                      "status": "failed", "incident": 3, "runner_section": 4,
                      "source": "reconstructed: incident-3 run list and kernel log "
                                "(runner log lost)"})
        if res.exists() and not logged_done:
            d = json.loads(res.read_text())
            if n in mig:      # finished on the old instance after its runner stopped at 16:11
                a.append({"host": OLD, "gpu": "0", "start_utc": None,
                          "end_utc": "2026-10-22T16:11Z..16:39Z", "minutes": d["minutes"],
                          "status": "complete", "incident": None, "runner_section": 3,
                          "source": "reconstructed: completed after the old runner was "
                                    "stopped (16:11 UTC 10-22; grid status at 16:39 UTC)"})
            else:             # new instance, runner section 4 (log lost)
                end = dt.datetime.fromtimestamp(res.stat().st_mtime, UTC)
                a.append({"host": NEW, "gpu": None,
                          "start_utc": iso(end - dt.timedelta(minutes=d["minutes"])),
                          "end_utc": iso(end), "minutes": round(d["minutes"], 1),
                          "status": "complete", "incident": None, "runner_section": 4,
                          "source": "reconstructed: end = results-file time, start = end "
                                    "minus training minutes (data loading excluded); GPU "
                                    "unknown (runner log lost)"})
        a.sort(key=lambda x: x["end_utc"])
        for i, x in enumerate(a, 1):
            x["attempt"] = i
        status = ("complete" if res.exists() else "running" if n in live
                  else "failed" if a and a[-1]["status"] == "failed" else "pending")
        out_runs[n] = {**{k: r[k] for k in ("family", "config", "seed")}, "status": status,
                       "results_sha256": hashlib.sha256(res.read_bytes()).hexdigest()
                       if res.exists() else None, "attempts": a}
    by_inc = {i: sum(x["incident"] == i for v in out_runs.values() for x in v["attempts"])
              for i in (1, 2, 3)}
    if by_inc[2] != 12:
        notes.append(f"incident 2: the runner log shows {by_inc[2]} failed attempts "
                     "(07:27:21-07:31:21 UTC 10-22); reports/AMEND02_PROGRESS.md said 12")
    final = {}
    for v in out_runs.values():
        done = [x for x in v["attempts"] if x["status"] == "complete"]
        if done:
            h = done[-1]["host"]
            final[h] = final.get(h, 0) + 1
    out = {"generated_utc": iso(dt.datetime.now(UTC)), "n_runs": len(out_runs),
           "status_counts": {s: sum(v["status"] == s for v in out_runs.values())
                             for s in ("complete", "running", "pending", "failed")},
           "completed_on": final, "failed_attempts_by_incident": by_inc,
           "incidents": {"1": "2026-10-21 CUDA out-of-memory, 4 processes on one 40 GB GPU",
                         "2": "2026-10-22 07:27-07:31 UTC CUDA out-of-memory, three 50 M runs "
                              "on one 40 GB GPU",
                         "3": "2026-10-23 06:20 UTC unattended OS upgrade restarted the "
                              "NVIDIA services "
                              "(logs/migration/incident_1023_nvidia_restart.log)"},
           "runner_sections": [{"n": i + 1, "start_utc": s + "Z", "host": h,
                                "log": "lost (G3)" if i == 3 else "logs/a2_val/grid.log"}
                               for i, (s, h, _) in enumerate(SECTIONS)],
           "notes": notes, "runs": out_runs}
    (REPO / "results" / "a2" / "grid_runs.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
