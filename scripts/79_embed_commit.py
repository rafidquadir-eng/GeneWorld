"""Embed commit and allowed-paths check of the one-time test (completion plan v4 S17, D2).

Order of `scripts/run_stage1_test.sh`: freeze commit -> embed -> embed commit -> check ->
evaluation. The embed commit holds the new shards' manifest rows (SHA-256), the embed failure
list, the S3 tables and logs, and no evaluation output.

  manifest --split S   append every shard of the split's root that is not yet listed to the
                       split's manifest (SHA-256 from the file, checked against the backup
                       ledger's latest row); the rehearsal has its own manifest
  commit --split S     commit exactly the allowed files of the split (explicit paths) and push
  check --split S --base FREEZE --head EMBED
                       refuse unless FREEZE is an ancestor of EMBED, EMBED is pushed and on a
                       non-main branch, and every file changed between them is allowed
  preflight --mode test|rehearsal [--plates P1,P2,...]
                       the refusal conditions of scripts/run_stage1_test.sh (see `preflight`)
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from geneworld import backup  # noqa: E402

SPLITS = {
    "test": {"root": "data/latents_test", "manifest": "manifests/cache_shards.tsv",
             "results": ["results/a2/embed_failures_test.json",
                         "results/a2/missing_frames_test.json"]},
    "screenB": {"root": "data/latents_screenB", "manifest": "manifests/cache_shards.tsv",
                "results": ["results/a2/embed_failures_screenB.json",
                            "results/a2/missing_frames_screenB.json"]},
    "rehearsal": {"root": "data/latents_rehearsal",
                  "manifest": "manifests/cache_shards_rehearsal.tsv",
                  "results": ["results/a2/rehearsal/embed_failures_rehearsal.json",
                              "results/a2/rehearsal/missing_frames_rehearsal.json"]},
}
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def allowed(split: str) -> list[str]:
    s = SPLITS[split]
    return [re.escape(s["manifest"]), *map(re.escape, s["results"]), r"logs/.+"]


def offending(paths: list[str], split: str) -> list[str]:
    pats = [re.compile(p + r"\Z") for p in allowed(split)]
    return [p for p in paths if not any(r.match(p) for r in pats)]


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=REPO, check=True, capture_output=True,
                          text=True).stdout.strip()


def check(split: str, base: str, head: str) -> list[str]:
    """Problems (empty = the check passes)."""
    problems = []
    if subprocess.run(["git", "merge-base", "--is-ancestor", base, head], cwd=REPO).returncode:
        problems.append(f"freeze commit {base} is not an ancestor of {head}")
        return problems
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "main":
        problems.append("on main")
    git("fetch", "-q", "origin")
    if subprocess.run(["git", "merge-base", "--is-ancestor", head, f"origin/{branch}"],
                      cwd=REPO).returncode:
        problems.append(f"embed commit {head} is not pushed to origin/{branch}")
    changed = [p for p in git("diff", "--name-only", base, head).splitlines() if p]
    bad = offending(changed, split)
    if bad:
        problems.append(f"files changed between {base} and {head} outside the allowed paths: "
                        f"{bad}")
    if not changed:
        problems.append(f"no embed commit between {base} and {head}")
    return problems


def manifest(split: str) -> int:
    s = SPLITS[split]
    man = REPO / s["manifest"]
    listed = set()
    if man.exists():
        with open(man) as fh:
            listed = {r["path"] for r in csv.DictReader(fh, delimiter="\t")}
    led = {}
    with open(backup.LEDGER) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            led[r["path"]] = r["sha256"]
    rows, bad = [], []
    for f in sorted((REPO / s["root"]).rglob("*.npz")):
        rel = str(f.relative_to(REPO))
        if rel in listed:
            continue
        h = backup.sha256(f)
        if led.get(rel) != h:
            bad.append(rel)
        rows.append(f"{f.parent.name}\t{rel}\t{h}\n")
    if bad:
        print(f"refusing: {len(bad)} shards without a matching backup ledger row: {bad[:5]}")
        return 1
    new = not man.exists()
    with open(man, "a") as fh:
        if new:
            fh.write("backbone\tpath\tsha256\n")
        fh.writelines(rows)
    print(f"{len(rows)} shards added to {s['manifest']}")
    return 0


def commit(split: str, logs: list[str]) -> int:
    s = SPLITS[split]
    files = [s["manifest"], *[f for f in s["results"] if (REPO / f).exists()]]
    git("add", "--", *files)
    git("add", "-f", "--", *[f"logs/{x}" for x in logs if (REPO / "logs" / x).exists()])
    staged = [p for p in git("diff", "--cached", "--name-only").splitlines() if p]
    bad = offending(staged, split)
    if bad:
        print(f"refusing: staged files outside the allowed paths: {bad}")
        return 1
    msg = (f"Embed commit ({split}): shard manifest rows with SHA-256, embed failure list, S3 "
           "tables and logs; no evaluation output (completion plan v4 D2)")
    git("commit", "-q", "-m", msg + TRAILER)
    git("push", "-q", "origin", "HEAD")
    print(git("rev-parse", "--short", "HEAD"))
    return 0


TEST_RESULTS = ["results/h1/h1_test_*.json", "results/h2/h2_test_*.json",
                "results/h3/h3_test_*.json", "results/a2/h1/h1_test_*.json",
                "results/a2/h2/h2_test_*.json", "results/a2/h3/h3_test_*.json",
                "results/a2/h1/h1_screenB_*.json", "results/a2/h2/h2_screenB_*.json",
                "results/a2/gates_test.json"]


def preflight(mode: str, plates: list[str], root: Path | None = None) -> list[str]:
    """Refusal conditions (empty list = go).
    test: not in rehearsal mode; no uncommitted change under src, scripts, splits, manifests or
    docs; HEAD pushed and not on main; validation gates recorded; no test or screenB result.
    rehearsal: no test or screenB shard exists; no earlier rehearsal shards; every plate is a
    validation plate; selection.json exists."""
    root = root or REPO
    out = []
    def shards(d: str) -> list[Path]:
        return list((root / d).rglob("*.npz")) if (root / d).exists() else []

    if mode == "rehearsal":
        for d in ("data/latents_test", "data/latents_screenB"):
            if shards(d):
                out.append(f"{d} holds shards: the rehearsal runs only before any test or "
                           "screenB embedding")
        if (root / "data/latents_rehearsal").exists():
            out.append("data/latents_rehearsal exists: remove the earlier rehearsal first")
        ps = pd.read_csv(root / "splits/plate_split.tsv", sep="\t").set_index("plate").split
        bad = [p for p in plates if ps.get(p) != "val"]
        if not plates or bad:
            out.append(f"rehearsal plates must be validation plates: {bad or 'none given'}")
        if not (root / "results/a2/selection.json").exists():
            out.append("results/a2/selection.json does not exist")
        return out
    if os.environ.get("GENEWORLD_REHEARSAL"):
        out.append("GENEWORLD_REHEARSAL is set: the real test never runs in rehearsal mode")
    if not (root / "results/a2/gates_val.json").exists():
        out.append("validation gates not recorded (results/a2/gates_val.json)")
    for pat in TEST_RESULTS:
        found = sorted(root.glob(pat))
        if found:
            out.append(f"test results already exist ({found[0].relative_to(root)}): the test "
                       "splits are evaluated once")
    if root == REPO:
        if git("status", "--porcelain", "--", "src", "scripts", "splits", "manifests", "docs"):
            out.append("uncommitted changes under src, scripts, splits, manifests or docs")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        git("fetch", "-q", "origin")
        if branch == "main" or subprocess.run(
                ["git", "merge-base", "--is-ancestor", "HEAD", f"origin/{branch}"],
                cwd=REPO).returncode:
            out.append("the freeze commit is not pushed (or is on main)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "commit", "check", "preflight"])
    ap.add_argument("--split", choices=sorted(SPLITS))
    ap.add_argument("--logs", default="", help="comma-separated directories under logs/")
    ap.add_argument("--mode", choices=["test", "rehearsal"])
    ap.add_argument("--plates", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--head", default="HEAD")
    args = ap.parse_args()
    if args.cmd == "preflight":
        problems = preflight(args.mode, [p for p in args.plates.split(",") if p])
        for p in problems:
            print("refusing:", p)
        return 1 if problems else 0
    if args.cmd == "manifest":
        return manifest(args.split)
    if args.cmd == "commit":
        return commit(args.split, [x for x in args.logs.split(",") if x])
    problems = check(args.split, args.base, git("rev-parse", args.head))
    for p in problems:
        print("refusing:", p)
    if not problems:
        print(f"allowed-paths check passed ({args.base}..{args.head})")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
