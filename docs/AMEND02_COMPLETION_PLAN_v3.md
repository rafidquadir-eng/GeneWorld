# Amendment 02: completion plan, v3 (author review, 2026-10-23)

Review of `stage1` up to `54da373` (grid 245 of 288 runs, migration done, S2/S3 done).
Extends `docs/AMEND02_COMPLETION_PLAN.md` (v2); every v2 step and rule still applies. Nothing
here changes a gate, threshold, metric, preprocessing, split, grid value or the frozen lists.
Everything is operational, record-keeping, a review checkpoint, or exploratory after the test.

The reviewer checked the training summaries only for health and consistency, not for
per-family validation errors.

## 1. What was checked and is in order

- Migration: 1,956 files restored, hashes match; caches byte-identical; shard manifest and guard
  with a test; ledger reconciled without editing rows; smoke checkpoints and the 24 early 10-19
  shards handled as decided.
- Erratum E-02.1: implemented in `63_plan_a2.py` and enforced by `65_gates_a2.py`, with a test.
- 245 completed runs: no NaN; no best step <= 500 or at the 12,000 cap (median best 2,500, stop
  5,500); every family has a single (train, val) well count across both instances; all carry
  design commit `dd57079`.
- S2: no frame recovered; 1,083 wells excluded (745 train, 338 validation; 33 H1 primary
  validation wells); 868 of them miss only frames >= 88. S3 rule registered before any test
  or screenB embedding.
- F6 well-weighted test is secondary and outside `G_Signal_2`.

## 2. Problems and fixes

| # | problem | severity | fix |
|---|---|---|---|
| G1 | **Unattended OS upgrades kill CUDA.** `apt-daily-upgrade` restarted `nvidia-persistenced` / `nvidia-fabricmanager` at 06:20 UTC 10-23 (27 runs lost). The next run is scheduled for 10-24 06:55 UTC, inside the evaluation and test window. A failure during the one-time test run would be much more costly than a lost grid run. | **high** | S10 now: disable the timers and services that can restart system services, hold the NVIDIA packages, and verify. Author approves (this instance is temporary; nothing is re-enabled). |
| G2 | 550-server and 570-server driver packages are installed but not loaded (580 is loaded). Harmless while running; on a reboot DKMS could build or load the wrong module. | low | Hold all NVIDIA packages (S10). Do not purge or reboot while jobs run. Record in the run report. |
| G3 | The grid runner's log for 05:36–14:26 UTC 10-23 was lost (a `git stash` replaced the file being written). | low | S11: reconstruct a per-run grid record from per-run logs and results files. |
| G4 | Git operations in the working tree while jobs write into it caused G3. | medium (process) | Rule from now on: no `git stash`, `checkout`, `reset`, `rebase` or `pull` that touches the working tree while any job runs; commit by explicit paths only; do git maintenance in a separate worktree (`git worktree add`) if needed. |
| G5 | Mixed hardware and reruns: 52 runs on 1 x A100, the rest on 8 x A100; 27 runs restarted after incident 3 (12 after incident 2, 277 after incident 1). Only bf16 nondeterminism differs. | low | S11 records hardware and attempt number per run; the run report states it. |
| G6 | **Idle GPUs.** The validation evaluations are planned on a single GPU while the instance bills for eight ($10.32/h). | cost | S12: run independent evaluation jobs (H1, H2, H3, controls, alternative backbones, secondary runs) on separate GPUs where they do not depend on each other. Operational only; the order constraints of `run_a2_val.sh` stay (selection before H2 validation; planning source before complex-split runs; gates last). |
| G7 | **The one-time test is irreversible**, and the last review caught a gate-definition error (E-02.1) that no automatic check would have. | process | S13: stop after `gates_val.json` is committed; run the sanity checklist; send the author the summary; **wait for "go"** before the freeze. Bug fixes found at this point are allowed only if they are evaluation-code bugs (wrong data, wrong comparator, crash), never threshold or metric changes, and are recorded as errata. |
| G8 | Pre-registered exclusion drops wells missing only frames >= 88, which the primary horizons do not use (336 of the 338 failed validation wells; likely similar on test). | interpretation | Keep the rule. Add exploratory **E3** after the test (S15). |
| G9 | Cost control at the end. | cost | S16: when all GPU work is done, back up and push, then say so; the author terminates the instance. |

## 3. Steps (in order)

**S10. Stop automatic system changes (now; does not touch running jobs).**
```
sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer
sudo systemctl disable --now unattended-upgrades.service
sudo systemctl mask apt-daily.service apt-daily-upgrade.service
dpkg -l | awk '/^ii/ && $2 ~ /(nvidia|libnvidia|cuda|fabricmanager)/ {print $2}' | xargs sudo apt-mark hold
```
Verify and record in `logs/migration/os_freeze.log`:
- `systemctl list-timers --all | grep -i apt` shows nothing active;
- `apt-mark showhold` lists the NVIDIA packages;
- `nvidia-smi` and `systemctl is-active nvidia-fabricmanager` are unchanged.

Do not run `apt upgrade` and do not reboot until the author terminates the instance. If
`needrestart` is installed, set it to list-only (`$nrconf{restart} = 'l';` in
`/etc/needrestart/conf.d/99-noauto.conf`).

**S11. Grid record.** Write `results/a2/grid_runs.json` from `logs/a2/<run>.log`, the results
files, the locks and the old runner logs. For each of the 288 configuration x seed runs, give:
- host (old 1 x A100 / new 8 x A100) and GPU index where known;
- attempt number, and the incident that ended each failed attempt (1, 2 or 3);
- start, end and minutes;
- status;
- the results-file SHA-256.

Label anything reconstructed rather than logged. Commit with a note in the progress report.

**S12. Finish step 5, using the GPUs in parallel.** Let the grid finish. Then:
- `selection.json`, committed and pushed before anything else;
- H2 validation;
- the secondary runs across all 8 GPUs;
- the validation evaluations, with independent jobs on separate GPUs;
- `results/a2/gates_val.json`, committed and pushed.

Results must not depend on which GPU ran them; record the GPU per job.

**S13. Review checkpoint (stop here).** Write `reports/GATES_VAL_SUMMARY.md` and check each
item, marking it pass or fail.
1. Commit order: `selection.json` before any H2/H3 output, which comes before `gates_val.json`.
   Give the hashes and times.
2. Each gated family has 3 seeds at its selected configuration. Errors are averaged over seeds
   before testing.
3. Trained-shuffled controls used a fixed, recorded permutation per seed (give its hash). The
   true and shuffled families of each source were scored on identical wells and genes.
4. The H1 primary validation wells evaluated equal the frozen list minus the S2 exclusions
   (535 - 33 = 502). The H2 primary genes and the truth-list counts equal the frozen ones minus
   any exclusions, with the exclusions listed.
5. Every gate field reports n, effect, p or CI, and the pass/fail rule text. G-Plan-2 trajectory
   uses `retrieval_knn` (erratum E-02.1), and the oracle ceiling is reported.
6. Baselines are present in H1: replicate drift, replicate endpoint, slot and control drift,
   copy-last, and action-agnostic. The fallback counts for the replicate baselines are given.
7. No file under `results/` refers to a test plate, test gene or screenB well.
   `scripts/run_stage1_test.sh` has not been run.
8. Normalised and global results are both present.

Then give the gate outcomes (pass/fail with the main numbers) and the checklist in one message
to the author, and **wait for "go"**. While waiting, run nothing on test data. Exploratory work
on validation and training data is allowed, but must not be shown before the test.

**S14. After "go": the v2 step S6.** Freeze, embed test and screenB (one embedder per GPU,
BIA fetchers sharing at most 128 streams in total, S3 exclusion rule), and run
`scripts/run_stage1_test.sh` once.

**S15. Exploratory, after the test only** (labelled "exploratory, after the test").
- **E1 and E2:** as in v2.
- **E3, excluded wells:** embed clips 0–12 only (frames 0–87) for the wells excluded *only*
  because of missing frames >= 88, in train, validation, test and screenB. Then re-evaluate the
  frozen models' H1 primary metric on validation and test with these wells added. Report
  whether any gate's conclusion would differ. No model is retrained, and nothing changes a gate.

**S16. Report and stop.** Write `reports/RUN_REPORT_STAGE2.md` (v2 S8 plus G1–G8, the grid
record and E3). Back up every new file with SHA-256, commit and push. Then tell the author:
"all GPU work done; the instance can be terminated".
