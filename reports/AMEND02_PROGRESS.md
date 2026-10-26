# Amendment 02: progress (updated 2026-10-26 06:45 UTC)

Interim status while the amendment-02 round runs (design commit `dd57079`). Final results go in
`reports/RUN_REPORT_STAGE2.md`. No gate, selection or amended validation evaluation has been
computed before 10-26; validation gates are now recorded (`d0ab6e1`). No test plate has been
embedded and nothing has been evaluated on the test split or screenB.

| step | status | evidence |
|---|---|---|
| 1. Provenance (section 9) | done | V-JEPA 2 revision and weights SHA-256, verified on disk, in every results file; 28 stage-1 files backfilled, no value changed (`63613c0`) |
| 2. Freeze before training (sections 2, 6) | done | `splits/amend02_h1_primary_wells.tsv` (535 val / 1,112 test), `splits/amend02_h2_primary_genes.tsv` (204 / 417), truth lists with SHA-256 and unseen counts in `manifests/amend02_freeze.json` (`6716199`) |
| 3. Stage 2 (section 7) | done 20:21 UTC | 60,289 of 61,372 wells on 404 of 410 plates; 1,212 shards backed up with SHA-256 (`results/a2/stage2_data.json`). All 1,083 failures are HTTP 404 for frames missing from the S3 conversion (movies shorter than 93 frames; mostly frames 89–92): 6 plates entirely (LT0003_40, LT0008_14, LT0012_29, LT0079_38, LT0084_46, LT0096_29; 4 training, 2 validation) and 1–3 wells on 24 others |
| 4. Per-plate control normalisation and plate probe | done 20:57 UTC | no plate dropped (4–8 negative-control wells per plate, median 8); probes below (`results/a2/plate_probe.json`) |
| 5. Grid x 3 seeds, replicate baselines, probe CIs | done 05:41 UTC 10-26 (288 of 288 grid runs, `0a26d51`; evaluations `e866090`) | first attempt: 277 of 288 runs failed with CUDA out-of-memory (4 concurrent processes on 40 GB); 11 completed and are kept. Operational fix (evaluation chunks of 256 wells, expandable CUDA segments, 3 workers; no protocol value changed, `c03b433`); 277 runs restarted 23:17 UTC. Second incident 07:30 UTC 10-22: 50 M runs need ~14.6 GB in training, so three at once overflowed; 12 more runs failed (rerun). The runner now schedules by GPU memory (22 M = 1 unit, 50 M = 2, capacity 4) and respects live locks (`77c1905`); 52 of 288 runs done, 236 queued. Migrated 10-23 to 8 x A100-40GB (per-GPU slots, `264fee5`; section "Migration" below). Third incident 06:20 UTC 10-23: an unattended OS upgrade restarted the NVIDIA services and 27 in-flight runs failed (rerun from scratch; section "Incident 3"). 245 of 288 done at 14:30 UTC 10-23, 43 running or queued |
| S2/S3 missing frames (completion plan v2) | done 10-23 | S2 `65db5d2` (`results/a2/missing_frames.json`); S3 rule pre-registered for test and screenB in the protocol, `884a9c0` |
| 6. Gates on validation | done 05:41 UTC 10-26 (S12, `d0ab6e1`); S13 checklist in `reports/GATES_VAL_SUMMARY.md`; waiting for the author's "go" | `results/a2/gates_val.json`, committed before the freeze |
| 7. Freeze, embed test plates, test once | prepared; D2 order and rehearsal (S17/S18) done | `splits/test_wells_to_embed.tsv` (14,186 wells, 100 plates); extended `scripts/run_stage1_test.sh` |
| 8. Run report | after step 7 | |

## Decisions and deviations so far

- **Stage-2 fetch: BIA bucket only** (author decision). The IDR bucket holds one MitoCheck
  plate (LT0001_02), so plates cannot be split across buckets; `reports/DATA_ACCESS.md` is
  corrected. Native pycurl fetcher: 4.6 wells/s, frames identical to the stage-1 reader.
- **Author decisions for section 5** (`manifests/amend02_protocol.json`): trained
  shuffled-encoding controls (3 seeds per source); every gated family selects its own grid
  configuration; the H1 action-conditioned model is the learned per-gene embedding.
- **Fixed before training** (protocol): 22 M = width 512 x depth 6 and 50 M = width 768 x
  depth 7 (transformer core); rollout 6, lr 3e-4, batch 256, at most 12,000 steps, early stop
  3,000 steps after the last validation improvement; planning-source rule; secondary analyses
  use each family's selected configuration with 3 seeds.
- **Replicate baselines** average the replicate change over wells and their four tiles (tile
  positions are exchangeable). Matching tile k to tile k made replicate drift worse than
  copy-last on development data.
- **Plate-identity probe, early check on stage-1 data only:** 86.1% after per-plate control
  normalisation versus 77.8% with global standardisation (chance 5.6%); centring on the plate's
  control mean alone gives 86.3%. **Correction (2026-10-21):** this probe cannot tell batch
  effect from gene content. The validation plates are one per library series (51 plates, 51
  series), so each plate carries a different siRNA layout and a classifier can identify a
  plate from what is on it. The earlier reading (an estimation offset from the 5–8-well
  control mean) is not supported by this probe and is withdrawn. Section 1 stays as committed
  for the gates. Two diagnostics separate the two sources and are reported in step 4 for the
  global, control-normalised and centre-only versions:
  (a) a within-series replicate probe: training series with >= 3 training plates, well
  positions present on every replicate plate, decode the replicate plate, cross-validation
  grouped by well position;
  (b) a negative-control-only probe on validation plates, with each plate's normalisation
  statistics recomputed without the scored control well. Under this leave-one-out centring a
  plate's control wells are anticorrelated with one another, so accuracy at or below chance
  is the expected outcome when no plate offset remains.

## Plate-identity probes on the full stage-1 + stage-2 data (step 4)

Clip 12 (36 h), logistic regression; global / control-normalised / centre-only.

| probe | wells | chance | global | normalised | centre-only |
|---|---|---|---|---|---|
| main: 49 validation plates (one per series) | 8,574 | 2.0% | 74.4% | 93.9% | 92.9% |
| (a) within-series replicates: 76 training series, grouped by well position | — | 31.0% | 95.1% | 94.4% | 94.3% |
| (b) negative controls only, leave-one-out statistics: 49 validation plates | 369 | 2.0% | 50.9% | 0.0% | 0.0% |

Replicate plates of the same library series are told apart in about 95% of cases under all
three versions, with no help from gene content; the control normalisation lowers this by under
one point. In control wells alone the plate is decodable from global latents (50.9%), and the
leave-one-out normalisation removes it (0.0%; below chance is the expected result of
leave-one-out centring when no offset remains). So the plate effect that the controls measure
is removed by section 1, but most of the batch difference between replicate plates is not
carried by the control wells' offset.

## Grid runs at the migration stop (2026-10-22 16:39 UTC)

76 of 288 grid runs complete, 212 incomplete (never started or failed with out-of-memory; no run was stopped part-way: the two in flight at the stop finished normally). Cells list the completed seeds per family and configuration (per-plate normalisation, V-JEPA 2, gene split); "–" = none. Incomplete runs restart from scratch on the new instance; completed runs are never rerun or overwritten. Full lists: `results/a2/grid_status_at_migration.json`.

| family | `22M_d0.1_w0.05` | `22M_d0.1_w0.2` | `22M_d0.3_w0.05` | `22M_d0.3_w0.2` | `50M_d0.1_w0.05` | `50M_d0.1_w0.2` | `50M_d0.3_w0.05` | `50M_d0.3_w0.2` |
|---|---|---|---|---|---|---|---|---|
| `h1_learned` | 0 | 0 | 0 | 0 | 0 | 0 | – | – |
| `h1_none` | 0 | 0,1 | 0 | 0 | 0 | 0 | – | – |
| `h2:perturbseq` | 0 | 0 | 0 | 0 | 0 | 0 | – | – |
| `h2:jump_crispr` | 0 | 0 | 0 | 0 | 0 | 0 | – | – |
| `h2:jump_orf` | 0 | 0 | 0 | 0 | 0 | 0 | – | – |
| `h2:string` | 0 | 0,1 | 0 | 0 | 0 | 0 | – | – |
| `h2:esm2` | 0,2 | 0 | 0 | 0 | 0 | 0 | – | – |
| `h2shuf:perturbseq` | 0 | 0 | 0 | 0 | 0 | 0 | – | – |
| `h2shuf:jump_crispr` | 0 | 0 | 0 | 0 | 0 | 0 | – | – |
| `h2shuf:jump_orf` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | – |
| `h2shuf:string` | 0 | 0 | 0 | 0 | 0 | 0 | – | – |
| `h2shuf:esm2` | 0 | 0 | 0 | 0 | 0 | 0 | – | – |

## Backup audit for the migration (2026-10-22 17:40 UTC)

`scripts/70_backup_audit.py` -> `results/a2/backup_audit.json`: 2,017 files, 303.9 GB (1,770
latent shards 244.5 GB; 24 cache files 44.5 GB; 94 run summaries; 94 checkpoints 14.1 GB; 35
annotation/encoding/reference tables 0.9 GB). 1,896 were already backed up with a matching
ledger hash; 121 (the caches, and run summaries and checkpoints written since their last ledger
row) were backed up now. Every backup copy was re-hashed: no mismatch, no file missing from
`~/LambdaFilesystem/geneworld_backups/` or from `reports/BACKUP_LEDGER.tsv`.

## Migration to 8 x A100-40GB (2026-10-23)

- **Hardware:** Lambda on-demand, us-east-1, 8 x NVIDIA A100-SXM4-40GB (40,960 MiB, VBIOS
  92.00.45.00.03), all pairs NVLink NV12 via NVSwitch; 2 x AMD EPYC 7542, 124 vCPUs, 1.77 TB RAM;
  Ubuntu 24.04.2, kernel 6.8.0-62. Driver installed by hand (not a Lambda Stack image): loaded
  kernel module **580.178.04**, **Fabric Manager 580.178.04** (active; 550-server and 570-server
  packages are also installed but not loaded). Python 3.12.3, `~/venv` from
  `requirements-lock.txt`, **torch 2.11.0+cu128** (CUDA 12.8): `torch.cuda.device_count()` = 8,
  bf16 matmul checked on every GPU.
- **Restore** (`scripts/71_restore.py` -> `results/a2/restore_audit.json`): 2,054 ledger paths
  (2,083 rows); 1,956 files restored (309.3 GB) and 95 already present (in git), each with a
  SHA-256 matching its latest ledger row. Every row reconciled: 2,029 restored or present with a
  matching hash, 24 superseded by a later row for the same path, 27 rows of 24 files kept aside,
  3 without a backup copy. Ledger rows are never edited; two reconciliation notes appended.
  - *Without a backup copy:* `checkpoints/predictors/smoke_{l,n,s}.pt` (10-19 stage-1 smoke
    tests, unreferenced, not needed; author decision 10-23).
  - *Kept aside:* 24 early 10-19 shards (8 stage-2 training plates x 3 backbones; 100 wells,
    no fingerprint) that were not on the old instance's disk when the caches were built (572
    shards per backbone; the 10-22 audit counted 1,770 = 572 x 3 + 54 static). They would
    have replaced the stage-2 latents of 58–100 wells per plate. Moved to
    `data/latents_unused_1019/` and not used (author decision 10-23).
- **Shard manifest** (author instruction 10-23): `manifests/cache_shards.tsv` lists the 1,770
  shards under `data/latents*/` (572 per backbone + 18 static per backbone) with SHA-256;
  `cache.build` and the backup audit refuse to run if a shard is not in it or a listed shard
  is missing (`data/latents_unused_1019/` excluded by name; `tests/test_cache_guard.py`).
- **Caches rebuilt and reproduced exactly:** for V-JEPA 2, DINOv2 and TiJepa-1, `stats.json` is
  byte-identical to `results/a2/cache_stats_*.json`, and every rebuilt cache file (raw, plate,
  global, fingerprint, meta, statistics) is byte-identical to the restored cache.
- **V-JEPA 2 weights:** SHA-256 `25466aef…786b` matches `manifests/models.yaml`.
- **Runner:** per-GPU slots (`CUDA_VISIBLE_DEVICES` per job, 4 memory units per GPU, 22 M = 1,
  50 M = 2); operational only. Input dry run: 12 of 12 families load; train/validation well
  counts identical to the completed runs.

## Erratum E-02.1 (2026-10-22, before any gate)

The G-Plan-2 trajectory rule compared the planner with replicate-endpoint retrieval, which for
unseen genes uses the unseen gene's own observed wells (an oracle; stage 1: hit@100 0.148 vs
planner 0.034). Corrected per the author's completion plan v2 (F8): the trajectory half passes
if hit@100 is above random (binomial p < 0.05) and above kNN-imputed endpoint retrieval
(bootstrap CI over target genes above 0). Replicate-endpoint retrieval stays in every table as
the oracle ceiling. Applied before any selection, H2/H3 validation output or gate existed
(`docs/DESIGN_AMENDMENT_02.md` section 10; `scripts/63_plan_a2.py`, `scripts/65_gates_a2.py`,
`tests/test_erratum_e021.py`).

## Author decisions 2026-10-21

- ScreenB: evaluated after the freeze under the global preprocessing only, outside the gates
  (H1 and H2, siRNA token masked); reported next to the global test results.
- Plate-probe diagnostics (a) and (b) added; section 1 unchanged.

## Timeline (revised 10-23, 8 x A100-40GB)

Grid: 245 of 288 runs done at 14:30 UTC 10-23; the remaining 43 (27 reruns after incident 3,
16 last 50 M configurations) to about 16:00–16:30 UTC. Then selection (committed before any
gate), H2 validation, 57 secondary runs (~1–1.5 h on 8 GPUs), the validation evaluations
(single GPU) and `results/a2/gates_val.json`, probably late 10-23 or overnight. Then the freeze,
test and screenB embedding (~1.5 h on 8 GPUs), the test run, E1/E2 and the run report:
about 10-24 to 10-25.

## Completion plan v3 (2026-10-23)

- **S10 (author-approved):** apt timers and unattended-upgrades disabled, apt services masked,
  24 NVIDIA/CUDA packages held, needrestart list-only; NVIDIA state unchanged
  (`logs/migration/os_freeze.log`). No upgrade or reboot until the instance is terminated.
- **G4:** no stash, checkout, reset, rebase or pull in the working tree while jobs run; commits
  by explicit paths. The v3 plan was merged in a separate worktree (`b9a1f76`); the main
  worktree's branch ref and index were advanced without writing files, so
  `docs/AMEND02_COMPLETION_PLAN_v3.md` is materialised in the working tree only once no job
  runs.
- **S11 grid record** (`scripts/73_grid_record.py` -> `results/a2/grid_runs.json`): per run
  the attempts with host, GPU where known, start/end/minutes, status, the incident ending each
  failed attempt and the results-file SHA-256; every attempt labelled "logged" (runner line)
  or "reconstructed: <how>". Snapshot 14:57 UTC 10-23: 245 complete (76 old instance, 169 new),
  16 running, 27 awaiting rerun; failed attempts 277 (incident 1), 13 (incident 2), 27
  (incident 3). Correction: the runner log shows **13** failed attempts in incident 2
  (07:27:21–07:31:21 UTC 10-22), not 12 as stated above. Reconstructed: the 169 runs of runner
  section 4 (end = results-file time, GPU unknown, G3), the 27 incident-3 attempts, and 2
  runs that finished on the old instance after its runner was stopped (16:11–16:39 UTC 10-22).
  The final record is regenerated when the grid completes and committed with the selection.
- **S12:** `scripts/run_a2_val.sh` was stopped during its grid step (its grid runner
  continues); `scripts/74_val_parallel.py` takes over from the end of the grid (selection
  first, then the independent validation jobs in parallel on the 8 GPUs, then the gates). Grid at 15:44 UTC: 261 of 288 runs complete.
- **CPU-side work while the GPUs finish step 5** (author, 10-23; no result read): E1-E3 code
  written and unit-tested, not run on real data (`src/geneworld/explore.py`,
  `scripts/8[023]_*.py`; `91f9d23`); run-report generator `scripts/90_run_report_stage2.py`
  (`9aa0ac0`); manuscript draft `paper/geneworld/` with a verified novelty search and numbers
  generated from results (`scripts/91_paper_numbers.py`; `14b63b8`). Details and open points in
  `reports/AMEND02_REVIEW.md` (update 15:50 UTC).

## Incident 3: OS upgrade restarted the NVIDIA services (2026-10-23 06:20 UTC)

- **What happened:** Ubuntu's `apt-daily-upgrade` (unattended-upgrades) ran from 06:09 UTC. After
  the libc6 upgrade (06:20:09–06:20:15) systemd re-executed and restarted
  `nvidia-persistenced` and `nvidia-fabricmanager` at 06:20:17; Fabric Manager came back at
  06:20:23 and was reconfigured again at 06:23:31. Every CUDA context was torn down (Xid 45 on
  all 8 GPUs; no ECC, row-remapping or other hardware error; driver, Fabric Manager and CUDA
  versions unchanged). Evidence: `logs/migration/incident_1023_nvidia_restart.log`.
- **Effect:** 27 grid runs failed (`cudaErrorLaunchFailure` / `CUBLAS_STATUS_EXECUTION_FAILED`):
  the 16 in flight at 06:20 and 11 started before the second reconfiguration at 06:23. None
  had a results file; they restart from scratch (partial checkpoints overwritten), as the
  completion plan prescribes for stopped runs. No completed run was in flight at 06:20 (start =
  results-file time minus run minutes), so all 245 completed runs are unaffected. No protocol
  value changed.
- **Lost runner log (my error):** at about 05:45 UTC a `git stash` / `git stash pop` in the
  working tree replaced `logs/a2_val/grid.log` while the grid runner (PID 43528) was writing to
  it, so that runner's per-run lines and final summary (05:36–14:26 UTC) went to an unlinked
  file and are lost. Per-run logs (`logs/a2/<run>.log`) and every results file are intact; the
  restarted pipeline's runner logs normally. The working tree is no longer touched while jobs
  write to it.
- **Open (author decision):** the next `apt-daily-upgrade` is scheduled for 10-24 06:55 UTC,
  inside the evaluation / test window; disabling `apt-daily.timer` and
  `apt-daily-upgrade.timer` until the work is done has been proposed.
