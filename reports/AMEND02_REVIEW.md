# GeneWorld amendment 02: review packet (updated 2026-10-26 06:45 UTC)

## Update 2026-10-26 06:45 UTC: S12 finished, S18 finished, S13 checklist (stopped for "go")

Covers the work since the 10-23 15:50 update, under completion plans v3 and v4 (`8953484`).
Validation only: **no test plate embedded, no test or screenB evaluation, no exploratory
analysis on real data.** Gate numbers and the checklist are in `reports/GATES_VAL_SUMMARY.md`.

| step | status | commits / evidence |
|---|---|---|
| S11 grid record | done; 288 of 288 runs complete; failed attempts by incident 277 / 13 / 27 | `0a26d51`; `results/a2/grid_runs.json` |
| S12 parallel validation | done 05:41 UTC 10-26; all jobs exit 0 | `e866090` (evaluations), `d0ab6e1` (gates); `results/a2/eval_jobs.json` |
| S17 test path | done; tests pass | `72f12a8` |
| S18 rehearsal | done 06:22 UTC 10-26; re-embedding byte-identical for all three backbones; full path ran, one step (`h2_global`) rerun | `5918338`, `83127b8`, this update; `results/a2/rehearsal/summary.json` |
| S13 checklist | 9 of 10 pass (item 9 after a rerun); item 7 flagged for the author | `reports/GATES_VAL_SUMMARY.md` |

### Incidents and deviations since 10-23

- **S12 stopped twice at the grid record** (attempt 1 at 16:27 UTC 10-23, attempt 2 at 22:55
  UTC 10-25): `scripts/73_grid_record.py` refused a runner section with no date (the
  orchestrator's own grid step adds one per start). Fixed by dating each such section from its
  `start grid` line (`7856987`). Both attempts stopped before any selection or evaluation; their
  job records are kept (`results/a2/eval_jobs_attempt1.json`, `_attempt2.json`). The GPUs were
  idle about 54.5 h (10-23 16:27 to 10-25 22:55 UTC) while the session was paused.
- **Parallelisation (author, 10-25):** S18 ran on GPU 2 alongside S12; the 9 complex-split
  secondary runs were trained by a complex-only runner (GPUs 0-2) from 02:07 UTC 10-26, so the
  orchestrator's `secondary_complex` step found nothing to run. Same configurations, seeds and
  inputs (`e7aa35e`).
- **S18 `h2_global` ordering:** the rehearsal's first `h2_global` attempt (01:59 UTC 10-26)
  failed because the global secondary models were still training in S12; it was rerun after
  they finished (`logs/rehearsal/resume.sh`) and exited 0. The summary keeps the strict flag
  (`evaluation_all_ok: false`) and adds `evaluation_last_run_ok: true` with the failed attempt
  listed.
- **Item 7 (author judgment):** two results files contain test-split gene names.
  `results/a2/missing_frames.json` labels failed wells on training and validation plates (S2).
  `results/a2/rehearsal/selection_of_wells.json` lists the 100 training-plate wells re-embedded
  for the S18 comparison: 18 of them are wells of test-split genes on training plates, already
  embedded in stage 2. They were used only in the byte-identity comparison (identical); no
  evaluation used them.


## Update 2026-10-23 15:50 UTC: completion plan v3 (S10-S12 under way) and CPU-side work

Covers the work since the 14:45 update below, under `docs/AMEND02_COMPLETION_PLAN_v3.md`
(`bd9d376`) and the author's instructions of 10-23. **Still no selection, no H2/H3 or amended H1
validation output, no gate; no test plate, test gene or screenB well embedded or evaluated; no
exploratory analysis run on real data.** No per-family validation errors are reported here.

### Status

| step | status | commits / evidence |
|---|---|---|
| v3 merged into stage1 | done, in a separate worktree; main worktree's ref and index advanced without writing files during jobs | `b9a1f76` |
| S10 stop automatic system changes | done, verified (author-approved) | `6683e51`; `logs/migration/os_freeze.log` |
| G4 git hygiene while jobs run | in force: no stash/checkout/reset/rebase/pull; commits by explicit paths | |
| S11 grid run record | snapshot committed; final version regenerated after the grid and committed with the selection | `726aae6`; `results/a2/grid_runs.json` |
| S12 finish step 5 in parallel | grid 261 of 288 at 15:44 UTC (last 50 M runs; about 1 h left); orchestrator `scripts/74_val_parallel.py` waiting on the grid runner | `9e2044a`, `726aae6` |
| S13 review checkpoint | pending: GATES_VAL_SUMMARY.md with the 8-point checklist, then stop for "go" | |
| CPU work (author, 10-23) | E1-E3 code, run-report generator, manuscript draft: done, see below | `91f9d23`, `9aa0ac0`, `14b63b8` |

### S10-S12 details

- **S10:** apt timers and unattended-upgrades disabled, apt services masked, 24 NVIDIA/CUDA
  packages held, needrestart list-only; driver 580.178.04 and Fabric Manager unchanged; the
  running grid was unaffected.
- **S11:** per run the attempts with host, GPU where known, times, status, the incident ending
  each failed attempt and the results-file SHA-256; each attempt labelled logged or
  reconstructed. Snapshot 14:57 UTC: failed attempts 277 / 13 / 27 for incidents 1 / 2 / 3.
  **Correction:** incident 2 had 13 failed attempts in the runner log, not 12 as reported
  earlier.
- **S12:** `scripts/run_a2_val.sh` (serial evaluations on one GPU) was stopped during its grid
  step; its grid runner continues. `scripts/74_val_parallel.py` waits for the grid, writes the
  final grid record, commits `selection.json` (pushed) before anything else, then runs H2
  validation, H1 validation, the plate controls and the secondary runs in parallel, the
  complex-split runs once the planning source exists, the remaining evaluations (each alone on
  one GPU), commits, then the gate collector, commits and pushes. A failed job blocks its
  dependents and the gates. Per-job GPU, times and exit codes go to
  `results/a2/eval_jobs.json`. Scheduler unit test `tests/test_val_parallel.py`.
- **Own error, caught before harm:** while relaunching the orchestrator to add the grid-record
  step, a `kill` hit the old orchestrator's wrapper shell instead of the Python process, and a
  commit command chain ending in `&` ran in the background and would have started a second
  orchestrator after its tests. Both were found within minutes and stopped before either
  started a job or a commit; exactly one orchestrator (PID 185417) runs. Launches are now
  separate commands, verified by PID.

### CPU-side work (no result read, nothing shown)

1. **E1, E2, E3 (S15)** written and unit-tested, not run on real data:
   `src/geneworld/explore.py` (median polish / B-score; the within-series replicate probe; partial
   movies of frames 0-87 with clip 13 invalid; per-plate statistics for new wells; cache
   extender that leaves existing rows byte-identical), `scripts/80_e1_bscore.py`,
   `82_e2_forecast_probe.py`, `83_e3_partial.py`. They reuse the committed training and
   evaluation scripts unchanged and write only to `checkpoints/a2_e1/` and
   `results/a2/exploratory/`, every output labelled "exploratory, after the test". Tests on
   synthetic data (`tests/test_explore.py`) caught one bug (chunked copy bound), fixed.
   Equivalence: `explore.within_series_probe` reproduces the committed probe (a) exactly in the
   plate space (only match/no-match was printed).
2. **Run report generator** `scripts/90_run_report_stage2.py`: builds
   `reports/RUN_REPORT_STAGE2.md` and figures from `results/` only; directional words from p
   values and CIs in code; absent inputs (test) give one-sentence placeholders; exploratory
   numbers only once `gates_test.json` exists. Dry run on the current files reproduces the
   committed numbers; figures use a palette that passes the dataviz validator, with numbers also
   in tables. To do: check on the validation outputs; add incidents, deviations and "not run".
3. **Manuscript** `paper/geneworld/` (TMLR; anonymised `main_anon.tex` and named
   `main_biorxiv.tex`): introduction, related work, data, model, evaluation design, controls and
   baselines, pre-registration and errata; results and discussion are placeholders. All 59
   numbers are macros from `scripts/91_paper_numbers.py`. Novelty search of 2026-10-23
   (`novelty_search_2026-10-23.md`): 27 references, each verified against Crossref or arXiv; no
   prior gene-conditioned latent world model of live-cell video with zero-shot transfer to
   unseen genes found (closest: transcriptomic cell world models, static-image perturbation
   generators, microscopy video diffusion without perturbation conditioning). Compiles with
   Tectonic (scratchpad binary; no system package installed). Author TODOs: e-mail,
   affiliation, data/code availability. Nothing submitted or posted.

### Points for the reviewer (in addition to those of the 14:45 update)

1. **S13 checklist item 4 will fail as written.** It expects 535 - 33 = 502 H1 primary
   validation wells; 362 are evaluated. Decomposition of the 535 frozen wells: 362 evaluated,
   98 of test-split genes and 42 of validation-split genes (the H1 action-conditioned model is a
   learned per-gene embedding and has no embedding for held-out-split genes; the evaluation mask
   requires the gene in the training vocabulary), 33 S2 exclusions. The same 362 wells were the
   training-time early-stopping set. Not changed (it would alter selection and need retraining);
   author decision at S13.
2. **`scripts/run_stage1_test.sh` before S14:** it embeds test plates with a single embedder
   (S14 requires one per GPU, <= 128 streams in total), and its `cache.build(..., '_test')` will
   be refused by the shard guard unless the test shards are first added to
   `manifests/cache_shards.tsv` (proposed: a commit right after embedding, before any
   evaluation). Same for screenB.
3. **E3 design choice:** wells added by E3 on training plates are kept out of every training set
   and replicate baseline (the frozen models' input check requires their original training
   wells, and the baselines were defined on them); only validation and test wells are added to
   the evaluated set.
4. **Validation evaluations now run in parallel** (point 5 of the 14:45 update resolved by S12).

---

## Update 2026-10-23 14:45 UTC: migration to 8 x A100-40GB, S2, S3, grid resumed

Covers the work since the 10-22 packet below, under `docs/AMEND02_COMPLETION_PLAN.md` v2 and
the author's instructions of 10-23. Commits `131156c`..HEAD on `stage1`; `main` untouched.
**Still no selection, no H2/H3 or amended H1 validation output, no gate; no test plate, test
gene or screenB well embedded or evaluated.** As before, no per-family validation errors are
reported here.

### Status

| completion-plan step | status | commits / evidence |
|---|---|---|
| S1 erratum E-02.1 | done 10-22 (confirmed on the new instance) | `3236065`, `5f5ca7e` |
| Migration (section 3, steps 2–3) | done; author told "restore verified; the old instance can be terminated" | `131156c`, `264fee5`, `54ffd2f`; `results/a2/restore_audit.json` |
| S2 missing frames confirmed | done | `65db5d2`; `results/a2/missing_frames.json` |
| S3 rule pre-registered for test and screenB | done (before the freeze, before any test/screenB embedding) | `884a9c0`; protocol `missing_frames_rule` |
| S5 grid | running: 245 of 288 runs done at 14:30 UTC, 43 left (27 reruns after incident 3) | `results/a2/train/*.json` |
| S5 selection -> H2 val -> secondary -> validation evaluations -> `gates_val.json` | queued, automatic (`scripts/run_a2_val.sh`, resumed at step 5; commits, backs up and pushes after each step) | |
| S6–S9 freeze, test/screenB embedding, test once, E1/E2, run report, GPUs idle | pending, by hand | |

### What was done and how

- **Hardware and environment:** 8 x A100-SXM4-40GB (NVSwitch, NV12), 2 x EPYC 7542, 1.77 TB
  RAM, Ubuntu 24.04.2; driver **580.178.04** and Fabric Manager **580.178.04** (installed by
  hand; the author's prompt said 550-server, which is installed but not loaded); `~/venv` from
  `requirements-lock.txt` (`3888e77`), torch 2.11.0+cu128; bf16 matmul checked on every GPU.
- **Restore** (`scripts/71_restore.py`): every ledger path copied from the persistent
  filesystem to a `.part` file, hashed, moved into place only on a match with the path's latest
  ledger row; files already present (git) hashed, never overwritten. 1,956 restored (309.3 GB),
  95 present, all matching. All 2,083 ledger rows reconciled (2,029 matching, 24 superseded by a
  later row, 27 rows of 24 kept-aside files, 3 without a backup copy).
- **Two author decisions on the restore (10-23):**
  1. `checkpoints/predictors/smoke_{l,n,s}.pt` (10-19 stage-1 smoke tests) have no backup copy;
     unreferenced and not on the old disk at its 10-22 audit: recorded as lost, not needed.
  2. 24 ledgered early shards (10-19, 8 stage-2 training plates x 3 backbones, 100 wells, no
     fingerprint) were not on the old disk when the caches were built (572 shards per
     backbone; 10-22 audit 1,770 = 572 x 3 + 54). With them, `cache.build` fails, and they would
     have replaced stage-2 latents of 58–100 wells per plate. Kept aside in
     `data/latents_unused_1019/`, not used.
  Ledger rows are never edited; two `#` reconciliation notes were appended.
- **Caches reproduced exactly:** rebuilt for all three backbones; `stats.json` byte-identical to
  `results/a2/cache_stats_*.json` and every cache array byte-identical to the restored cache.
  V-JEPA 2 weights hash matches `manifests/models.yaml`.
- **Shard guard (author instruction):** `manifests/cache_shards.tsv` (1,770 shards with SHA-256);
  `cache.build` and `scripts/70_backup_audit.py` refuse, listing paths, on a shard not in the
  manifest or a listed shard missing; `tests/test_cache_guard.py`.
- **Runner (operational only):** per-GPU slots (`CUDA_VISIBLE_DEVICES` per job, 4 units per GPU,
  22 M = 1, 50 M = 2; job on the GPU with most free units); the trainer lock records the GPU;
  `--check-inputs` dry run (12 of 12 families load; train/validation well counts identical to
  the completed runs). Completed runs are skipped, never rerun or overwritten.
- **S5 additions before any evaluation (`e0de078`):** F6 well-weighted H1 test (sign flip over
  plates of the well-weighted mean difference; secondary field in every H1 comparison and in
  `G_Signal_2.secondary_well_weighted`; pass rule unchanged; `tests/test_well_weighted.py`);
  the pipeline backs up every new results file with SHA-256 before each commit.
- **S2:** 1,083 failed wells re-requested frame by frame (100,719 HEAD requests, 3 attempts,
  back-off): 21,184 frames missing, no other status, none recovered. 214–215 wells miss every
  frame; 868 (532 training, 336 validation) miss only frames >= 88 (F9). Validation loses 338
  wells, 33 H1 primary, 70 primary-gene.
- **S3:** protocol `missing_frames_rule`: any frame missing after 3 attempts -> well excluded
  everywhere; embed failure lists committed and backed up before any evaluation; report
  counts and per-class tests. The fetcher now retries HTTP 404 up to 3 times
  (`MAX_404_ATTEMPTS`; previously final after 1), which applies only to embeds from now on
  (test, screenB); `tests/test_fetch_retry.py`. Tests: 11 passed.

### Incidents and own errors

- **Incident 3 (06:20 UTC):** unattended-upgrades (libc6) restarted `nvidia-fabricmanager` and
  `nvidia-persistenced`; all CUDA contexts torn down (Xid 45 on all GPUs, no hardware error); 27
  in-flight grid runs failed and rerun from scratch; no completed run affected. Evidence
  `logs/migration/incident_1023_nvidia_restart.log`; details in `reports/AMEND02_PROGRESS.md`.
- **Lost runner log (my error):** a `git stash`/`stash pop` at ~05:45 UTC replaced
  `logs/a2_val/grid.log` under the running grid runner; its lines from 05:36 to 14:26 UTC are
  lost. Per-run logs and results files are intact.

### Points for the reviewer

1. **Shard manifest at the freeze.** The author asked that test and screenB shards be added to
   the manifest *in the freeze commit*, but they are embedded after the freeze and
   `run_stage1_test.sh` builds the cache right after embedding, which the guard would block.
   Proposed: a commit right after embedding and before any evaluation that adds them to the
   manifest with SHA-256 and records the embed failure lists (author decision pending).
2. **OS upgrades.** The next `apt-daily-upgrade` is 10-24 06:55 UTC, likely during the
   evaluations or the test run; disabling the apt timers until the work is done is proposed
   (author decision pending).
3. **S3 test specification** (the plan says "a Fisher or chi-square test"): fixed as two-sided
   Fisher exact tests per class (mitotic delay, polylobed, binuclear, any reproducible) on
   library wells, classes from training-plate annotations, BH over the 4; descriptive only.
4. **F6 implementation:** plate signs flip, weights = primary wells per plate; equals the
   unweighted test when plates are equal-sized (unit test).
5. **Validation evaluations run on one GPU** (as committed in `run_a2_val.sh`); only training
   uses all 8.

### Files new since 10-22

`ENVIRONMENT.md`, `requirements-lock.txt`; `scripts/71_restore.py`, `scripts/72_missing_frames.py`;
`results/a2/restore_audit.json`, `results/a2/missing_frames.json`, `results/a2/backup_audit.json`,
`results/a2/grid_status_at_migration.json`; `manifests/cache_shards.tsv`;
`tests/test_cache_guard.py`, `tests/test_well_weighted.py`, `tests/test_fetch_retry.py`;
`logs/migration/`, `logs/restore.log`, `logs/check_inputs_grid.log`, `logs/missing_frames.log`.

---

# Packet as of 2026-10-22 07:45 UTC

For review by the orchestrating agent. Covers everything done under `docs/DESIGN_AMENDMENT_02.md`
(design commit `dd57079`) and the author's instructions of 2026-10-20 and 2026-10-21, up to
the current state. Branch `stage1` of `rafidquadir-eng/GeneWorld`; `main` is untouched at
`77ab684`. **No amended gate has been computed. No test plate, test gene or screenB well has been
evaluated.** This packet reports only summary statistics of the training runs, no per-family
validation comparisons, so nothing in it anticipates a gate.

## 1. Status at a glance

| step (author's list) | status | commits |
|---|---|---|
| 1. Provenance fix (section 9) | done | `63613c0` |
| 2. Freeze before training (sections 2, 6) | done | `6716199` |
| 3. Stage 2 (section 7) | done (60,289 of 61,372 wells) | `fa23c6f`, `b88f724`, `d5a5537` |
| 4. Per-plate control normalisation; plate probe | done | `8ea9c67` (+ diagnostics `445f83c`) |
| 5. Grid x 3 seeds; replicate baselines; probe CIs | running: grid 52 of 288 runs done | code `1511151`, fixes `c03b433`, `77c1905` |
| 6. Gates on validation, committed before step 7 | pending (automatic after step 5) | collector `scripts/65_gates_a2.py` |
| 7. Freeze; embed test plates; test once; screenB | prepared, not run | `cf3db6e`, `445f83c` |
| 8. Run report; GPU idle | pending | |

Unattended pipeline: `scripts/run_a2_val.sh` (PID 587473, resumed at step 5) runs the grid,
selection, H2 validation (planning source), secondary runs, every validation evaluation and the
gate collector, committing and pushing after each step. Step 7 is run by hand afterwards with
`scripts/run_stage1_test.sh`, which refuses to start without a clean, pushed, non-main freeze
commit and recorded validation gates, and refuses if any test result already exists.

## 2. Instructions and how each was implemented

### Amendment 02

| section | implementation | where |
|---|---|---|
| 1 per-plate control normalisation | per plate, clip and dimension: mean and SD over negative-control wells of the tile-mean latent; SD floored at the 5th percentile over all plates and clips for that dimension; plates with < 4 controls dropped (none were). Global standardisation kept as the sensitivity analysis; both preprocessings use the same wells | `src/geneworld/cache.py` |
| 2 primary horizons and wells | 24 h and 36 h = clips 8 and 12 from clip 0. H1 primary wells: held-out-plate wells whose siRNA carries a reproducible-phenotype term on training plates (training-plate rows only). H2/H3 primary genes: any reproducible term | `scripts/15_amend02_freeze.py`, `splits/amend02_*` |
| 3 replicate drift / endpoint | siRNA's training-plate wells, else the gene's other siRNAs, else plate control drift / training clip mean; fallback counts recorded. Averaged over replicate wells and their 4 tiles (see 4.4) | `src/geneworld/eval/a2eval.py` |
| 4 gates | G-Signal-2, G-ZeroShot-2, G-Plan-2 exactly as tabled; errors averaged over the three seeds' predictions before plate- or gene-level tests | `scripts/61-63_*_a2.py`, `65_gates_a2.py` |
| 5 training | grid {22 M, 50 M} x dropout {0.1, 0.3} x weight decay {0.05, 0.2}, 3 seeds, early stopping on validation primary error every 250 steps; selection = lowest mean over seeds per family, written before any gate | `scripts/50_train_a2.py`, `60_select_a2.py`, `run_a2_jobs.py` |
| 6 planning truth | primary = screenA reproducible terms (frozen with SHA-256); stringent = amendment-01 validated hits; nulls unchanged plus replicate-endpoint retrieval | `scripts/63_plan_a2.py` |
| 7 stage 2 | every library series, training and validation plates only, same-wells design, per-plate budget 190 (smallest step-10 budget reaching >= 60,000); test plates not embedded | `scripts/16_stage2_wells.py` |
| 8 test once | extended `run_stage1_test.sh`: part A stage-1 models (as registered), part B amended evaluations and gates, part C screenB | `scripts/run_stage1_test.sh` |
| 9 provenance | backbone revision and weights SHA-256 verified on disk against `manifests/models.yaml` in every result; 28 stage-1 files backfilled (provenance fields only; a leaf-by-leaf comparison showed no value changed); phenotype probe reports positive/negative counts and cluster-bootstrap CIs over plates | `src/geneworld/eval/provenance.py`, `scripts/14_backfill_provenance.py` |

### Author decisions

| date | decision | record |
|---|---|---|
| 10-20 | stage-2 fetch from the BIA bucket only (IDR holds one MitoCheck plate) | `manifests/amend02_protocol.json` |
| 10-20 | shuffled control = trained shuffled-encoding models, 3 seeds per source (permutation seed = model seed); evaluation-time shuffle reported as secondary | protocol |
| 10-20 | every gated family selects its own grid configuration | protocol |
| 10-20 | H1 action-conditioned model = learned per-gene embedding | protocol |
| 10-21 | screenB after the freeze, global preprocessing only, outside the gates; H1 (seen genes) and H2 (test-split genes); siRNA token masked; reported next to the global test results | protocol, `scripts/19_screenB_wells.py`, `manifests/amend02_screenB.json` |
| 10-21 | plate-probe diagnostics (a) within-series replicates, (b) negative controls with leave-one-out statistics; section 1 unchanged | `scripts/53_plate_probe.py` |
| 10-21 | run report to list the 20 primary test wells outside the layout and the data behind the tile-averaging choice | protocol `record_for_run_report` |

## 3. Results so far (no gates)

### Freeze (step 2), counted before any training

| list | validation | test |
|---|---|---|
| H1 primary wells | 535 | 1,112 |
| H2/H3 primary genes (gene split) | 204 | 417 |
| truth genes, mitotic delay / polylobed / binuclear (gene split) | 42 / 35 / 53 | 84 / 62 / 105 |

All 535 validation primary wells are in the stage-1 + stage-2 layouts; 1,092 of 1,112 test
primary wells are in the test layout. The other 20 are on LT0008_17 (17) and LT0155_08 (3),
plates absent from the S-BIAD865 S3 conversion.

### Stage 2 (step 3)

60,289 of 61,372 wells on 404 of 410 plates (54,052 training-plate and 7,320 validation-plate
wells selected). Every one of the 1,083 failures is an HTTP 404 for frames missing from the S3
conversion (movies shorter than 93 frames, mostly frames 89–92). Six plates fail entirely
(LT0003_40, LT0008_14, LT0012_29, LT0079_38, LT0084_46, LT0096_29: 4 training, 2 validation);
24 others lose 1–3 wells. All 1,212 shards are backed up with SHA-256.

### Plate-identity probes (step 4), clip 12 (36 h)

| probe | chance | global | normalised | centre-only |
|---|---|---|---|---|
| main: 49 validation plates, 8,574 wells (one plate per series) | 2.0% | 74.4% | 93.9% | 92.9% |
| (a) within-series replicates: 76 training series, CV grouped by well position | 31.0% | 95.1% | 94.4% | 94.3% |
| (b) negative controls only (369 wells), leave-one-out statistics | 2.0% | 50.9% | 0.0% | 0.0% |

The main probe confounds batch with gene content (one validation plate per series). (a) shows
replicate plates remain about 95% distinguishable under every version; normalisation lowers it
by under 1 point. (b) shows the plate offset in control wells is removed (below chance is the
expected result of leave-one-out centring once no offset remains). My first reading of the
main probe (a control-mean estimation offset) was withdrawn on 10-21.

### Training runs (step 5, in progress)

32 of 288 grid runs complete, 256 queued, 0 failures since the restart. 22 M runs: median 34
min with 3 concurrent, peak GPU memory up to 8.5 GB, median best step 2,250, median stop
5,250 (early stopping fires well before the 12,000-step cap). One 50 M run complete (62 min).

## 4. Deviations, interpretations and incidents (for review)

1. **IDR bucket.** Section 7's two-bucket split is impossible: IDR holds only LT0001_02, and its
   level-1 frames are not byte-identical to BIA's (correlation 0.974; different downsampling).
   My earlier `DATA_ACCESS.md` claim that IDR carried the same data was wrong and is corrected.
   The author chose BIA only. The native pycurl fetcher gives 4.6 wells/s, about 8x what the
   GPU embeds, so run time is unaffected.
2. **Underspecified values fixed before training** (protocol): 22 M = width 512 x depth 6 and
   50 M = width 768 x depth 7 (transformer core, excluding vocabulary tables); rollout 6 (the
   stage-1 validation selection); lr 3e-4; batch 256; at most 12,000 steps; patience 3,000.
   The planning-source rule and the secondary analyses' configurations (each family's selected
   configuration, 3 seeds) were also fixed before training.
3. **Shuffled control and grid scope** were author decisions (section 2).
4. **Replicate baselines averaged over tiles.** On development data, matching replicate tile k
   to evaluated tile k made replicate drift worse than copy-last (primary error 7.62 vs 5.54);
   averaged over tiles, 5.72. Tile positions are arbitrary fields of view, and the predictor
   and the other drift baselines already treat them as exchangeable. Data used: a throwaway
   cache of stage-1 latents only (test plates excluded), development models (22 M, 250 steps,
   3 seeds, deleted), the stage-1 validation plates' 104 primary wells on 13 plates. Applied to
   the nearest-gene lookup too. Recorded in the protocol before any real training.
5. **Grid incident.** First grid attempt: 277 of 288 runs failed with CUDA out-of-memory (4
   processes on 40 GB; up to 16 GB each, mainly validation evaluation in chunks of 1,024 wells
   on the 50 M model). Fix: chunks of 256 wells, expandable CUDA segments, 3 workers; peak
   memory now recorded per run. No protocol value changed; chunk size changes results only
   through bf16 rounding. The 11 completed runs are kept. A second incident (10-22 07:30 UTC)
   hit when the 50 M configurations started: they need ~14.6 GB in training itself, so three at
   once overflowed and 12 runs failed (they rerun). The runner now schedules by GPU memory
   (22 M = 1 unit, 50 M = 2, capacity 4) and counts runs holding a live lock (`77c1905`).
6. **Stage-2 data loss.** 6 plates (2 validation) and 1,083 wells, all from frames missing on
   S3. Validation plates available for H1: 49 of 51.
7. **ScreenB** has no negative-control wells (only empty wells and 155 transfection controls),
   so section 1 cannot apply. Per the author, it is evaluated only under the global
   preprocessing, outside the gates. Its primary-well rule is the gene-level analogue of the
   frozen H1 rule (screenB siRNAs never occur in screenA): 7,622 of 9,151 screenB wells, frozen
   with SHA-256 before the freeze.
8. **Stage-1 scripts keep `design_commit` = `deb443d`.** Step 1 briefly set them to
   `dd57079`; this was reverted before any test, so the stage-1 test results in part A will
   carry the pre-amendment design commit.
9. **Order inside the validation pipeline.** H2 validation runs before the secondary training
   because the complex-split runs need the planning source, which comes from H2 validation.
   Selection is committed before H2 validation.
10. **Own earlier errors, already disclosed to the author:** stage-1 milestones were pushed to
    GeneWorld `main` before the branch rule was applied to this repository (`main` frozen at
    `77ab684` since); the IDR claim in item 1.

## 5. Points worth the reviewer's attention

- **Same validation set for selection and gates.** Early stopping and configuration selection
  use the validation primary error, which is also the H1 G-Signal-2 metric (and the H2 analogue).
  This follows the amendment (gates on validation, then test once) but means the validation
  gates are optimistic; the test evaluation is the unbiased one.
- **Batch effect survives normalisation** (probe (a)). Section 1 is unchanged by design (one
  round, no further amendment). The global sensitivity analysis is the comparison point.
- **Trajectory-planning rule** compares the planner with replicate-endpoint retrieval, which
  uses the unseen gene's own observed wells (an oracle). The gate therefore asks the world model
  to beat a baseline that sees the target gene's data.
- **Phenotype-planning replicate-endpoint retrieval** exists only for genes with embedded
  wells and is reported on that subset, with the planner scored on the same subset.
- **Compute.** 22 M runs take ~34 min (3 at a time), 50 M runs ~50–62 min (2 at a time). The
  remaining ~236 runs need about 72–87 h, so the grid should finish around Oct 25 (morning to
  evening UTC); then secondary runs and validation evaluation (~0.5 day), the freeze, ~11 h of
  test-plate and screenB embedding and the test run. Final report about Oct 27–28. (Earlier
  estimates of Oct 24 and Oct 25 morning were too optimistic.)

## 6. File index

- Design and rules: `docs/DESIGN_AMENDMENT_02.md`, `docs/DESIGN_AMENDMENT_01.md`,
  `docs/tijepa_rules/RULES.md`
- Protocol (all decisions and fixed values): `manifests/amend02_protocol.json`
- Freeze: `manifests/amend02_freeze.json`, `splits/amend02_h1_primary_wells.tsv`,
  `splits/amend02_h2_primary_genes.tsv`, `manifests/amend02_truth/`
- Stage 2: `splits/stage2_wells.tsv`, `splits/stage2_layout.tsv`, `splits/stage2_summary.json`,
  `results/a2/stage2_data.json`
- Test and screenB lists: `splits/stage2_test_wells.tsv`, `splits/test_wells_to_embed.tsv`,
  `splits/screenB_wells.tsv`, `splits/amend02_screenB_primary_wells.tsv`,
  `manifests/amend02_screenB.json`, `manifests/amend02_primary_test_wells_outside_layout.tsv`
- Probes: `results/a2/plate_probe.json`, cache statistics `results/a2/cache_stats_*.json`
- Training summaries: `results/a2/train/*.json` (one per completed run)
- Progress: `reports/AMEND02_PROGRESS.md`; data access: `reports/DATA_ACCESS.md`;
  stage 1: `reports/RUN_REPORT_STAGE1.md`
- Code: `src/geneworld/{cache,annotations}.py`, `src/geneworld/train/a2.py`,
  `src/geneworld/eval/a2eval.py`, `src/geneworld/io/s3fetch.py`, `scripts/1[4-9]_*.py`,
  `scripts/5[0-3]_*.py`, `scripts/6[0-5]_*.py`, `scripts/run_a2_{val,jobs}.*`,
  `scripts/run_stage1_test.sh`
- Backups: `~/LambdaFilesystem/geneworld_backups/` (ledger `reports/BACKUP_LEDGER.tsv`)
