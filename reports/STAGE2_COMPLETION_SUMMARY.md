# GeneWorld amendment 02: completion summary (2026-10-27)

Every amendment-02 step is done: grid, validation, gates, freeze, one-time test, screenB and the
exploratory analyses E1-E3. All GPU work has finished; the instance can be terminated. The full
numbers are in `reports/RUN_REPORT_STAGE2.md` (generated from the results files by
`scripts/90_run_report_stage2.py`). The validation checklist is in
`reports/GATES_VAL_SUMMARY.md`. Every number below is copied from the results file named beside
it.

## Gates

| gate | validation (`results/a2/gates_val.json`) | test, once (`results/a2/gates_test.json`) |
|---|---|---|
| G-Signal-2 | pass: 40 plates; vs action-agnostic −6.2% (p = 5e-05), vs replicate drift −46.9% (p = 1e-05) | pass: 82 plates, 770 primary wells; vs action-agnostic −5.65% (p = 1e-05), vs replicate drift −45.9% (p = 1e-05) |
| G-ZeroShot-2 | fail: no source passes | pass, string only: true − shuffled −0.0088, 95% CI [−0.0176, −0.0001]; true − agnostic −0.039 [−0.0507, −0.0276] (396 genes) |
| G-Plan-2 | fail: planner − retrieval_knn hit@100 −0.016 [−0.050, 0.015]; oracle ceiling 0.169 | fail: planner − retrieval_knn −0.012 [−0.037, 0.010]; oracle ceiling 0.162 |

Points for interpretation (author's judgment):

- **G-ZeroShot-2 on test is narrow.** It passes on one source of five. The upper bound of the
  string CI is −0.0001, and string fails the same rule on validation. The other four sources'
  true − shuffled CIs include 0 on test. String is not the planning source (`jump_crispr`, fixed
  on validation before the test), so H3 does not use it.
- **G-Signal-2 holds on validation, test and every validation sensitivity analysis except
  TiJepa-1.** The global preprocessing and DINOv2 beat both baselines on validation. TiJepa-1
  does not beat action-agnostic (−2.1%, p = 0.082; `results/a2/h1/h1_val_plate_tijepa1.json`).
- **G-Plan-2 fails on both splits.** The planner is above random on validation (binomial
  p = 0.018). It is not distinguishable from kNN-imputed endpoint retrieval (erratum E-02.1).

## ScreenB (global preprocessing, outside the gates)

`results/a2/h1/h1_screenB_global_vjepa2.json`, 28 plates. Against action-agnostic: −0.76%,
p = 0.102 (does not beat). Against replicate drift: −36.6%, p = 1e-05 (beats). The G-Signal-2
rule therefore fails on screenB. Embedding: 9,151 wells, 24 listed as failed
(`results/a2/embed_failures_screenB.json`).

## Exploratory, after the test (never used to revise a gate or claim)

| analysis | result | file |
|---|---|---|
| E1 B-score spatial correction | G-Signal-2 comparisons still beat both baselines. Validation: −5.28% vs agnostic, −46.5% vs drift. Test: −3.78% and −45.1%. All p = 1e-05. | `results/a2/exploratory/e1/h1/` |
| E2 batch in the forecast | Replicate-plate probe accuracy over 76 series (chance 0.31). Observed clip 12: 0.944. Observed clip 0: 0.942. Learned forecast: 0.95. Agnostic forecast: 0.952. | `results/a2/exploratory/e2.json` |
| E3 wells missing only frames ≥ 88 | G-Signal-2 conclusion unchanged. Validation primary wells 362 → 386. Test: 0 candidate wells (770 → 770). | `results/a2/exploratory/e3/compare.json` |

## Order of the test (completion plan v4, D2)

| step | commit | time (UTC, 10-26) |
|---|---|---|
| S13 checklist, then the author's go | `ce5c3df` (freeze) | 06:45; go received before 13:02 |
| test embed: 14,186 wells on 100 plates, 4 failed; allowed-paths check passed | `bc186bb` | 13:02-13:58 |
| Part A (stage-1 models), Part B (amended evaluations and gates) | in `66696b4` | 13:58-20:00 |
| screenB embed; allowed-paths check passed | `f3a2c9d` | 20:00-20:44 |
| screenB H1/H2; test results committed | `66696b4` | 21:11 |
| E1-E3 (preparation from 13:59, evaluation only after the test) | `5a0f556`, `6733a43` | 21:11-23:50 |

The E1 models were trained on GPUs 2-7 while the test evaluation ran on GPU 0. Training used only
the training split and produced no exploratory number before the test finished.

## Incidents and deviations (whole amendment)

- **Grid** (288 of 288 runs complete, `results/a2/grid_runs.json`). Failed attempts by incident:
  - 1: CUDA out-of-memory, 10-21, 277 attempts;
  - 2: out-of-memory, 10-22, 13 attempts;
  - 3: an unattended OS upgrade restarted the NVIDIA services, 10-23, 27 attempts.

  Every failed run was rerun from scratch. The runner log for 10-23 05:36-14:26 UTC was lost
  (G3); the affected attempts are reconstructed and labelled in the grid record.
- **S12 orchestrator stopped twice at the grid record**, before any selection. The cause was an
  undated runner section, fixed in `7856987`. The GPUs were idle for 54.5 h (10-23 16:27 to 10-25
  22:55 UTC) while the session was paused.
- **Parallelisation** (author, 10-25):
  - the 9 complex-split secondary runs were trained early by a complex-only runner;
  - S18 ran beside S12.

  Configurations, seeds and inputs were unchanged.
- **Failed steps, each rerun with exit 0:**
  - the S18 `h2_global` step: its first run came before the global models existed;
  - E2: a read-only array in `explore.within_series_probe`, fixed in `5a0f556` with the same
    mask.

  The test evaluation ran its 36 steps with no failure.
- No gate, threshold, metric, preprocessing step, split, grid value, selection or frozen list
  was changed.

## Open items for the author

1. **S13 item 7.** Two files name test-split genes, as annotation only:
   - `results/a2/missing_frames.json` (S2) labels failed wells on training and validation plates;
   - `results/a2/rehearsal/selection_of_wells.json` lists 18 training-plate wells of test-split
     genes. They were re-embedded only for the S18 byte-identity check.

   No evaluation used either.
2. **S13 item 9.** S18 passed after one rerun. `evaluation_all_ok` is false, as recorded;
   `evaluation_last_run_ok` is true.
3. **Manuscript.** `paper/geneworld/` still has placeholder results and discussion sections.
   `scripts/91_paper_numbers.py` can now fill `numbers.tex` from the final results files.
   Nothing has been submitted or posted.

## Backups

The persistent filesystem `~/LambdaFilesystem/geneworld_backups/` holds every new results file,
checkpoint and shard, each verified by SHA-256 and listed in `reports/BACKUP_LEDGER.tsv`:

- 300 test shards, 84 screenB shards and 18 E3 shards;
- the 6 E1 checkpoints;
- the test, screenB and exploratory results;
- the reports and figures.
