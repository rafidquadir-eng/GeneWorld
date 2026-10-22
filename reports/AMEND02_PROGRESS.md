# Amendment 02: progress (updated 2026-10-22 07:45 UTC)

Interim status while the amendment-02 round runs (design commit `dd57079`). Final results go in
`reports/RUN_REPORT_STAGE2.md`. No amended model has been trained and no test plate, test gene
or screenB well has been evaluated.

| step | status | evidence |
|---|---|---|
| 1. Provenance (section 9) | done | V-JEPA 2 revision and weights SHA-256, verified on disk, in every results file; 28 stage-1 files backfilled, no value changed (`63613c0`) |
| 2. Freeze before training (sections 2, 6) | done | `splits/amend02_h1_primary_wells.tsv` (535 val / 1,112 test), `splits/amend02_h2_primary_genes.tsv` (204 / 417), truth lists with SHA-256 and unseen counts in `manifests/amend02_freeze.json` (`6716199`) |
| 3. Stage 2 (section 7) | done 20:21 UTC | 60,289 of 61,372 wells on 404 of 410 plates; 1,212 shards backed up with SHA-256 (`results/a2/stage2_data.json`). All 1,083 failures are HTTP 404 for frames missing from the S3 conversion (movies shorter than 93 frames; mostly frames 89–92): 6 plates entirely (LT0003_40, LT0008_14, LT0012_29, LT0079_38, LT0084_46, LT0096_29; 4 training, 2 validation) and 1–3 wells on 24 others |
| 4. Per-plate control normalisation and plate probe | done 20:57 UTC | no plate dropped (4–8 negative-control wells per plate, median 8); probes below (`results/a2/plate_probe.json`) |
| 5. Grid x 3 seeds, replicate baselines, probe CIs | running | first attempt: 277 of 288 runs failed with CUDA out-of-memory (4 concurrent processes on 40 GB); 11 completed and are kept. Operational fix (evaluation chunks of 256 wells, expandable CUDA segments, 3 workers; no protocol value changed, `c03b433`); 277 runs restarted 23:17 UTC. Second incident 07:30 UTC 10-22: 50 M runs need ~14.6 GB in training, so three at once overflowed; 12 more runs failed (rerun). The runner now schedules by GPU memory (22 M = 1 unit, 50 M = 2, capacity 4) and respects live locks (`77c1905`); 52 of 288 runs done, 236 queued |
| 6. Gates on validation | queued (automatic) | `results/a2/gates_val.json`, committed before the freeze |
| 7. Freeze, embed test plates, test once | prepared | `splits/test_wells_to_embed.tsv` (14,186 wells, 100 plates); extended `scripts/run_stage1_test.sh` |
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

## Timeline (revised 10-22)

22 M runs take ~34 min (3 at a time) and 50 M runs ~50–62 min (2 at a time). The remaining
~94 22 M and ~142 50 M runs need about 72–87 h: grid to about Oct 25 morning–evening UTC, then
secondary runs and validation evaluations (~0.5 day), the freeze, test-plate and screenB
embedding (~11 h) and the test run. Run report and GPU idle about Oct 27–28.
