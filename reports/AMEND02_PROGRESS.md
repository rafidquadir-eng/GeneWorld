# Amendment 02: progress (2026-10-21 16:30 UTC)

Interim status while the amendment-02 round runs (design commit `dd57079`). Final results go in
`reports/RUN_REPORT_STAGE2.md`. No amended model has been trained and no test plate, test gene
or screenB well has been evaluated.

| step | status | evidence |
|---|---|---|
| 1. Provenance (section 9) | done | V-JEPA 2 revision and weights SHA-256, verified on disk, in every results file; 28 stage-1 files backfilled, no value changed (`63613c0`) |
| 2. Freeze before training (sections 2, 6) | done | `splits/amend02_h1_primary_wells.tsv` (535 val / 1,112 test), `splits/amend02_h2_primary_genes.tsv` (204 / 417), truth lists with SHA-256 and unseen counts in `manifests/amend02_freeze.json` (`6716199`) |
| 3. Stage 2 (section 7) | running, 85% | 51,893 of 61,372 wells on 351 of 410 train/validation plates, 0.59 wells/s, ~4.5 h left; 1,053 shards all backed up with SHA-256; 33 wells missing so far on 22 plates (expected 51,926 on those plates) |
| 4. Per-plate control normalisation and plate probe | queued (automatic) | code `src/geneworld/cache.py`, `scripts/53_plate_probe.py` |
| 5. Grid x 3 seeds, replicate baselines, probe CIs | queued (automatic) | 288 grid runs, then secondary runs and validation evaluations (`scripts/run_a2_val.sh`) |
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

## Open question for the author

- **ScreenB has no negative-control wells** (only empty wells and 155 transfection controls),
  so the section 1 normalisation would drop every screenB plate. Options: skip screenB, or
  evaluate it under the global preprocessing only, outside the gates. Needed before the freeze.
