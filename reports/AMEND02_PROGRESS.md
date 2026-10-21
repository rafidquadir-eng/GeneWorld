# Amendment 02: progress (updated 2026-10-21 23:40 UTC)

Interim status while the amendment-02 round runs (design commit `dd57079`). Final results go in
`reports/RUN_REPORT_STAGE2.md`. No amended model has been trained and no test plate, test gene
or screenB well has been evaluated.

| step | status | evidence |
|---|---|---|
| 1. Provenance (section 9) | done | V-JEPA 2 revision and weights SHA-256, verified on disk, in every results file; 28 stage-1 files backfilled, no value changed (`63613c0`) |
| 2. Freeze before training (sections 2, 6) | done | `splits/amend02_h1_primary_wells.tsv` (535 val / 1,112 test), `splits/amend02_h2_primary_genes.tsv` (204 / 417), truth lists with SHA-256 and unseen counts in `manifests/amend02_freeze.json` (`6716199`) |
| 3. Stage 2 (section 7) | done 20:21 UTC | 60,289 of 61,372 wells on 404 of 410 plates; 1,212 shards backed up with SHA-256 (`results/a2/stage2_data.json`). All 1,083 failures are HTTP 404 for frames missing from the S3 conversion (movies shorter than 93 frames; mostly frames 89–92): 6 plates entirely (LT0003_40, LT0008_14, LT0012_29, LT0079_38, LT0084_46, LT0096_29; 4 training, 2 validation) and 1–3 wells on 24 others |
| 4. Per-plate control normalisation and plate probe | done 20:57 UTC | no plate dropped (4–8 negative-control wells per plate, median 8); probes below (`results/a2/plate_probe.json`) |
| 5. Grid x 3 seeds, replicate baselines, probe CIs | running | first attempt: 277 of 288 runs failed with CUDA out-of-memory (4 concurrent processes on 40 GB); 11 completed and are kept. Operational fix (evaluation chunks of 256 wells, expandable CUDA segments, 3 workers; no protocol value changed, `c03b433`); 277 runs restarted 23:17 UTC, GPU memory stable at 26.8 GB |
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

## Author decisions 2026-10-21

- ScreenB: evaluated after the freeze under the global preprocessing only, outside the gates
  (H1 and H2, siRNA token masked); reported next to the global test results.
- Plate-probe diagnostics (a) and (b) added; section 1 unchanged.

## Timeline (revised)

The grid takes longer than estimated (22 M runs ~25 min, 50 M runs ~55 min with 3 concurrent):
grid ~2.5 days (to about Oct 24 midday), secondary runs and validation evaluations ~0.5 day,
then the freeze, test-plate and screenB embedding (~11 h) and the test run. Run report and
GPU idle about Oct 25–26.
