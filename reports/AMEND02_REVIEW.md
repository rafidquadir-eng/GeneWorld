# GeneWorld amendment 02: review packet (as of 2026-10-22 03:40 UTC)

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
| 5. Grid x 3 seeds; replicate baselines; probe CIs | running: grid 32 of 288 runs done | code `1511151`, fix `c03b433` |
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
   through bf16 rounding. The 11 completed runs are kept.
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
- **Compute.** Grid throughput is about 5 runs/h on 22 M configurations and lower on 50 M. The
  grid should finish around Oct 25 morning UTC, then secondary runs and validation evaluation
  (~0.5 day), then the freeze, ~11 h of test-plate and screenB embedding and the test run. The
  final report is expected about Oct 26–27. (The 10-21 estimate of Oct 24 midday for the grid
  was too optimistic.)

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
