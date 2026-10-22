# Amendment 02: completion plan, v2 (author review, 2026-10-22)

Review of `stage1` up to `b2af5d9` (grid 52 of 288 runs done). Supersedes v1 of this file
(`c80b229`), whose findings F1–F7 and steps are kept. New in v2: **F8**, an error in the
G-Plan-2 trajectory rule, corrected here before any H3 or gate output exists; **F9**,
missing-frame scope; **F10**, migration to an 8-GPU instance (author decision: see section 3).

At the time of this review, `results/a2/` on `stage1` holds only cache statistics, plate
probes, stage-2 data and 52 training-run summaries: no `selection.json`, no H2/H3 validation
output and no gates file. The reviewer did not read per-family validation errors in the
training summaries.

Nothing here changes the preprocessing used by the gates, a primary metric, a threshold, the
frozen splits or the grid. F8 corrects a comparator that made one part of one gate unpassable
by construction; it is recorded as an erratum, not a tuning change.

## 1. Findings

| # | finding | severity | action |
|---|---|---|---|
| F1 | Replicate plates of a series are ~95% decodable (chance 31%) under every preprocessing; section 1 removes the control-well offset (control-only probe 50.9% -> 0%) but not the batch signature in library wells. | interpretation | No gate change. Forecasts start from the well's own clip-0 state and replicate drift uses changes, so the comparisons are paired within plate. Report as finding R1; exploratory E1/E2 after the test. |
| F2 | Probe (b) is 0% by construction after leave-one-out centring. | minor | Report (b) as uninformative after normalisation; quote only its global value (50.9%). |
| F3 | 1,083 wells (1.8%) not embedded (frames missing on S3): 6 whole plates (2 validation), 33 of 535 primary validation wells. | moderate | Keep train/validation as embedded. Confirm permanence (S2); pre-register the same exclusion for test and screenB (S3). |
| F4 | OOM fixes changed only evaluation chunking, worker count and the allocator. | none | Record in the run report. |
| F5 | Remaining compute ~86 h for the grid on one A100-40GB (the 142 remaining 50 M runs dominate), then ~0.5 day of secondary runs and validation evaluations, then ~11 h of test and screenB embedding. Report ~Oct 27–28. | schedule | Migration (F10). |
| F6 | The H1 gate weights plates with one primary well like plates with eight. | minor | Unchanged (pre-specified). Report wells per plate and a well-weighted secondary test. |
| F7 | 20 frozen primary test wells on LT0008_17 and LT0155_08, plates absent from S3. | minor | Report (already recorded). |
| **F8** | **G-Plan-2 trajectory rule uses an oracle comparator.** Amendment 02 section 4 requires the planner's hit@100 to beat *replicate-endpoint retrieval*. For unseen genes that baseline ranks candidates by the observed clip-12 latents of each candidate gene's own wells, i.e. the unseen gene's real data, which the planner never receives (stage 1: oracle hit@100 0.148 vs planner 0.034). The rule asks a zero-shot model to beat a method that sees the answer's replicates, so it cannot test planning. The error is in the amendment text (author side), not in the implementation, which follows the text. | **must fix before gates** | **Erratum E-02.1 (S1).** Gate comparator = kNN-imputed endpoint retrieval (endpoint imputed from the k nearest *training* genes in the planning source's encoding space, as in stage 1, using no data from the unseen gene). Replicate-endpoint retrieval stays in every table, labelled "oracle ceiling (uses the unseen gene's observed wells)". The phenotype-planning half of G-Plan-2 is unchanged. |
| **F9** | Most missing frames are 88–92, which only clip 13 (frames 77–92) uses; the primary horizons use clips 0, 8 and 12 (frames 0–87). Those wells could have been embedded with clip 13 masked. | minor | No change: the grid is training on the current data, and test and screenB must follow the same rule (S3). Count, in the run report, how many excluded wells were missing only frames >= 88; a partial-movie embedding is listed under "not run". |
| **F10** | The grid is memory-bound on one 40 GB GPU (capacity 4 units: three 22 M or two 50 M runs at once). An 8 x A100-40GB instance runs the same per-GPU schedule eight times over: grid ~11 h, test + screenB embedding ~1.5 h (8 embedders at 0.59 wells/s together reach the fetcher's measured ~4.6 wells/s). Lambda on-demand list prices (2026-10-22): 1 x A100-40GB $1.29/h; 8 x A100-40GB $10.32/h. Remaining work on one GPU ~110 h ≈ $140; on eight ~16 h ≈ $165 plus ~3 h of migration. Same cost within ~20%, about 4 days sooner. | schedule | Migrate if the author launches the instance (section 3). |

## 2. Steps (in order)

**S1. Erratum E-02.1 (now; commit and push before anything else).**
1. Verify and record in the commit message that `stage1` has no `results/a2/selection.json`,
   no H2/H3 validation output and no `results/a2/gates_val.json`. If any exists, stop and
   report to the author instead of applying S1.
2. Append section "10. Erratum E-02.1 (2026-10-22, before any gate)" to
   `docs/DESIGN_AMENDMENT_02.md` with the F8 text and the corrected rule: *trajectory
   planning passes if hit@100 is above random (one-sided binomial p < 0.05) and above
   kNN-imputed endpoint retrieval (bootstrap CI of the difference over targets above 0).*
3. Change `scripts/63_plan_a2.py` (the `planner_vs_..._hit@100` comparison and the rule text)
   and `scripts/65_gates_a2.py` accordingly; keep `replicate_endpoint_retrieval` computed and
   reported, renamed in output tables as the oracle ceiling. Add a unit check that the gate
   field references `retrieval_knn`.
4. Record the erratum in `manifests/amend02_protocol.json` (`errata` list, with date and
   commit) and in `reports/AMEND02_PROGRESS.md`.

**S2. Confirm the missing frames (CPU only; does not touch the grid).** Re-request every
failed frame with 3 attempts and back-off. Write `results/a2/missing_frames.json`: per failed
well, plate, split, gene, primary flag, training-plate phenotype classes, and missing frame
indices; flag wells whose missing frames are all >= 88 (F9). Recovered frames are listed only;
nothing is added to train or validation.

**S3. Pre-register missing-frame handling for test and screenB (before the freeze).** In the
protocol: a well with any frame missing after 3 attempts is excluded from every analysis, as
in train and validation. The run report gives, for test and screenB, the wells, plates, primary
wells and primary-gene wells lost, and a Fisher or chi-square test that loss is unrelated to
phenotype class.

**S4. Migration, only if the author has launched the new instance (section 3).**

**S5. Finish steps 5 and 6 as committed.** Grid, then `results/a2/selection.json` committed
before any gate, H2 validation, secondary runs, validation evaluations, then
`results/a2/gates_val.json` committed and pushed **before** the freeze. Add the
well-weighted H1 test (F6) as a secondary field.

**S6. Step 7 as committed.** Freeze (clean, pushed, non-main commit), embed the test plates and
screenB (one embedder per GPU, BIA fetchers sharing at most 128 streams in total), and run
`scripts/run_stage1_test.sh` once: Part A test (normalised, gates), Part B global
sensitivity, Part C screenB (global only, outside the gates, siRNA token masked).

**S7. Exploratory, only after S6, never used to revise a gate or claim.**
- **E1, spatial batch correction:** per plate and clip, subtract a row/column median polish of
  the library wells' latents (B-score style) after section 1. Re-run probes (main and (a)).
  Retrain `h1_learned` and `h1_none` with their selected configurations, 3 seeds; evaluate H1
  on validation and test, with replicate drift recomputed in that space.
- **E2, batch carried by the forecast:** within-series replicate probe on forecast vs observed
  clip-12 latents.
- Every number is labelled "exploratory, after the test".

**S8. Step 8: `reports/RUN_REPORT_STAGE2.md`.** It covers:
- every gate on validation and on test;
- all baselines, including the oracle ceiling, and all controls;
- normalised versus global results;
- screenB;
- the plate-probe table, with (b) labelled uninformative;
- R1;
- missing-frame accounting (S2, S3, F9);
- the incidents (F4) and the migration, if done;
- deviations and errata: BIA-only fetch, tile-averaged replicate baselines and the data used
  to choose them, the 20 out-of-layout primary test wells, the screenB decision, E-02.1;
- E1/E2 in a separate "Exploratory, after the test" section;
- what was not run.

**S9. Tell the author when every GPU is idle and which instances can be terminated.**

## 3. Migration to 8 x A100-40GB (only when the author says the instance is up)

The new instance must be in the **same region as the persistent filesystem** holding
`~/LambdaFilesystem/geneworld_backups/`, with that filesystem attached. If no 8-GPU instance
is available in that region, the author decides between staying on one GPU or another size;
do not copy data between regions.

1. **Old instance:** stop launching new grid runs. Let in-flight runs finish if they end
   within ~1 h; otherwise stop them (a stopped run restarts from scratch and its partial
   output is discarded). Stop `scripts/run_a2_val.sh` cleanly. Commit and push. Check that every
   latent shard, cache file and completed run summary or checkpoint is in
   `reports/BACKUP_LEDGER.tsv` with SHA-256 and present on the persistent filesystem. Tell the
   author the old instance can be terminated.
2. **New instance:** clone `stage1` (with this plan merged), set up the environment from the
   repository's pinned requirements, restore latents and completed run outputs from the
   persistent filesystem, and verify every SHA-256 against the ledger. Rebuild the caches and
   check that `results/a2/cache_stats_*.json` are reproduced exactly; stop and report if not.
   Re-verify the V-JEPA 2 weights hash for the later test embedding.
3. **Runner:** give `scripts/run_a2_jobs.py` per-GPU slots (`CUDA_VISIBLE_DEVICES` per worker;
   memory capacity 4 units per 40 GB GPU, 22 M = 1, 50 M = 2). Operational only: no protocol
   value changes; completed runs are skipped and never rerun or overwritten.
4. Resume S5, then S6–S9. Record hardware, start/stop times and any run reruns in the report.

## 4. What the paper can say, depending on the test

- **R1 (any outcome):** a frozen V-JEPA 2 latent world model of genome-scale RNAi video, with
  the batch structure of the latent quantified (replicate plates ~95% decodable; control
  normalisation removes the control offset; the remaining signature sits in library wells).
- **G-Signal-2 passes on test:** action conditioning carries gene-specific dynamics beyond
  replicate memorisation, which is the core world-model claim; H2/H3 then read as confirmatory
  or negative on their own gates.
- **G-Signal-2 fails on test:** a negative result under fair metrics and a memorisation
  baseline. GeneWorld stops; DrugWorld (design section F) is next.
