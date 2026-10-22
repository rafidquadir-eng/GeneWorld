# Amendment 02: completion plan (author review, 2026-10-22)

Review of `stage1` up to `5fef0d3`. Nothing here changes a gate, a threshold, a primary
metric, the preprocessing used by the gates, the frozen splits or the grid. Everything below is
either operational, a pre-registration of how already-known data problems are handled at the
test, or an analysis labelled **exploratory** that runs only after the one-time test.

## 1. Review findings

| # | finding | severity | action |
|---|---|---|---|
| F1 | Replicate plates of the same series are told apart in ~95% of cases (chance 31%) under all three preprocessings; section 1 removes the control-well offset (control-only probe 50.9% -> 0%) but not this. The batch signature lives in library wells, not in the control means: for example a per-plate spatial gradient, or transfection efficiency that varies between replicate plates and interacts with the siRNA. | important for interpretation; not a gate defect | No change to the gates: every forecast starts from the well's own clip-0 state, which carries its plate's signature, and replicate drift uses changes, not levels. Report it as a finding (section 4, R1). Exploratory analysis E1 after the test. |
| F2 | Diagnostic (b) (negative controls, leave-one-out statistics) is 0% by construction after centring, so it cannot show a remaining offset. | minor | Report (b) as uninformative by construction; quote only its global value (50.9%) as evidence that controls carry a plate offset before normalisation. |
| F3 | 1,083 wells (1.8%) not embedded: frames missing from the S3 conversion (HTTP 404, one attempt). 6 whole plates (4 train, 2 validation) and 1–3 wells on 24 plates. This removes 33 of the 535 frozen primary validation wells (6%). Test plates and screenB will have the same problem. | moderate | Keep the current train/validation data (the grid is running; missingness is technical and plate-level). Step A1 confirms the 404s are permanent. Step A2 pre-registers the same exclusion rule for test and screenB and the counts to report. |
| F4 | The OOM fix changed only evaluation batch size, the worker count and the CUDA allocator; the 11 completed runs are comparable with the restarted ones. | none | Record it in the run report. |
| F5 | Grid at ~2.5 days on one A100-40GB (3 concurrent runs, memory-bound), then secondary runs (~57 runs), test and screenB embedding (~23,300 wells, ~11 h). | schedule | Optional migration to a multi-GPU instance (section 3), by author decision only. |
| F6 | The H1 gate test is over plates, using the per-plate mean of 1–8 primary wells; plates with one primary well get full weight. | minor | Unchanged (pre-specified). Report the number of primary wells per plate, and as a secondary result a well-weighted version of the same test. |
| F7 | 20 frozen primary test wells on 2 plates fall outside the test layout (already recorded). | minor | Report, as already planned. |

## 2. Steps (in order)

**A1. Confirm the missing frames (now, CPU only, does not touch the grid).** Re-request every
failed frame once more with 3 attempts and back-off. Record permanent versus recovered in
`results/a2/missing_frames.json`, with, per failed well: plate, split, gene, primary flag, the
phenotype classes from training-plate annotations, and which frames are missing. If any frames
are recovered, **do not** embed them into the training or validation data while the grid
runs; list them for the run report only.

**A2. Pre-register missing-frame handling for test and screenB (commit before the freeze).**
In `manifests/amend02_protocol.json`: a well with any frame missing on S3 after 3 attempts is
excluded from every analysis (no partial-movie embedding; identical to train/validation). The
run report gives, for test and screenB: wells and plates lost, primary wells lost (H1 primary
test wells; H2 primary-gene wells), and the lost wells' phenotype classes against the evaluated
wells' (a chi-square or Fisher test that loss is unrelated to phenotype).

**A3. Finish step 5 and step 6 as committed.** Grid, selection (`results/a2/selection.json`,
committed before any gate), secondary runs, validation evaluations, `results/a2/gates_val.json`
committed and pushed **before** the freeze. Add the secondary well-weighted H1 test (F6) to the
H1 results file; it is not part of the gate.

**A4. Step 7 as committed.** Freeze (clean, pushed, non-main commit), embed the test plates and
screenB, then run `scripts/run_stage1_test.sh` once: Part A test (normalised, gates), Part B
global sensitivity, Part C screenB (global only, outside the gates, siRNA token masked).

**A5. Exploratory analyses (only after A4; never used to revise any gate or claim).**

- **E1 spatial batch correction.** Per plate and clip, subtract a two-way (row and column)
  median polish of the library wells' latents (B-score style), after section 1. Re-run the
  within-series probe and the main probe. Retrain only `h1_learned` and `h1_none` with their
  selected configurations (3 seeds each) and evaluate H1 on validation **and** test with
  replicate drift recomputed in that space. Label every number "exploratory, after the test".
- **E2 batch carried by the forecast.** Within-series replicate probe on forecast clip-12
  latents versus observed (as in stage 1 for the main probe).

**A6. Step 8: `reports/RUN_REPORT_STAGE2.md`.** It covers:
- every gate on validation and test;
- all baselines and controls;
- normalised versus global results;
- screenB;
- the plate-probe table, with (b) labelled uninformative;
- F1 as a finding;
- missing-frame accounting (A1, A2);
- the OOM incident (F4);
- deviations: BIA-only fetch, tile-averaged replicate baselines and the data used to choose
  them, the 20 out-of-layout primary test wells, the screenB decision;
- E1/E2 in a separate "Exploratory, after the test" section;
- what was not run.

## 3. Optional: move to a multi-GPU instance (author decision)

Only if the author launches one and says so. The new instance must be in the **same region as
the persistent filesystem** that holds `~/LambdaFilesystem/geneworld_backups/`.

1. On the old instance, let the in-flight runs finish (or stop them; a stopped run restarts from
   scratch), then commit and push. Confirm that every latent shard and every completed
   checkpoint is in the backup ledger with SHA-256.
2. On the new instance, clone `stage1`, restore the latents from the persistent filesystem and
   verify every SHA-256 against `reports/BACKUP_LEDGER.tsv`. Rebuild the caches and check that
   `results/a2/cache_stats_*.json` are reproduced exactly; stop if they are not.
3. Give `scripts/run_a2_jobs.py` per-GPU assignment (`CUDA_VISIBLE_DEVICES` per worker, up to
   3 workers per 40 GB, or 6 per 80 GB). This is operational only: no protocol value changes,
   and completed runs are skipped.
4. Resume step 5, then continue with A3–A6. Embed test and screenB with one embedder per GPU,
   with fetchers sharing the BIA bucket at most at 128 streams in total.
5. Tell the author when the old instance can be terminated (after step 1's backup check), and
   when the new one is idle.

## 4. What the paper can say, depending on the gates

- **R1 (any outcome):** a frozen V-JEPA 2 latent world model of genome-scale RNAi video, with
  the batch structure of the latent quantified: replicate plates are ~95% decodable, control
  normalisation removes the control offset, and the remaining signature is in library wells.
- **G-Signal-2 passes on test:** action conditioning carries gene-specific dynamics beyond
  replicate memorisation, which is the core world-model claim. H2/H3 become confirmatory.
- **G-Signal-2 fails on test:** a negative result under fair metrics and a memorisation
  baseline. GeneWorld stops, and DrugWorld (design section F) is next.
