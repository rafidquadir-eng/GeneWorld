# Design amendment 02 (after stage-1 validation, before any test data is read)

Amends `docs/NEXT_PROJECT_DESIGN.md` section A and `docs/DESIGN_AMENDMENT_01.md`. Written
after the stage-1 validation run (`reports/RUN_REPORT_STAGE1.md`, G-Signal / G-ZeroShot /
G-Plan failed on validation) and **before any test plate, test gene, screenB well or transfer
set has been evaluated**. The commit hash of this file becomes `design_commit` in every later
results file. Stage-1 results stay in the repository and in the paper as the pre-amendment
analysis; nothing here re-labels them.

## 0. Why this amendment exists, and its limits

Stage 1 showed that the primary metric and preprocessing, not only the model, may have hidden
the gene effect:

1. **Plate effects dominate the latent.** A linear probe identifies the plate of a 36 h latent
   with 73.7% accuracy (chance 5.6%). Latents were standardised globally, not per plate.
2. **The primary metric dilutes the gene effect.** Most MitoCheck genes have no phenotype, and
   phenotypes appear late, yet the primary metric averaged every library well and included the
   3 h and 12 h horizons. The only significant gain from the action was at 36 h
   (-6.4%, p = 0.002), a secondary test.
3. **No memorisation baseline.** The phenotype-probe gain (e.g. polylobed AUROC 0.55 -> 0.90)
   could be the per-gene embedding recalling what the gene's training replicates looked like.
   A world model must beat that lookup to claim it combines state and action.
4. **Data-limited training.** Every selected checkpoint came at step 500–1,500 of 8,000.
5. **Underpowered planning ground truth.** 7–10 validated hits per phenotype among
   unseen validation genes.

This amendment fixes those five things and nothing else. **One round only**: if the amended
gates fail on validation, the models are frozen as they are, the test splits are run once,
the result is reported as negative, and the project moves to the backup (design section F,
DrugWorld). No further amendment to GeneWorld's gates is allowed after this one.

## 1. Preprocessing: per-plate normalisation to negative controls (all hypotheses)

- For each plate p, clip index t and latent dimension d: mu_{p,t,d} and sigma_{p,t,d} are the
  mean and SD over that plate's **negative-control wells** (tile-mean latent). Every well's
  latent is replaced by (z - mu_{p,t}) / sigma_{p,t}, with sigma floored at the 5th percentile
  of sigma over all plates for that d. If a plate has < 4 usable negative-control wells it is
  dropped and listed.
- Controls are part of every plate and of no split question, so this uses no evaluation label.
- The globally standardised pipeline is kept as a reported **sensitivity analysis** (results
  for both preprocessings are reported side by side; the gates use the normalised one).
- Check: the plate-identity probe at 36 h is re-run after normalisation and reported.

## 2. Metrics: primary horizons and primary wells

- **Primary horizons: 24 h and 36 h** (mean of the two). 3 h and 12 h become secondary.
- **Primary wells for H1: "phenotypic-siRNA wells".** A held-out-plate well is primary if its
  siRNA's wells **on training plates** carry a MitoCheck reproducible-phenotype term, decided
  from training-plate annotations only (never from the evaluated well or other held-out
  plates). The list is frozen in `splits/amend02_h1_primary_wells.tsv` with SHA-256 before
  any amended model is trained. All library wells remain a reported secondary analysis.
- **Primary genes for H2/H3** (unseen genes have no training wells): genes whose MitoCheck
  annotation carries a reproducible-phenotype term, frozen in
  `splits/amend02_h2_primary_genes.tsv` before training. This uses the external annotation
  only to choose which genes are scored, not as a model input. All validation genes remain a
  secondary analysis.

## 3. New baselines that the action-conditioned model must beat

- **Replicate drift (memorisation baseline, H1):** prediction for well w of siRNA s on
  plate p at horizon h = z_w(0) + mean over s's training-plate wells of [z(h) - z(0)], in the
  normalised space. If s has no training-plate well, the gene's other siRNAs; else the plate
  control drift.
- **Replicate endpoint:** mean of s's training-plate latents at h (no use of the well's own
  state).
- Both are added to H1 tables, to the phenotype probe, and as H3 endpoint-retrieval variants.

## 4. Gates (replace the A.10 conditions for G-Signal, G-ZeroShot and G-Plan)

All p values are one-sided Monte Carlo sign-flip tests over plates (H1) or bootstrap CIs over
genes (H2), as before; errors are averaged over the three seeds' predictions before testing.

| gate | condition (validation) |
|---|---|
| **G-Signal-2** | On primary wells at 24/36 h, the action-conditioned predictor beats (a) the action-agnostic predictor **and** (b) replicate drift, each p < 0.05 over plates. |
| **G-ZeroShot-2** | On primary genes at 24/36 h, at least one encoding source beats its shuffled-encoding control with the 95% CI over genes below 0, **and** beats the action-agnostic predictor. |
| **G-Plan-2** | Phenotype planning (amendment 01 section 1) on unseen genes, with the ground truth of section 6 below: AP beats the permutation null (BH q < 0.05) for >= 2 of 3 primary phenotypes **and** beats guilt-by-association for >= 1. Trajectory planning: hit@100 above random and above replicate-endpoint retrieval. |

If G-Signal-2 fails, H2/H3 are exploratory and reported as such. The amended gates are
evaluated once on validation, then the test splits are evaluated once (section 8) regardless
of outcome.

## 5. Training changes (bounded)

- **Three seeds** for every model entering a gate (action-conditioned with the selected
  source, action-agnostic, each H2 source and its shuffled control). Selection is on
  validation, by the mean over seeds.
- **Configuration grid is fixed in advance** and small: predictor width {22 M, 50 M},
  dropout {0.1, 0.3}, weight decay {0.05, 0.2}, early stopping on validation primary error
  (evaluate every 250 steps). No other hyper-parameter is tuned. The selected configuration
  is recorded before the gate is computed.

## 6. Planning ground truth with adequate power

- **Primary ground truth:** genes whose MitoCheck screenA annotation carries the phenotype's
  reproducible-phenotype term (frozen with SHA-256 in `data/annotations/amend02_truth_<P>.tsv`;
  counts per phenotype among unseen validation and test genes reported before evaluation).
- **Stringent ground truth (secondary):** the validated-hit lists frozen in amendment 01.
- Targets stay built from training-split genes only (amendment 01). Nulls unchanged
  (permutation, expression-matched, guilt-by-association, endpoint retrieval), plus
  replicate-endpoint retrieval.

## 7. Stage 2: more data (replaces "expand only after G-Signal passes")

Stage 1 was data-limited (finding 4), and EBI S3 now gives ~181 MB/s (`reports/DATA_ACCESS.md`),
so data volume is no longer the constraint. Stage 2 is run **before** the amended gates:

- **Scope:** every library series, **train and validation plates only**, same-wells design as
  stage 1 (all controls, every siRNA with a reproducible phenotype, random library wells up to
  the per-plate budget), target ~60,000 additional wells (~30 GPU-h at the measured
  0.55 wells/s). If throughput and budget allow, extend toward all train/validation wells.
- **Splits are the frozen whole-screen splits** (`splits/plate_split.tsv`,
  `splits/gene_split.tsv`, `splits/gene_split_complex.tsv`); nothing is re-split.
- **Test plates are not embedded until the freeze** (section 8).
- Fetcher: native parallel client, plates split across the BIA and IDR S3 buckets; same
  stream-and-embed rule (no raw movie on disk; in-memory buffer <= 300 GB).

## 8. Test, once

After the amended validation gates are computed and recorded, the models are frozen (clean,
pushed, non-main commit), the test plates are embedded, and `scripts/run_stage1_test.sh`
(extended to the amended metrics and baselines) evaluates every gate once on test plates,
test genes, and, if embedded, screenB (independent reagents). Every number is reported,
including failures. No model, threshold or preprocessing change after this point.

## 9. Provenance fixes

- The V-JEPA 2 revision (`b3c1679b`) and checkpoint SHA-256 are written into every results
  file (stage-1 files record `None`).
- The phenotype probe reports, per class: number of positive and negative evaluation wells,
  and a 95% cluster-bootstrap CI over plates for each AUROC and for each difference
  (action-conditioned minus action-agnostic, minus replicate drift).

## 10. Erratum E-02.1 (2026-10-22, before any gate)

Recorded from `docs/AMEND02_COMPLETION_PLAN.md` v2, finding F8, before any selection, H2/H3
validation output or gate existed.

**G-Plan-2 trajectory rule uses an oracle comparator.** Section 4 requires the planner's
hit@100 to beat *replicate-endpoint retrieval*. For unseen genes that baseline ranks candidates
by the observed clip-12 latents of each candidate gene's own wells, i.e. the unseen gene's real
data, which the planner never receives (stage 1: oracle hit@100 0.148 vs planner 0.034). The
rule asks a zero-shot model to beat a method that sees the answer's replicates, so it cannot
test planning. The error is in the amendment text, not in the implementation, which followed
the text.

**Corrected rule.** Trajectory planning passes if hit@100 is above random (one-sided binomial
p < 0.05) **and above kNN-imputed endpoint retrieval** (endpoint imputed from the k nearest
*training* genes in the planning source's encoding space, as in stage 1, using no data from the
unseen gene; bootstrap CI of the difference over targets above 0). Replicate-endpoint retrieval
stays in every table, labelled "oracle ceiling (uses the unseen gene's observed wells)". The
phenotype-planning half of G-Plan-2 is unchanged.
