# Design amendment 01 (before any test data is read)

Amends `docs/NEXT_PROJECT_DESIGN.md` section A. Committed before any MitoCheck movie or
annotation of a test split has been read; the commit hash of this file is recorded in every
later results file as `design_commit`.

## 1. H3 gets a second family of targets: phenotype-defined planning

H3 as designed scores planning against **held-out trajectories** (find the gene whose
knockdown produced a given held-out well). It now also scores planning against
**phenotype-defined targets**: reach a MitoCheck phenotype state.

**Targets.** For each MitoCheck phenotype P with a validated hit list — at minimum *mitotic
delay/arrest*, *polylobed nuclei*, *binuclear cells* (cytokinesis failure), and *cell death* if
its validated list has >= 20 genes — the target state z*_P is the mean predicted-horizon latent
(48 h; also 24 h as a secondary horizon) of **training-split** wells that the MitoCheck
primary-screen scoring labels with P. Targets are built only from training-split genes, so no
evaluated gene contributes to its own target.

**Planner output.** For each target, the planner ranks every candidate gene by the latent
distance between its rolled-out state and z*_P (exhaustive over all genes with an action
encoding), from the plate-matched negative-control start state.

**Ground truth.** MitoCheck's validated hits for P: genes confirmed in the validation screen
(screenB) with independent siRNAs, as published by Neumann et al. 2010 (Supplementary
Tables / MitoCheck database phenotype calls). The exact list and its source file are frozen in
`data/annotations/validated_hits_<P>.tsv` with SHA-256 before evaluation.

**Evaluation sets.** Scored separately on (i) **unseen genes** (the H2 held-out split; the
primary test) and (ii) seen genes (secondary; tests the predictor, not zero-shot transfer).

**Metrics.** Average precision and AUROC of the ranking against the validated hits;
enrichment and hit counts at k = 10, 50, 100; per phenotype and averaged.

**Nulls.** (a) Permutation null: the planner's ranking is compared with 10,000 random
permutations of gene labels within the evaluated set (one-sided p, and BH across phenotypes);
(b) a degree/expression-matched null, permuting labels within bins of gene expression level in
HeLa (genes not expressed in HeLa are rarely hits); (c) baselines: endpoint retrieval (rank by
distance of the gene's observed or kNN-imputed endpoint embedding, no dynamics), the
action-agnostic predictor (constant ranking = null by construction), and ranking by the gene
encoding's similarity to the mean encoding of known hits (a "guilt-by-association" baseline
that needs no world model).

**Decision rule (added to gate G-Plan).** Phenotype planning counts as successful if, on
unseen genes, average precision beats the permutation null (BH q < 0.05) for at least two of
the three primary phenotypes **and** beats the guilt-by-association baseline for at least one.
Either result is reported.

## 2. Data handling constraints (author instructions, 2026-10-19)

- **Stream-and-embed only.** Raw movies are never stored beyond a rolling buffer of
  <= 300 GB; each well is embedded and its raw frames deleted.
- **Byte-range reads** of only the needed wells from the per-plate OME-Zarr zip archives
  (S-BIAD865); use the downsampled pyramid level closest to 256 x 256 tiles if one exists.
- **Endpoint choice by measurement:** test 128–256 parallel streams on EBI HTTP, the EBI S3
  endpoint, and a free Globus personal endpoint (no AWS account), and use the fastest.
- Latents and annotations stay on local disk and are backed up to the Lambda persistent
  filesystem with SHA-256 (`src/geneworld/backup.py`, ledger `reports/BACKUP_LEDGER.tsv`).
- **Stage 1:** a stratified subset of ~15,000 movies covering every phenotype class, with
  the negative controls of every sampled plate; expand only after G-Signal passes.

## 3. Selection discipline

All model, recipe, horizon and hyper-parameter choices are made on validation splits
(validation plates for H1; validation genes for H2/H3). Test plates, test genes, screenB and
the transfer sets are evaluated once, after the models are frozen, and the freeze commit is
recorded.
