# DrugWorld: plan (author, 2026-10-30)

A drug-conditioned latent world model of live-cell time-lapse video, built so that a positive
result can only come from **generalising through what the action is** (its dose, its chemical
structure, its composition with another drug), never from memorising an action label. This plan
is the design document: every gate, split rule, baseline and claim rule below is fixed before
any movie is read.

## 0. Why DrugWorld, and what GeneWorld taught

GeneWorld (repo `GeneWorld`, branch `stage1`; `reports/ROBUSTNESS_STAGE2.md`) found:

- a pre-registered gain from action conditioning (−5.7% vs action-agnostic on test) that was
  equivalent to an **additive per-reagent memorisation baseline** (agnostic forecast +
  λ × the reagent's training-plate residual, λ chosen on validation, beat the model by 2.2%);
- that the gain **did not replicate on independent reagents** (screenB);
- that zero-shot transfer through gene encodings and planning both failed.

Lessons that are built into this design from day 1:

| # | lesson | how DrugWorld applies it |
|---|---|---|
| L1 | Beating an action-agnostic model is not evidence of action understanding. | Every gate also requires beating a **lookup baseline** that uses the same information without a model (nearest dose, interpolation, additive composition, nearest compound). |
| L2 | A trained-shuffled control is a relabelling (and so uninformative) when the action is a **learned ID embedding**. It is informative only when the action input carries **content**. | Actions are content only (log dose, structure encodings); **no learned per-condition ID embedding is used in any gated model.** Shuffled controls permute content (dose values within a drug; encodings across compounds). |
| L3 | Memorisation baselines must be strong. | Baseline strength is checked on validation before the gates: each lookup baseline must beat copy-last; otherwise it is reported as weak and the next stronger one is used. |
| L4 | Independent replication is the strongest test. | Held-out **doses**, held-out **combinations**, held-out **cell lines** and held-out **compound scaffolds**, each a separate gate or claim tier. |
| L5 | Primary metrics must target where the effect is. | Primary horizons and wells are fixed from the protocol and from training-split dose-response, never from evaluation data. |
| L6 | Operational: idle GPUs, OS upgrades and git operations in a live working tree cost days. | Section 11 (operations rules) is mandatory. |

## 1. Questions and hypotheses

- **H1, dose generalisation (primary; idr0119).** A model whose action is (drug structure
  encoding, log dose) forecasts the latent trajectory of cells at **held-out doses** better than:
  - the action-agnostic model;
  - nearest-measured-dose lookup;
  - a model trained with dose values **shuffled within each drug**.

  Tier 2: it also beats linear interpolation between the neighbouring measured doses.
- **H2a, composition (idr0119, AU565).** A model trained on single drugs, and on combinations
  other than the held-out ones, forecasts **held-out drug combinations** better than:
  - the stronger single drug (highest single agent);
  - additive composition of the single-drug latent changes.

  Tier 2: it also beats the Gross et al. phase-specific ODE model on decoded cell counts and
  phase fractions.
- **H2b, unseen compounds (idr0064).** Structure-conditioned forecasts of Erk-reporter
  dynamics for compounds from **held-out scaffolds** beat:
  - the action-agnostic model;
  - the DMSO trajectory;
  - nearest-compound lookup by Tanimoto similarity;
  - a model trained on **shuffled structure encodings**.
- **H1c, cell-line transfer (idr0119; claim tier).** For drugs measured in all lines,
  leave-one-line-out forecasts beat:
  - other lines' mean response at the same dose;
  - an additive per-drug offset learned from the training lines.
- **H3, planning (idr0119).**
  - (a) **Dose inversion:** recover the log dose of held-out-dose wells from their observed
    trajectory, compared with nearest-dose and interpolation inversion.
  - (b) **Combination ranking:** rank held-out combinations by predicted 72 h cell-count
    reduction, compared with Bliss, the highest single agent and the ODE model.

## 2. Data

| set | content (to be verified in D0) | use | licence |
|---|---|---|---|
| **idr0119** (Gross et al., Nat Commun 2023; DOI 10.17867/10000189) | AU565, HCC1143, 21MT1, MDAMB157 breast-cancer lines; phase contrast plus FUCCI fluorescence; single drugs over a dose grid (reported 8 doses) in all lines; combinations in AU565; ~96 h at 30 min; ~3,026 movies, ~3.16 TB | H1, H1c, H2a, H3 | CC BY 4.0 |
| **idr0064** (Goglia et al., Cell Syst 2020) | primary mouse keratinocytes, Erk-KTR plus H2B-RFP, 429 kinase inhibitors at **one dose (2.5 µM)**, DMSO in every batch, every 3 min for 5 h, 9 plates, ~200 cells per drug | H2b | CC BY 4.0 |
| Compound structures | SMILES from the IDR annotations, else PubChem by name or CAS (record the source per compound) | action encodings | PubChem: public domain |
| Encoders | ECFP4 (RDKit, BSD-3); one pretrained molecular encoder with a permissive licence (MoLFormer-XL, Apache-2.0, or ChemBERTa-2, MIT; record the licence and revision); target-class one-hot from the vendor annotation as a separate source | H2b sources | as stated |
| Gross et al. ODE model | reimplemented from the paper's equations, fitted to training-split readouts only | H2a/H3 tier-2 baseline | own code |

**D0.0 data expansion search (half a day, before any design freeze).** Search IDR, the BioImage
Archive, Zenodo, figshare, LINCS (HMS and OHSU) and the Cell Painting Gallery for **live-cell
time-lapse** compound datasets meeting all of:
- >= 50 compounds at >= 3 doses;
- >= 12 h of imaging;
- replicate wells;
- an open licence.

Record every candidate in `reports/D0_DATA_SEARCH.md`. If one qualifies, propose (do not adopt)
adding it as the primary H1/H2b set and stop for the author's decision. If none qualifies, go
on with idr0119 and idr0064 as designed.

## 3. Phase D0: audit and feasibility (days 1–2; stop rules)

1. **Annotation audit (CPU).** From the IDR annotation files, write
   `reports/D0_DATA_AUDIT.md` with:
   - per line: drugs, the dose grid, replicate wells per condition, vehicle wells per plate,
     plates, fields per well, frames, frame interval, channels and channel order;
   - the combinations (which pairs, which dose pairs, which lines);
   - for idr0064: compounds, wells per compound, DMSO wells per plate, frames, interval and
     channels;
   - the **timing of drug addition relative to the first frame** for both sets (from the
     annotations or the papers' methods).
2. **Access and throughput.** Find the OME-Zarr or OME-TIFF location on EBI (BIA S3 bucket,
   IDR zarr, FTP) and measure throughput with curl `--parallel` at 64 and 128 streams, as in
   GeneWorld's `DATA_ACCESS.md`. Estimate the transfer time for both sets.
3. **Embedding pilot on training-split material only** (after D2's split is frozen, using only
   training conditions; until then, use only vehicle wells):
   - channel mapping candidates for V-JEPA 2, which expects RGB:
     - (i) phase only, as grey;
     - (ii) FUCCI-red, FUCCI-green and phase as R, G, B;
     - (iii) idr0064 Erk-KTR and H2B as two channels plus their mean.
   - Choose the mapping by a linear probe on training wells: cell count, and FUCCI phase
     fractions from a simple segmentation (Cellpose, BSD-3, or thresholding). For idr0064,
     probe the Erk C/N ratio.
4. **Feasibility gates** (fixed now; if one fails, stop and report to the author):
   - **F-Decode:** on training wells, latent → log cell count R² >= 0.8, and latent → G1
     fraction R² >= 0.5 (idr0119). For idr0064, latent → Erk C/N ratio R² >= 0.5. Without
     this, decoded readouts are dropped and the ODE comparisons are not run.
   - **F-Dose:** on **training-split** conditions, the latent displacement from vehicle at
     48 h increases monotonically with dose (Spearman ρ >= 0.6) for >= 50% of drug × line
     pairs. Otherwise H1 is not testable with this encoder; stop.
   - **F-Power:** the frozen test split (D2) contains >= 20 held-out-dose conditions for H1,
     >= 6 held-out combination conditions for H2a, and >= 60 held-out compounds for H2b. If a
     hypothesis falls short, it is dropped (and recorded) before training, not after.

## 4. Phase D1: embedding (days 2–3)

- Frozen V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`, the same revision and weights hash
  as GeneWorld), with the channel mapping chosen in D0.
- **idr0119:**
  - 16-frame clips (8 h) every 6 frames (3 h), on 256 × 256 tiles from the chosen pyramid
    level;
  - the tile grid is fixed per field before embedding;
  - the stored state is tile-mean pooled tokens, as in GeneWorld, plus a 4 × 4 token grid for
    one probe.
- **idr0064:** clips of 16 frames (48 min) every 4 frames (12 min).
- **Stream and embed only.** No raw movie is kept beyond an in-memory buffer. Leakage
  fingerprints go in every shard, and a shard manifest is kept with SHA-256, with the GeneWorld
  guard.
- **All embedding of a dataset happens on one instance type in one software environment.** If
  that changes, re-embed a seeded sample and require byte identity (GeneWorld S18) before
  continuing.
- **G-Data:** >= 95% of wells of each dataset embedded; failures listed by cause.

## 5. Phase D2: freeze (day 3; one commit before any model is trained)

Commit `splits/`, `manifests/drugworld_protocol.json` and the frozen decoders together.

**Splits (idr0119), all seeded and stratified:**
- **dose split:** per drug × line, of the measured doses:
  - the lowest dose and vehicle are always training;
  - the **highest dose** is the extrapolation set, held out entirely (val/test by line, see
    below);
  - of the interior doses, one is validation and two are test, chosen by seed with no two
    adjacent doses both held out where the grid allows.
- **combination split (AU565):** if >= 3 distinct drug pairs exist, **leave-pair-out**: one
  pair for validation, >= 2 pairs for test; all their dose combinations are held out. Otherwise,
  **leave-dose-combination-out** within pairs (val/test by seed). The single-drug wells of
  every drug are always available to training.
- **line split (H1c):** each line is held out once in a separate set of models (4 folds);
  one fold is fixed as validation and three are test.
- **plate and replicate:** held-out conditions are evaluated on all their replicate wells.
  Replicate wells of *training* conditions are split 80/20 by plate for selection only.

**Splits (idr0064):** a Bemis–Murcko scaffold split of compounds, 70/10/20 (train/val/test),
stratified by target class and by Goglia's published Erk-phenotype class (an external
annotation, frozen as provided). DMSO wells are training, except a plate-stratified 20% kept
for normalisation checks.

**Protocol (fixed now):**
- **Normalisation:** per plate and clip, mean and SD over the **vehicle** wells (DMSO); the SD
  is floored at the 5th percentile over plates. Global standardisation is kept as a
  sensitivity analysis.
- **Start state:** if drug addition precedes the first frame by <= 1 h, use the well's own
  first clip. Otherwise use the **plate-matched vehicle wells' first-clip mean**
  (counterfactual start), so that the effect of the drug is never already in the context. The
  rule is applied from the D0 audit, before any model is trained.
- **Horizons:**
  - idr0119: 24, 48 and 72 h after drug addition; primary = the mean of 48 h and 72 h; 96 h is
    secondary.
  - idr0064: 1, 2, 3 and 4 h; primary = the mean of 2–4 h.
- **Primary wells:** all wells of the held-out conditions; nothing is filtered by outcome.
- **Decoders:** linear (ridge, alpha chosen by cross-validation on training plates) from
  latent to log cell count, G1 fraction and S/G2/M fraction (idr0119), and to Erk C/N mean and
  pulse rate (idr0064). They are trained on training conditions only and frozen with SHA-256.
- **Action encodings:**
  - drug = [ECFP4 reduced to 128 dimensions by PCA fitted on training drugs; pretrained
    molecular embedding reduced to 128 dimensions; vendor target-class one-hot];
  - dose = log10(dose / max dose of that drug in the grid), plus an indicator for vehicle;
  - combination = a **set of two (drug, dose) tokens**, order-invariant.
  - **No per-condition, per-drug or per-well learned ID embedding** in any gated model (L2).

## 6. Model

- **Predictor:** the block-causal action-conditioned transformer from GeneWorld
  (`src/geneworld/train/a2.py`), with the action pathway replaced by a small MLP over the
  content encodings and a DeepSets sum for the combination set. Rollout as in GeneWorld.
- **Families (each with 3 seeds):**

  | family | action input |
  |---|---|
  | `dw_full` | structure and dose |
  | `dw_none` | none (action-agnostic) |
  | `dw_doseshuf` | dose values permuted **within each drug** over its measured grid, fixed permutation per seed, identically in training and evaluation |
  | `dw_structshuf` (H2b) | encodings permuted across compounds |
  | `dw_doseonly` | dose without the drug identity encoding (idr0119; diagnostic) |
  | `dw_line_k` | `dw_full` trained without line k (H1c) |

- **Grid (fixed now):** width {512, 768} × dropout {0.1, 0.3} × weight decay {0.05, 0.2},
  lr 3e-4, batch 256, at most 12,000 steps, early stopping on the validation primary error
  every 250 steps with patience 3,000.
- **Selection:** per family, by the mean over seeds, recorded in `selection.json` and committed
  before any gate.
- **Config-matched references:** `dw_none` and every shuffled family are also trained at
  `dw_full`'s selected configuration (3 seeds), so that regularisation never explains a gain.

## 7. Baselines (all computed in the normalised space and, where decoders pass F-Decode, on
decoded readouts)

| baseline | used in | definition |
|---|---|---|
| copy-last, vehicle trajectory, time-slot mean | all | as in GeneWorld |
| **nearest measured dose** | H1, H3a | trajectory mean of the same drug × line at the nearest *training* dose (in log dose) |
| **linear dose interpolation** | H1 tier 2, H3a | per clip, linear in log dose between the two neighbouring training doses (extrapolation: the last two) |
| **Hill-fit per readout** | H1 (readouts) | four-parameter Hill curve per drug × line × horizon on training doses |
| **highest single agent (HSA)** | H2a | the single-drug component with the larger effect at its dose |
| **additive latent composition** | H2a | vehicle + Δdrug A + Δdrug B (latent changes from vehicle at the component doses, from training wells) |
| **Bliss** | H2a, H3b (readouts) | Bliss independence on decoded cell-count fractions |
| **Gross et al. ODE** | H2a and H3b tier 2 (readouts) | phase-specific rate model fitted to training conditions only, per the paper |
| **nearest compound (Tanimoto kNN)** | H2b | mean training trajectory of the k = 5 most similar training compounds (ECFP4 Tanimoto), similarity-weighted |
| **lines mean / additive per-drug offset** | H1c | other lines' mean response; vehicle-anchored training-line drug offset added to the held-out line's vehicle |
| **additive residual (GeneWorld R2 form)** | H1, H2b | agnostic forecast + λ × residual of the nearest training condition (nearest dose / nearest compound), λ from {0, 0.25, 0.5, 0.75, 1} chosen on validation |

**Baseline-strength check (L3), on validation before the gates:** report every baseline against
copy-last. A lookup baseline that does not beat copy-last is labelled "weak", and gates that
name it use the next baseline in its row's family (for H1: interpolation, then Hill; for H2a:
additive composition, then Bliss; for H2b: kNN, then the additive residual). This rule is fixed
now.

## 8. Gates (validation first, then test once)

The statistical unit is the **held-out condition**: drug × line × dose for H1, the combination
condition for H2a, the compound for H2b. Errors are averaged over replicate wells, then over 3
seeds. The test is a one-sided Monte Carlo sign flip over conditions (100,000 draws), with
results also given per drug. With fewer than 10 drugs, no cluster CI over drugs is claimed (TiJepa
rule 3); per-drug results are shown instead.

| gate | pass requires (primary horizons, normalised space) |
|---|---|
| **G-Data** | >= 95% of the wells of each set embedded |
| **G-Dose** (H1) | `dw_full` beats (a) `dw_none` (config-matched), (b) nearest measured dose, (c) `dw_doseshuf`, each p < 0.05 |
| **G-Dose+** (H1 tier 2) | G-Dose, and beats linear dose interpolation (or Hill, per L3), p < 0.05 |
| **G-Combo** (H2a) | `dw_full` beats HSA and additive latent composition on held-out combinations, p < 0.05 each |
| **G-Combo+** | G-Combo, and beats Bliss and the ODE model on decoded 72 h cell count, p < 0.05 each (only if F-Decode passed) |
| **G-Compound** (H2b) | for >= 1 structure source: `dw_full` beats `dw_structshuf` of that source, kNN-Tanimoto and `dw_none`, p < 0.05 each, with Holm correction across the sources |
| **G-Plan** (H3) | (a) dose inversion: median absolute log-dose error below nearest-dose inversion and interpolation inversion (sign-flip over wells); (b) combination ranking: Spearman ρ with the measured 72 h cell-count reduction above Bliss and HSA (bootstrap over conditions) |

Pre-registered claim rules (applied mechanically in the final report):

| claim | requires |
|---|---|
| "The model generalises across dose" | G-Dose on test |
| "…beyond interpolation" | G-Dose+ on test |
| "The model composes drug actions" | G-Combo on test |
| "…better than standard synergy and mechanistic models" | G-Combo+ on test |
| "The model generalises to unseen compounds through structure" | G-Compound on test |
| "The learned dynamics transfer across cell lines" | H1c: `dw_line_k` beats the additive per-drug offset in >= 2 of 3 test folds, p < 0.05 each |
| "The model can be used for planning" | G-Plan (a) or (b) on test, stated per part |

Holm correction is applied across the four primary gates (G-Dose, G-Combo, G-Compound, G-Plan)
on test, in addition to the per-gate rules; both corrected and uncorrected outcomes are
reported. Any claim requires its gate on **test**; validation outcomes are reported beside.

## 9. Controls (from day 1)

- **Static-clip control** (each clip repeats its middle frame), **context swap** (start state
  from another well of a different condition) and the **time-slot mean**.
- **Evaluation-time action shuffle:** dose shuffled within drug; structures across compounds.
- **Leakage:** fingerprint duplicates across splits, and identical compounds under different
  names (InChIKey). Scaffold-split integrity is checked: no test scaffold in training.
- **Plate-batch probe:** a within-condition replicate-plate probe (GeneWorld lesson). It is
  reported and never used to change the gates.
- **Re-embedding identity:** whenever hardware or the environment changes (section 4).

## 10. Phases and timeline

| phase | days | output | stop / review |
|---|---|---|---|
| D0 | 1–2 | data search, audit, throughput, pilot, feasibility gates | stop if a feasibility gate fails or a better dataset is found (author decides) |
| D1 | 2–3 | embeddings, G-Data | stop if G-Data fails |
| D2 | 3 | frozen splits, protocol, decoders | — |
| D3 | 3–5 | grid, selection (committed), config-matched references, secondary families | — |
| D4 | 5–6 | validation evaluations, baseline-strength check, `gates_val.json`, `reports/DW_GATES_VAL_SUMMARY.md` with a checklist | **STOP for the author's "go"** |
| D5 | 6 | freeze, test once (all gates), screen of all claim rules | — |
| D6 | 6–8 | exploratory after the test, run report, manuscript (`paper/drugworld/`) | — |

**Checklist at D4** (the agent checks it and reports pass/fail per item):
1. commit order (protocol, splits and decoders, then selection, then any evaluation output,
   then `gates_val.json`);
2. 3 seeds per family, with config-matched references present;
3. no learned ID embedding in any gated model;
4. permutation hashes for the shuffled families;
5. evaluated condition counts equal the frozen ones minus listed exclusions;
6. the baseline-strength table and the substitutions it implies;
7. no file refers to test conditions;
8. normalised and global results both present;
9. the feasibility-gate results recorded.

## 11. Operations (mandatory)

- **Instance:**
  - D0–D2 on **1 x A100-40GB** (Lambda Stack image, us-east-1, with the persistent filesystem
    attached);
  - D3 grid: the agent estimates the GPU-hours from D2 and asks the author whether to use one
    GPU or launch 8, giving the cost of each option;
  - never keep a multi-GPU instance through a review pause.
- **Cost rule:** tell the author whenever all GPUs have been idle for > 30 minutes and no job
  is queued, and at every stop point, with the instance's hourly cost.
- **OS freeze at instance start:**
  - disable the apt timers and unattended-upgrades;
  - mask the apt services;
  - hold the NVIDIA, CUDA and Fabric Manager packages;
  - set needrestart to list-only;
  - no reboot, and no `apt upgrade`.
- **Git (G4):** no stash, checkout, reset, rebase or pull in the working tree while jobs run;
  commit by explicit paths; git maintenance in a separate worktree.
- **Backups:** every shard, checkpoint and results file is copied to
  `~/LambdaFilesystem/drugworld_backups/` with SHA-256 and recorded in
  `reports/BACKUP_LEDGER.tsv`; ledger rows are appended only. A shard manifest and guard are
  kept as in GeneWorld.
- **Repository:**
  - a new private repository `rafidquadir-eng/DrugWorld` if the agent can create one;
    otherwise branch `drugworld` in GeneWorld;
  - code shared with GeneWorld is copied with its commit recorded (provenance, claims checker,
    stats, cache, s3fetch, the trainer);
  - `main` is never pushed to; no force-push, publish, post or release.
- **Rules carried over:** the TiJepa `RULES.md`; never bypass access controls (Dryad-style
  browser checks included); never print or commit credentials.
