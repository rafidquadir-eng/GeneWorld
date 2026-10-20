# GeneWorld stage 1: run report (validation splits)

Run 2026-10-19 21:20 UTC to 2026-10-20 09:37 UTC on one A100-40GB. Design:
`docs/NEXT_PROJECT_DESIGN.md` section A and `docs/DESIGN_AMENDMENT_01.md` (design commit
`deb443d`). All numbers below are on **validation** splits. **Test plates and test genes have
not been evaluated**; that is a one-time step after freezing (`scripts/run_stage1_test.sh`).
Results files: `results/{gates,train,h1,h2,h3,controls}/`; logs: `logs/stage1_val/`.

## Gate outcomes

| gate | rule | outcome (validation) |
|---|---|---|
| **G-Data** | >= 10,000 movies decoded and embedded | **pass**: 14,663 movies on 168 plates (90 of 14,753 failed) |
| **G-Signal** | action-conditioned beats action-agnostic **and** time-slot mean, p < 0.05 over plates | **fail**: beats time-slot mean (-62.5%, p = 1e-5) but not action-agnostic (-1.9%, p = 0.088; 13 of 18 plates better) |
| **G-ZeroShot** | >= 1 encoding source beats shuffled encodings, 95% CI over genes below 0 | **fail**: no source; closest STRING (-0.6%, CI -0.0041..+0.0002) and ESM-2 (-0.5%, CI -0.0036..+0.0006) |
| **G-Plan**, trajectory | planner hit@100 above random and above endpoint retrieval | **fail**: hit@100 0.034 vs random 0.033 (p = 0.48); vs retrieval +0.012 (CI -0.011..+0.037) |
| **G-Plan**, phenotype (amendment) | unseen genes: AP beats permutation (BH q < 0.05) for >= 2 of 3 phenotypes and beats guilt-by-association for >= 1 | **fail**: 0 of 3 with q < 0.05; 0 of 3 beat guilt-by-association |

Under the design's rules: G-Signal failed, so H1 is the result (with its controls), H2/H3 are
exploratory, and **stage 2 (more plates) is not started**.

## Data (G-Data)

- Source: MitoCheck screenA, BioImage Archive S-BIAD865 OME-Zarr on EBI S3 (CC0), pyramid
  level 1 (512 x 672), four 256 x 256 tiles per well; streamed and embedded, never stored.
- Stage 1: 50 library series, all 169 replicate plates, 14,753 wells (controls on every plate,
  every phenotype class). Embedded: 14,663 wells on 168 plates.
- Failures: plate LT0003_40 (83 wells) lacks frame 92 on S3 (HTTP 404); 7 other single wells.
  Plate LT0078_05 (training split, 82 wells) has an all-black first frame (frame-mean 0) in
  every well; it was kept in training and touches no evaluation set.
- Backbones (frozen): V-JEPA 2 ViT-L `facebook/vjepa2-vitl-fpc64-256` @ `b3c1679b`, MIT;
  DINOv2 ViT-S/14 (timm @ `4610ca14`, Apache-2.0); TiJepa-1 (`tijepa1_T1_seed0.pt`).
  `manifests/models.yaml`.
- Backups: 558 latent shards (45 GB + 0.9 GB static) and 17 checkpoints on the persistent
  filesystem, SHA-256-verified (`reports/BACKUP_LEDGER.tsv`).

## H1: forecasting on held-out plates (V-JEPA 2)

Validation: 818 wells (683 library) on 18 plates; forecast from clip 0 for 12 clips (3 h apart).
Error = MSE in standardised latent space, mean over 3/12/24/36 h.

| method | error | vs action-conditioned |
|---|---|---|
| **action-conditioned predictor** (`h1_learned_B`, 22.2 M, step 1,500) | **0.319** | — |
| action-agnostic predictor (`h1_none_C`, 88.4 M, step 500) | 0.325 | -1.9%, p = 0.088 |
| plate negative-control drift (z0 + control change) | 0.488 | -33.5%, p = 1e-5 |
| time-slot drift (z0 + mean change) | 0.493 | -34.7%, p = 1e-5 |
| plate negative-control mean | 0.823 | -61.8%, p = 1e-5 |
| time-slot mean | 0.832 | -62.5%, p = 1e-5 |
| copy-last | 0.843 | -60.7%, p = 1e-5 |

p: one-sided Monte Carlo sign-flip over plates. Per horizon vs action-agnostic: 3 h -0.0024
(p = 0.058), 12 h +0.0106 (p = 0.99), 24 h -0.0021 (p = 0.37), **36 h -0.0312 (p = 0.002)**
(per-horizon tests are secondary; the pre-specified primary is the 4-horizon mean).

Configuration selection (validation, mean rollout error): learned A/B/C = 0.321 / **0.312** /
0.322; agnostic A/B/C = 0.326 / 0.320 / **0.320**. Every best checkpoint came early (step
500–1,500 of 8,000), so longer training overfit.

**Controls**

- Context swap (clip-0 context taken from a well with the same gene on another plate):
  error rises from 0.319 to 1.450. Forecasts are driven by the well's own starting state.
- Plate-identity probe at 36 h (18 plates, chance 5.6%): 73.7% from observed and 72.7% from
  predicted latents. The latent state is strongly plate-specific.
- Phenotype-class probe (trained on observed training-well latents; AUROC at 36 h,
  observed / action-conditioned forecast / action-agnostic forecast):

  | class | observed | action-conditioned | action-agnostic |
  |---|---|---|---|
  | mitotic delay | 0.83 | 0.77 | 0.71 |
  | polylobed | 0.86 | 0.90 | 0.55 |
  | binuclear | 0.86 | 0.81 | 0.62 |
  | cell death | 0.86 | 0.80 | 0.71 |
  | nuclear shape | 0.82 | 0.81 | 0.76 |
  | segregation | 0.77 | 0.78 | 0.61 |

  The gene action carries phenotype-relevant information into the forecast (higher AUROC in
  every class). That information is a small part of the total latent error, which is
  dominated by plate- and well-specific variation.
- Static clips (300 validation wells, each clip = its middle frame repeated): V-JEPA 2's
  real-vs-static latent distance is 3.4 x the median distance between two wells of the same
  plate (DINOv2 0.25 x; TiJepa-1 0.15 x). V-JEPA 2 latents depend strongly on motion.
  Predictor error from a static context: 2.72 (vs 0.32 from the real context).
- Leakage: 4,965 held-out-plate wells vs 9,698 training-plate wells. Maximum fingerprint
  r = 0.47, no match >= 0.98, none >= 0.90. The 82 identical frame-0 hashes are LT0078_05's
  black frames. No siRNA sequence crosses gene-split parts (3 genes regrouped before any
  evaluation).

**Alternative backbones** (same recipe, config B): neither beats its own action-agnostic
predictor. DINOv2 0.364 vs 0.360 (+2.2%); TiJepa-1 0.365 vs 0.353 (+7.0%). Both are worse than
V-JEPA 2 (0.319).

## H2: unseen genes (exploratory after G-Signal)

Validation genes, wells on training or validation plates; predictor fed each source's encoding
(config B). Per gene, mean error over its wells.

| source | genes | true vs shuffled (CI over genes) | vs nearest-gene lookup | vs action-agnostic |
|---|---|---|---|---|
| Perturb-seq (K562) | 152 | -0.6% (-0.0050..+0.0011) | -33.2% | +3.1% |
| JUMP CRISPR | 131 | +0.5% (-0.0012..+0.0051) | -32.9% | +4.7% |
| JUMP ORF | 193 | +0.3% (-0.0021..+0.0044) | -34.0% | +3.4% |
| STRING | 284 | -0.6% (-0.0041..+0.0002) | -36.3% | -1.6% |
| ESM-2 | 287 | -0.5% (-0.0036..+0.0006) | -37.0% | -2.2% |

Complex-held-out split (STRING, 316 genes): -0.4% (CI -0.0031..+0.0001). No source passes.
Every predictor beats the nearest-gene trajectory lookup by about a third, but that comes from
the model's use of the well's own history, not from the gene encoding.

## H3: planning (exploratory; STRING, the source nearest to passing)

**Trajectory targets** (500 validation wells, 243 genes, 3,037 candidate genes):

| method | hit@1 | hit@10 | hit@100 | MRR | median rank | top-10 same class | top-10 same complex |
|---|---|---|---|---|---|---|---|
| planner, dynamics | 0.002 | 0.006 | 0.034 | 0.0043 | 1,510 | 0.155 | 0.007 |
| planner, endpoint only | 0.002 | 0.006 | 0.030 | 0.0046 | 1,493 | 0.162 | 0.009 |
| endpoint retrieval (kNN-imputed) | 0.000 | 0.002 | 0.034 | 0.0021 | 1,340 | 0.077 | 0.007 |
| random | 0.0003 | 0.0033 | 0.033 | 0.0028 | — | — | — |
| *observed-replicate retrieval (oracle)* | *0.002* | *0.048* | *0.148* | *0.019* | *858* | *0.270* | *0.010* |

**Phenotype targets** (amendment 01; target = mean clip-12 latent of training wells MitoCheck
labels with the phenotype; start = negative-control states of 20 training plates). Unseen
validation genes (1,701 with STRING encodings):

| phenotype | validated hits | planner AP | prevalence | perm p (BH q) | expression-matched p | guilt-by-association AP | endpoint retrieval AP | AUROC |
|---|---|---|---|---|---|---|---|---|
| mitotic delay | 10 | 0.020 | 0.006 | 0.050 (0.149) | 0.105 | 0.030 | 0.007 | 0.50 |
| polylobed | 9 | 0.010 | 0.005 | 0.22 (0.27) | 0.20 | 0.031 | 0.009 | 0.64 |
| binuclear | 7 | 0.008 | 0.004 | 0.27 (0.27) | 0.28 | 0.013 | 0.005 | 0.59 |

Seen genes (secondary; 1,880 training genes): mitotic delay AP 0.179 and polylobed AP 0.133
(both q = 0.0001, both above guilt-by-association at 0.156 and 0.117); binuclear AP 0.020
(q = 0.07). Seen genes' own wells contributed to the targets and to training, so this tests the
fitted predictor, not planning for new genes.

## Deviations from the design (all made before any test data was read)

1. Stage 1 was sampled by whole library series (all replicate plates, same wells), so that seen
   genes on held-out plates have training replicates. Random plate sampling gave only 36%
   overlap.
2. State = four tile tokens, each the mean of V-JEPA 2's pooled patch tokens, rather than the
   full patch-token grid. Horizons are 3/12/24/36 h (clip spacing 3 h, 93 frames) instead of
   4/12/24/40 h. The phenotype target uses clip 12 (ends 43.5 h after imaging starts) for the
   amendment's 48 h, and clip 6 (25.5 h) for 24 h.
3. Predictor size was selected on validation among 21 M and 88 M (the design said 50–100 M).
   The 22 M model won for the action-conditioned predictor.
4. Complexes from EBI Complex Portal (CC0); CORUM was unreachable from this instance.
5. Not yet run: screenB independent-reagent test, idr0002 transfer, CEM refinement and
   combination proposals (A.6), the phenotype-score regression baseline (A.8), and multiple
   seeds.

## Status

- **GPU idle since 09:37 UTC 2026-10-20.**
- Test splits untouched. `scripts/run_stage1_test.sh` evaluates the frozen models once and
  records the freeze commit.
