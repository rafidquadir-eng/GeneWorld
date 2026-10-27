# Stage 2: robustness controls after the test (author review, 2026-10-27)

Review of `stage1` up to `369b3e7` (amendment 02 complete; test run once on 10-26).

**Status of what follows.** Everything in this file is **post hoc and exploratory**. The
pre-registered results stand as recorded in `results/a2/gates_val.json` and
`results/a2/gates_test.json`. The controls below test whether the G-Signal-2 pass means what the
paper would claim. Their interpretation rules (section 3) are fixed in this commit, before any
of them is run, and every outcome is reported whatever it shows.

## 1. Review: what holds

- Procedure:
  - selection was committed before any H2/H3 output, which came before `gates_val.json`, which
    came before the freeze `ce5c3df`;
  - test and screenB embed commits passed the allowed-paths check;
  - the test was evaluated once, with 36 steps and no failure.
- S18: all 656 re-embedded comparisons were byte-identical to the stored latents. The test data
  went through exactly the training-time pipeline.
- Checklist item 7 "FAIL" is benign: two files carry test-split gene names as annotations of
  training- and validation-plate wells, and no evaluation used them.
- **G-Signal-2 passed on test:** 770 primary wells on 82 plates. Action-conditioned beats
  action-agnostic by −5.65% (p = 1e-5), with the same direction on validation, under the global
  preprocessing, with DINOv2, and after the E1 B-score correction.

## 2. Review: what the gate does not yet establish

| # | issue | evidence |
|---|---|---|
| P1 | **No shuffled control for H1.** The H1 action-conditioned (`50M_d0.1_w0.2`) and action-agnostic (`50M_d0.3_w0.05`) models differ in regularisation as well as in action input. On unseen genes, H2 models fed *shuffled* encodings beat action-agnostic too (string, esm2: true − agnostic < 0 with true − shuffled ≈ 0). So part of the H1 gain may come from training differences, not gene information. | H2 tables, test and validation |
| P2 | **Replicate drift is too weak to test memorisation.** It is worse than copy-last (test primary 5.58 vs 5.27), so beating it by 46% says little. Meanwhile the replicate baselines carry **more** phenotype information than the action-conditioned forecast (probe AUROC lower for action-conditioned in 4 of 6 classes on test, e.g. binuclear −0.113 [−0.145, −0.078]). The obvious alternative, the action-agnostic forecast plus the siRNA's average residual on training plates, was never tested. | H1 tables and probe, test |
| P3 | **The effect did not replicate with independent reagents.** ScreenB (new siRNAs, siRNA token masked, global preprocessing): −0.76% vs agnostic, p = 0.10. The test gain may depend on the siRNA token, i.e. be reagent-specific (including off-target signatures), not gene-level. | `h1_screenB_global_vjepa2.json` |
| P4 | **G-ZeroShot-2 test pass is not robust.** One of five sources passes, the CI's upper bound is −0.0001, there is no correction across the five, and the same source fails on validation. For perturbseq and the JUMP sources, unseen-gene encodings make forecasts worse than no action. | H2 tables |
| P5 | G-Plan-2 fails on both splits; the planner is not distinguishable from kNN-imputed retrieval. | H3 tables |

P4 and P5 need no new analysis, only honest wording (section 4). P1–P3 need the controls below.

## 3. Controls (run now; exploratory, after the test)

All on V-JEPA 2, the control-normalised space, primary wells, the 24/36 h primary metric, and
the same validation (362 wells, 40 plates) and test (770 wells, 82 plates) sets. Use the same
statistics as G-Signal-2 (one-sided Monte Carlo sign flip over plates, errors averaged over 3
seeds). Report every number with "exploratory, after the test". Outputs go to
`results/a2/robustness/`.

**R1. Trained-shuffled H1 control (answers P1; GPU, about 1 h on 8 GPUs).**
- Train `h1_learned` at its selected configuration `50M_d0.1_w0.2`, 3 seeds, identical in every
  respect except the action labels.
- Labels: one fixed permutation π over the training siRNAs (`default_rng(20261027 + seed)`,
  hash recorded). Each well with siRNA s receives siRNA token π(s) and the gene embedding of
  gene(π(s)), identically in training and evaluation.
- This keeps the number of actions and the replicate consistency, and destroys the biological
  correspondence.
- Evaluate on validation and test. Also train `h1_none` at `50M_d0.1_w0.2` (3 seeds) as a
  config-matched agnostic reference.

**R2. Additive memorisation baseline (answers P2; inference only).**
- Prediction for evaluated well w with siRNA s at horizon h:
  `agnostic(w, h) + λ · mean_{w' ∈ training-plate wells of s} [obs(w', h) − agnostic(w', h)]`,
  using the gene's other siRNAs if s has none on training plates. The agnostic predictions are
  the seed-averaged selected `h1_none` models.
- Variants: λ = 1, and λ chosen on **validation** from {0, 0.25, 0.5, 0.75, 1} and then applied
  unchanged to test.
- Caveat to report: the residuals on training-plate wells are in-sample for the agnostic model.
- Also run the phenotype probe on R2's forecasts.

**R3. Reagent dependence (answers P3; inference only).** Evaluate the selected `h1_learned`
models on validation and test:
- (a) with the siRNA token zeroed (gene embedding kept), as on screenB;
- (b) with gene and siRNA labels permuted across wells within each plate, at evaluation time;
- (c) under the global preprocessing with the token zeroed, which is the closest
  like-for-like comparison with screenB.

Report the fraction of the test gain over agnostic retained under (a).

**R4. Where the gain sits (descriptive).** Give the gain over agnostic per horizon (3/12/24/36 h)
and per training-plate phenotype class of the primary wells. Also give it for non-primary
library wells.

## 3a. Interpretation rules (fixed before running)

| claim in the paper | requires (test; validation reported beside) |
|---|---|
| **I1. The action input carries gene information** (minimum for any gene-conditioning claim) | real `h1_learned` beats the R1 trained-shuffled model, p < 0.05 over plates |
| **I2. Beyond additive memorisation** (gene × state, the world-model claim) | I1, and real `h1_learned` beats R2 at the validation-chosen λ, p < 0.05 |
| **I3. Gene-level rather than reagent-level** | I1, and under R3(a) the gain over agnostic keeps >= 50% of its size and stays p < 0.05 |

- **If I1 fails:** the G-Signal-2 pass is reported as recorded, together with the statement that
  a trained-shuffled control matches it. No gene-conditioning claim is made, and the paper is a
  negative result.
- **If I1 holds but I2 fails:** the claim is "gene-conditioned forecasts, equivalent to an
  additive per-reagent correction".
- **I3:** worded by its outcome; if it fails, the screenB result is the explanation.

## 4. Wording fixed now, whatever R1–R4 show

- **G-ZeroShot-2:** "passes the pre-registered rule on test for one of five sources (string) with
  a CI bound at −0.0001; the rule has no correction across sources; the same source fails on
  validation. We do not claim zero-shot transfer."
- **G-Plan-2:** "fails on validation and test; planning is not distinguishable from kNN-imputed
  retrieval."
- **ScreenB:** reported in the main text, not only in the supplement.

## 5. Steps

1. Commit this file unchanged on `stage1` before any control runs (merge `plan-a02`).
2. Run R2, R3 and R4 (inference) on one GPU, and R1 (6 training runs) on the others, in parallel.
   Back up checkpoints and results with SHA-256, commit and push by explicit paths.
3. Write `reports/ROBUSTNESS_STAGE2.md`. For each rule, give the numbers and which of I1–I3 hold,
   applying the rules of section 3a mechanically. Include a short "what the paper can claim"
   paragraph in the section 3a wording.
4. Fill `paper/geneworld/` results and discussion from the results files only, through
   `scripts/91_paper_numbers.py` macros:
   - pre-registered results first (gates as recorded);
   - then the robustness controls, labelled post hoc;
   - the section 4 wording verbatim.

   Do not change the novelty claim. Build both PDFs.
5. Tell the author when all GPU work is done so the instance can be terminated. Nothing is
   posted or submitted.
