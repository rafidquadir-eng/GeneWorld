# GeneWorld stage 2: robustness controls, completion summary (2026-10-30)

This summary covers the robustness work set out in `docs/STAGE2_ROBUSTNESS_PLAN.md`
(`78c8775`) and the manuscript update that followed. The full numbers are in
`reports/ROBUSTNESS_STAGE2.md`, generated from `results/a2/robustness/` by
`scripts/94_robustness_report.py`. Every number below is copied from that report. All of it is
**exploratory, after the test**. The pre-registered results stand as recorded in
`reports/STAGE2_COMPLETION_SUMMARY.md`; no gate, threshold, metric, preprocessing step, split,
grid value, selection or frozen file was changed. All GPU work finished on 10-28; the instance
can be terminated.

## Outcome

The pre-registered G-Signal-2 pass is real, but a trained-shuffled control matches it. Under
the section 3a rules, applied mechanically, the paper makes **no gene-conditioning claim and is
a negative result**.

| rule | test | validation | deciding numbers (test) |
|---|---|---|---|
| I1: real beats trained-shuffled, p < 0.05 | does not hold | does not hold | real − shuffled −0.10%, p = 0.388 (validation +0.17%, p = 0.567) |
| I2: I1, and real beats R2 at the validation λ | does not hold | does not hold | λ = 0.25; real − R2 +2.19% (validation +1.59%) |
| I3: I1, and R3(a) keeps ≥ 50% of the gain at p < 0.05 | does not hold | does not hold | fails because I1 fails; the R3(a) part alone holds: 71.0% retained, p = 1e-05 (validation 78.7%) |

## Controls (test, 770 primary wells on 82 plates)

- **R1, trained-shuffled control.** Retrained at `50M_d0.1_w0.2` with permuted siRNA labels,
  3 seeds. Compared with the selected action-agnostic models:
  - the shuffled model's error is 5.56% lower (p = 1e-05), which is 98.4% of the real model's
    gain (−5.65%);
  - the config-matched action-agnostic model differs by −0.00% (p = 0.496), so the
    regularisation difference explains none of the gain.
- **R2, additive memorisation.** The agnostic forecast plus 0.25 × the siRNA's
  training-plate residual has 7.67% lower error than agnostic, which beats the real model.
- **R3, reagent dependence.**
  - (a) With the siRNA token zeroed, the error is 4.01% lower than agnostic (71.0% of the gain
    kept).
  - (b) With labels permuted within each plate, the error is 5.53% *higher* than agnostic.
  - (c) Under the global preprocessing with the token zeroed, the error is 2.02% lower than
    agnostic (p = 0.020; 62.8% of the gain kept).
- **R4, where the gain sits.**
  - No gain at 3 h (+0.74%) or 12 h (−0.07%, p = 0.442).
  - Gain at 24 h (−3.93%) and 36 h (−7.03%).
  - Gain in every training-plate phenotype class (−5.1% to −9.4%), and on non-primary library
    wells (−2.45%).

Before writing a control number, each results file recomputed the real model's comparison with
the agnostic models and checked it against the recorded `results/a2/h1` file. All matched.

## Choices the plan left open (recorded in `scripts/93_robustness.py` and the results files)

1. **R1, siRNAs absent from the training plates.** These wells keep the unknown token. They take
   the permuted gene of their gene's first training siRNA. No primary well is affected.
2. **R2 residual.** The mean over the four tiles, as the replicate baselines use. The
   residuals are in-sample for the agnostic model on training plates; the results file states
   this caveat.
3. **R3(b) permutation.** Labels are permuted across the evaluated library wells of each plate,
   with `default_rng(20261027)`.
4. **R1 GPUs.** The 6 trainings ran on GPUs 1–6, so GPU 7 was idle. R2–R4 ran on GPU 0 in
   parallel.

## Manuscript (`paper/geneworld/`)

- The results and discussion are filled from the results files through
  `scripts/91_paper_numbers.py` (199 macros in `numbers.tex`). The order is:
  1. the pre-registered gates, with the gates table and screen B in the main text;
  2. the robustness controls, labelled post hoc;
  3. the section 4 wording for zero-shot transfer and planning, verbatim.
- The abstract and contribution (iv) state the negative result. The novelty claim is unchanged.
- No internal IDs or paths appear in the text.
- Both PDFs are built with tectonic, with no unresolved references:
  `paper/geneworld/pdf/main_anon.pdf` and `main_biorxiv.pdf`.

## Commits (branch `stage1`)

| commit | content |
|---|---|
| `78c8775` | robustness plan merged from `plan-a02` and pushed before anything ran |
| `d4d0b81` | `scripts/93_robustness.py` and the driver `logs/robustness/run.sh` |
| `2cc7744` | results in `results/a2/robustness/`, 6 checkpoints in `checkpoints/a2_robust/`, logs |
| `ecdd7a4` | `reports/ROBUSTNESS_STAGE2.md`; new macros in `scripts/91_paper_numbers.py` |
| `8676ccc` | manuscript results and discussion; both PDFs |

Backups: every new results file, checkpoint, report and manuscript file is on
`~/LambdaFilesystem/geneworld_backups/`, verified by SHA-256 and listed in
`reports/BACKUP_LEDGER.tsv`.

## Open items for the author

1. **Manuscript marks:**
   - `body.tex`: review the pre-registration section, write the code and data availability
     statement, and review the AI-disclosure sentence;
   - `main_biorxiv.tex`: contact e-mail and affiliation.
2. **S13 items 7 and 9**, unchanged from `reports/STAGE2_COMPLETION_SUMMARY.md`.
3. **The completion summary's "Manuscript" item is superseded:** the results and discussion are
   no longer placeholders.

Nothing has been submitted, posted, published or released, and nothing was pushed to `main`.
