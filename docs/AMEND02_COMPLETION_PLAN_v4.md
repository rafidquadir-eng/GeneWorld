# Amendment 02: completion plan, v4 (author review, 2026-10-23 afternoon)

Review of `stage1` up to `4085075`: grid 261 of 288, S10 and S11 done, S12 orchestrator waiting
on the grid, CPU-side work done (E1–E3 code, run-report generator, manuscript draft). Extends v2
and v3; their rules still apply. Nothing here changes a gate, threshold, metric, preprocessing,
split, grid value, selection rule or frozen list. It settles the open points, corrects one item
of the author's own S13 checklist, and adds a pre-freeze rehearsal of the one-time test.

## 1. Checked and in order

- **S10:** apt timers off, services masked, 24 NVIDIA/CUDA packages held, needrestart
  list-only; the driver and Fabric Manager are unchanged.
- **S11:** the grid record has attempts per run (incidents 277 / 13 / 27), each labelled logged
  or reconstructed. The incident 2 count is corrected from 12 to 13.
- **S12 orchestrator:**
  - `selection.json` is committed before anything else;
  - dependencies are respected (H2 validation, then the planning source, then the
    complex-split runs);
  - a failed job blocks its dependents and the gates;
  - the GPU and exit code are recorded per job.
- **E1–E3 code:**
  - outputs are confined to `results/a2/exploratory/` and labelled;
  - nothing is run on real data;
  - E3 keeps the added training-plate wells out of training sets and replicate baselines
    (accepted).
- **Report generator:** takes numbers only from results files, words directions from p values
  and CIs in code, and leaves placeholders for test results.
- **Manuscript draft:**
  - every number is a macro;
  - no result or superiority claim appears ahead of results;
  - the novelty search is dated, with every reference verified and unverifiable items left out,
    and the novelty claim is worded as the outcome of that dated search.
- **Agent's own errors** (wrong PID killed; a background `&`) were caught before any job or
  commit; launches are now verified by PID.

## 2. Decisions on the agent's open points

| # | point | decision |
|---|---|---|
| D1 | **S13 checklist item 4 expected 502 H1 primary validation wells; 362 are evaluated.** The 535 frozen wells comprise wells of training-split genes, wells of validation- and test-split genes, and S2 exclusions. H1 asks whether the action-conditioned model forecasts **seen genes on held-out plates** (design A.4: "within seen genes, whole plates are held out for H1"). The learned per-gene embedding has no entry for held-out-split genes, which belong to H2. | **The author's checklist was wrong; the implementation is right.** Record in the protocol, before the freeze, as a clarification (not an amendment): *H1 primary wells = frozen H1 primary wells whose gene is in the training gene split, minus S2/S3 exclusions.* Expected: validation **362** (386 training-gene wells − 24 S2 exclusions); test **<= 770** wells on <= 82 plates (783 training-gene wells, 13 outside the test layout), minus S3 exclusions. Item 4 is replaced (section 4). Wells of held-out-split genes stay in H2. |
| D2 | **Test-time embedding.** `run_stage1_test.sh` embeds with one embedder, while S14 needs one per GPU (BIA fetchers sharing <= 128 streams in total). Its `cache.build(..., '_test')` would be refused by the shard guard. | Change the script **before the freeze** (S17). The order becomes: freeze commit → embed → **embed commit** → evaluation. The embed commit adds the test (and later screenB) shards to `manifests/cache_shards.tsv` with SHA-256, plus the embed failure lists and the S3 Fisher tables, and **no evaluation output**. Before evaluating, the script checks that the files changed between the freeze commit and the embed commit are only the manifest, the failure lists, the S3 tables and logs; otherwise it refuses. This replaces the author's earlier "in the freeze commit". |
| D3 | S3 test specified as two-sided Fisher exact per class (mitotic delay, polylobed, binuclear, any reproducible), BH over the 4, descriptive only. | Accepted. |
| D4 | E3 keeps the added training-plate wells out of training and baselines; only validation and test wells are added to the evaluated set. | Accepted. |

## 3. New problem: the test data would be embedded on different hardware, and nothing checks it

All training and validation latents were embedded on the old 1 x A100 instance (stage 1 on
10-19 to 10-20, stage 2 on 10-21). The test plates and screenB will be embedded on the
8 x A100 instance, with the fetcher changed since (404 retries) and per-GPU embedders.

V-JEPA 2 runs in bf16, so different GPUs, batch compositions or library heuristics can give
slightly different latents. A shift that affects **only** the test latents would bias every
test comparison against the frozen models, the baselines being computed from training latents.
Nothing yet checks for it. This is the one problem that can silently spoil the one-time test.

**Fix: S18, a rehearsal on data already used, before the freeze.** It must pass before the S13
summary goes out.

## 4. Steps (in order; S12 continues unchanged)

**S17. Make the test script ready (CPU work now; commit before the freeze).**
Implement D2 in `scripts/run_stage1_test.sh`:
- per-GPU embedders, <= 128 streams in total;
- test shards to `data/latents_test/`, screenB shards to `data/latents_screenB/`;
- the embed commit and the allowed-paths check before any evaluation.

Add a `--rehearsal PLATES` mode for S18 that runs the identical code path:
- on a given list of **validation** plates, re-embedded into `data/latents_rehearsal/`;
- outputs to `results/a2/rehearsal/`;
- a rehearsal manifest, separate from the real one;
- it can never touch test or screenB lists, and refuses if any test or screenB shard exists.

Unit-test the allowed-paths check and the refusal conditions. Record the D1 clarification in
`manifests/amend02_protocol.json`. Commit by explicit paths (G4).

**S18. Rehearsal (GPU; run as soon as GPUs are free without delaying S12's gates, at the latest
during the S13 wait).** Using `--rehearsal`:
1. Re-embed a seeded sample of already-embedded wells with the exact test-time embedder on the
   new instance: 3 whole validation plates (one from stage 1, two from stage 2) plus 100 random
   training wells across stage 1 and stage 2. Record which GPUs ran them.
2. Compare with the stored latents, per well and clip, in raw space and in the control-normalised
   space. Report: byte identity, max abs difference, relative L2 difference, and the mean squared
   difference as a fraction of the stage-1 H1 action-conditioned validation error level
   (`results/h1/h1_val_vjepa2.json`, a stage-1 number already known).
3. **Criterion (fixed now):** pass if the latents are byte-identical, or if the mean squared
   re-embedding difference in the normalised space is <= 1% of that error level **and** shows
   no systematic shift (the mean difference per dimension is within the noise, by a sign-flip
   test over wells at p > 0.01 after BH over dimensions). If it fails, stop and report to the
   author with the numbers. Do not change the embedder.
4. Run the full rehearsal path: the embed commit check, the cache build with the rehearsal
   manifest, and Parts A–C evaluation code on the 3 rehearsal plates (frozen models not
   required; use the selected models once `selection.json` exists, otherwise stage-1 models).
   It must finish without errors. Its numbers are validation numbers and are not reported as
   results.
5. Write `results/a2/rehearsal/summary.json` and commit.

**S13 (unchanged except the checklist).** Item 4 becomes: *H1 primary validation wells evaluated
= 362 = frozen validation H1 primary wells of training-split genes (386) minus S2 exclusions
(24), with the decomposition of all 535 shown.* Add:
- **item 9:** S18 passed (link to the summary; byte-identical or within the criterion);
- **item 10:** S17 committed and its tests pass; `run_stage1_test.sh` has run only in rehearsal
  mode.

Send the gate outcomes, the checklist and the S18 summary to the author, then **wait for "go"**.

**After "go": v3 S14–S16 unchanged**, with the D2 order.

## 5. Author items (not for the agent)

Fill the manuscript author block: name, e-mail, affiliation (or "Independent researcher") and
ORCID. Code and data availability: the repository becomes public only on the author's decision.
