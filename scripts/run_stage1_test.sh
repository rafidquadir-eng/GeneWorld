#!/usr/bin/env bash
# One-time test evaluation (amendment 02 section 8; completion plan v4 S17, decision D2).
#
#   scripts/run_stage1_test.sh                       the test, once
#   scripts/run_stage1_test.sh --rehearsal P1,P2,P3  rehearsal on validation plates (v4 S18)
#
# Order (D2): freeze commit (HEAD, clean, pushed, non-main) -> embed the test plates (one
# embedder per GPU, BIA fetchers sharing <= 128 streams) -> embed commit (shard manifest rows
# with SHA-256, embed failure list, S3 tables, embed logs; no evaluation output) ->
# allowed-paths check between the freeze and the embed commit -> evaluation:
#   Part A: the frozen stage-1 models on stage-1 test data (the pre-amendment analysis, as
#           registered in stage 1).
#   Part B: amendment 02: caches that add the test plates (training wells and their statistics
#           unchanged; checked), every amended evaluation and the gates, once.
#   Part C: screenB (author instruction 2026-10-21): embed screenB, embed commit and check
#           as above, then H1 (seen genes) and H2 (test-split genes) under the global
#           preprocessing only, outside the gates, siRNA token masked for screenB's siRNAs.
# Refusal conditions: scripts/79_embed_commit.py preflight.
#
# Rehearsal (GENEWORLD_REHEARSAL=1): the identical embed / commit / check path on the given
# validation plates (shards in data/latents_rehearsal/, own manifest, outputs in
# results/a2/rehearsal/), plus 100 training wells for the re-embedding comparison; the
# comparison must pass before any evaluation code runs. The plates play the test plates (and
# screenB) in caches built without any test well; H2 and H3 run with --split val because their
# test masks select test-split genes, which the rehearsal must not read; Part C runs H1 only for
# the same reason. Refuses if any test or screenB shard exists.
set -eu
cd "$(dirname "$0")/.."
P=~/venv/bin/python
MODE=test
PLATES=""
if [ "${1:-}" = "--rehearsal" ]; then
  MODE=rehearsal
  PLATES=${2:?usage: --rehearsal P1,P2,P3}
fi
if [ $MODE = test ]; then L=logs/stage1_test; else L=logs/rehearsal; fi
mkdir -p $L
log() { echo "[$(date -u "+%m-%d %H:%M:%S")] $*" | tee -a $L/pipeline.log; }
# run one step; record name, exit code and seconds; stop on failure
step() {
  local name=$1; shift
  local t0=$(date +%s) rc=0
  "$@" > $L/$name.log 2>&1 || rc=$?
  printf "%s\t%s\t%s\n" "$name" "$rc" "$(( $(date +%s) - t0 ))" >> $L/steps.tsv
  [ $rc = 0 ] || { log "step $name FAILED (exit $rc); see $L/$name.log"; exit 1; }
}
[ -e $L/steps.tsv ] || printf "step\texit_code\tseconds\n" > $L/steps.tsv
$P scripts/79_embed_commit.py preflight --mode $MODE --plates "$PLATES"
FREEZE=$(git rev-parse --short HEAD)
log "$MODE: freeze $FREEZE on $(git rev-parse --abbrev-ref HEAD)"

embed() {   # embed <split> <wells.tsv> <out-root> <tag> <failures.json>
  step embed_$4 $P scripts/78_parallel_embed.py --wells "$2" --out-root "$3" --tag "$4" \
       --failures "$5" --streams-total 128
}
embed_commit() {   # embed_commit <split> <wells.tsv> <failures.json> <base> <log dirs>
  step s3_$1 $P scripts/76_s3_accounting.py --split $1 --wells "$2" --failures "$3"
  step manifest_$1 $P scripts/79_embed_commit.py manifest --split $1
  step commit_$1 $P scripts/79_embed_commit.py commit --split $1 --logs "$5"
  step check_$1 $P scripts/79_embed_commit.py check --split $1 --base $4 --head HEAD
  log "embed commit $(git rev-parse --short HEAD) ($1); allowed-paths check passed"
}
build() {   # build <backbone> <suffix> <include_test> <roots> <tsvs> [exclude-python-expr]
  $P -c "import sys; sys.path.insert(0,'src'); from geneworld import cache
print(cache.build('$1', roots=$4, tsvs=$5, include_test=$3, suffix='$2', exclude=${6:-None}))"
}

if [ $MODE = test ]; then
# ---------------- Part B embedding first (D2 order), then Part A and Part B evaluation
log "B: embedding test plates ($(($(wc -l < splits/test_wells_to_embed.tsv) - 1)) wells)"
embed test splits/test_wells_to_embed.tsv data/latents_test test_embed \
      results/a2/embed_failures_test.json
embed_commit test splits/test_wells_to_embed.tsv results/a2/embed_failures_test.json $FREEZE \
             test_embed
EMBED_TEST=$(git rev-parse --short HEAD)

# ---------------- Part A: stage-1 frozen models
SEL=results/train/h1_selection.json
BL=$($P -c "import json;print(json.load(open('$SEL'))['selected_learned'])")
BN=$($P -c "import json;print(json.load(open('$SEL'))['selected_agnostic'])")
SRC1=$($P -c "import json;r=json.load(open('results/h2/h2_val_vjepa2.json'));print(r['G_ZeroShot']['selected_source'] or min(r['sources'],key=lambda s:r['sources'][s]['true_vs_shuffled']['mean']))")
log "A: stage-1 H1 $BL vs $BN; planning source $SRC1"
step s1_h1 $P scripts/41_eval_h1.py --model $BL --agnostic $BN --split test --freeze-commit $FREEZE
for bb in dinov2 tijepa1; do
  step s1_h1_$bb $P scripts/41_eval_h1.py --backbone $bb --model h1_learned_$bb \
       --agnostic h1_none_$bb --split test --freeze-commit $FREEZE
done
MODELS=$(for s in perturbseq jump_crispr jump_orf string esm2; do printf "%s=h2_%s," $s $s; done)
step s1_h2 $P scripts/42_eval_h2.py --models ${MODELS%,} --agnostic $BN --split test --freeze-commit $FREEZE
step s1_h3 $P scripts/43_plan.py --source $SRC1 --model h2_$SRC1 --split test --freeze-commit $FREEZE
step s1_h2_complex $P scripts/42_eval_h2.py --gene-split complex --models $SRC1=h2c_$SRC1 \
     --agnostic h2c_none --split test --freeze-commit $FREEZE --tag _complex
step s1_h3_complex $P scripts/43_plan.py --gene-split complex --source $SRC1 --model h2c_$SRC1 \
     --split test --freeze-commit $FREEZE --tag _complex
log "A done"

# ---------------- Part B: amendment 02 evaluation
for bb in vjepa2 dinov2 tijepa1; do
  step cache_$bb build $bb _test True "['data/latents','data/latents_s2','data/latents_test']" \
       "['splits/stage1_wells.tsv','splits/stage2_wells.tsv','splits/stage2_test_wells.tsv']"
done
T="--split test --suffix _test --freeze-commit $FREEZE"
step h1 $P scripts/61_eval_h1_a2.py $T
step h2 $P scripts/62_eval_h2_a2.py $T
step h3 $P scripts/63_plan_a2.py $T
step h1_global $P scripts/61_eval_h1_a2.py $T --norm global
step h2_global $P scripts/62_eval_h2_a2.py $T --norm global
step h3_global $P scripts/63_plan_a2.py $T --norm global
for bb in dinov2 tijepa1; do
  step h1_$bb $P scripts/61_eval_h1_a2.py $T --backbone $bb
done
SRC=$($P -c "import json;print(json.load(open('results/a2/h2/h2_val_plate_vjepa2.json'))['planning_source'])")
step h2_complex $P scripts/62_eval_h2_a2.py $T --gene-split complex --sources $SRC
step h3_complex $P scripts/63_plan_a2.py $T --gene-split complex
step controls_plate $P scripts/64_controls_a2.py --split test --suffix _test --norm plate --freeze-commit $FREEZE
step controls_global $P scripts/64_controls_a2.py --split test --suffix _test --norm global --freeze-commit $FREEZE
step gates $P scripts/65_gates_a2.py --split test
cat $L/gates.log | tee -a $L/pipeline.log

# ---------------- Part C: screenB, global preprocessing only, outside the gates
log "C: embedding screenB ($(($(wc -l < splits/screenB_wells.tsv) - 1)) wells)"
embed screenB splits/screenB_wells.tsv data/latents_screenB screenB_embed \
      results/a2/embed_failures_screenB.json
embed_commit screenB splits/screenB_wells.tsv results/a2/embed_failures_screenB.json \
             $(git rev-parse --short HEAD) screenB_embed
step cache_screenB build vjepa2 _screenB False \
     "['data/latents','data/latents_s2','data/latents_screenB']" \
     "['splits/stage1_wells.tsv','splits/stage2_wells.tsv','splits/screenB_wells.tsv']"
B="--split screenB --norm global --suffix _screenB --freeze-commit $FREEZE"
step h1_screenB $P scripts/61_eval_h1_a2.py $B
step h2_screenB $P scripts/62_eval_h2_a2.py $B
tail -1 $L/h1_screenB.log | tee -a $L/pipeline.log
log "test evaluation done (freeze $FREEZE, test embed commit $EMBED_TEST); GPU idle"
exit 0
fi

# ================= rehearsal (v4 S18) =================
export GENEWORLD_REHEARSAL=1
R=results/a2/rehearsal
mkdir -p $R/partA
cp results/a2/selection.json $R/selection.json
REH="rehearsal-$FREEZE"
step wells $P scripts/77_rehearsal.py wells --plates $PLATES
embed rehearsal logs/rehearsal_embed/rehearsal_plates.tsv data/latents_rehearsal/plates \
      rehearsal_embed_plates $R/embed_failures_rehearsal.json
embed rehearsal logs/rehearsal_embed/rehearsal_train.tsv data/latents_rehearsal/train \
      rehearsal_embed_train logs/rehearsal_embed_train/failures.json
embed_commit rehearsal logs/rehearsal_embed/rehearsal_plates.tsv $R/embed_failures_rehearsal.json \
             $FREEZE rehearsal_embed,rehearsal_embed_plates,rehearsal_embed_train
step compare $P scripts/77_rehearsal.py compare
step tables $P scripts/77_rehearsal.py tables --plates $PLATES

# Part A code (stage-1 models) on validation, outputs moved to $R/partA
SEL=results/train/h1_selection.json
BL=$($P -c "import json;print(json.load(open('$SEL'))['selected_learned'])")
BN=$($P -c "import json;print(json.load(open('$SEL'))['selected_agnostic'])")
SRC1=$($P -c "import json;r=json.load(open('results/h2/h2_val_vjepa2.json'));print(r['G_ZeroShot']['selected_source'] or min(r['sources'],key=lambda s:r['sources'][s]['true_vs_shuffled']['mean']))")
for f in results/h1/h1_val_*_rehearsal.json results/h2/h2_val_*_rehearsal.json \
         results/h3/h3_val_*_rehearsal.json; do
  [ -e "$f" ] && { log "refusing: $f exists"; exit 1; }
done
step s1_h1 $P scripts/41_eval_h1.py --model $BL --agnostic $BN --split val --tag _rehearsal
for bb in dinov2 tijepa1; do
  step s1_h1_$bb $P scripts/41_eval_h1.py --backbone $bb --model h1_learned_$bb \
       --agnostic h1_none_$bb --split val --tag _rehearsal
done
MODELS=$(for s in perturbseq jump_crispr jump_orf string esm2; do printf "%s=h2_%s," $s $s; done)
step s1_h2 $P scripts/42_eval_h2.py --models ${MODELS%,} --agnostic $BN --split val --tag _rehearsal
step s1_h3 $P scripts/43_plan.py --source $SRC1 --model h2_$SRC1 --split val --tag _rehearsal
step s1_h2_complex $P scripts/42_eval_h2.py --gene-split complex --models $SRC1=h2c_$SRC1 \
     --agnostic h2c_none --split val --tag _complex_rehearsal
step s1_h3_complex $P scripts/43_plan.py --gene-split complex --source $SRC1 --model h2c_$SRC1 \
     --split val --tag _complex_rehearsal
mv results/h1/h1_val_*_rehearsal.json results/h2/h2_val_*_rehearsal.json \
   results/h3/h3_val_*_rehearsal.json $R/partA/

# Part B code: the plates play the test plates; no test well is in these caches
EXCL="{f'data/{r}/{b}/{p}.npz' for r in ('latents','latents_s2') for b in ('vjepa2','dinov2','tijepa1') for p in '$PLATES'.split(',')}"
for bb in vjepa2 dinov2 tijepa1; do
  step cache_$bb build $bb _rehearsal True \
       "['data/latents','data/latents_s2','data/latents_rehearsal/plates']" \
       "['logs/rehearsal/rehearsal_test_wells.tsv','logs/rehearsal/stage1_wells_no_test.tsv','splits/stage2_wells.tsv']" \
       "$EXCL"
done
T="--split test --suffix _rehearsal --freeze-commit $REH"
V="--split val --suffix _rehearsal"
step h1 $P scripts/61_eval_h1_a2.py $T
step h2 $P scripts/62_eval_h2_a2.py $V
step h3 $P scripts/63_plan_a2.py $V
step h1_global $P scripts/61_eval_h1_a2.py $T --norm global
step h2_global $P scripts/62_eval_h2_a2.py $V --norm global
step h3_global $P scripts/63_plan_a2.py $V --norm global
for bb in dinov2 tijepa1; do
  step h1_$bb $P scripts/61_eval_h1_a2.py $T --backbone $bb
done
SRC=$($P -c "import json;print(json.load(open('$R/h2/h2_val_plate_vjepa2.json'))['planning_source'])")
step h2_complex $P scripts/62_eval_h2_a2.py $V --gene-split complex --sources $SRC
step h3_complex $P scripts/63_plan_a2.py $V --gene-split complex
step controls_plate $P scripts/64_controls_a2.py --split test --suffix _rehearsal --norm plate --freeze-commit $REH
step controls_global $P scripts/64_controls_a2.py --split test --suffix _rehearsal --norm global --freeze-commit $REH
# the gate collector's test path, fed with the rehearsal's validation-split H2/H3 files
cp $R/h2/h2_val_plate_vjepa2.json $R/h2/h2_test_plate_vjepa2.json
cp $R/h3/h3_val_plate_vjepa2_$SRC.json $R/h3/h3_test_plate_vjepa2_$SRC.json
step gates $P scripts/65_gates_a2.py --split test

# Part C code (H1 only): the plates play screenB
step cache_screenB build vjepa2 _rehearsal_screenB False \
     "['data/latents','data/latents_s2','data/latents_rehearsal/plates']" \
     "['logs/rehearsal/rehearsal_screenB_wells.tsv','logs/rehearsal/stage1_wells_no_test.tsv','splits/stage2_wells.tsv']" \
     "$EXCL"
step h1_screenB $P scripts/61_eval_h1_a2.py --split screenB --norm global --suffix _rehearsal_screenB --freeze-commit $REH
step summary $P scripts/77_rehearsal.py summary
log "rehearsal done (base $FREEZE)"
