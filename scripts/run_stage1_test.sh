#!/usr/bin/env bash
# One-time test evaluation (amendment 02 section 8). Requires a clean, pushed, non-main freeze
# commit; that commit is recorded in every test result. Nothing is changed after this runs.
#   Part A: the frozen stage-1 models on stage-1 test data (the pre-amendment analysis, as
#           registered in stage 1).
#   Part B: amendment 02: embed the test plates (native fetcher, frozen layout), build caches
#           that add the test plates (training wells and their statistics unchanged; checked),
#           then every amended evaluation and the gates, once.
set -eu
cd "$(dirname "$0")/.."
P=~/venv/bin/python
L=logs/stage1_test
mkdir -p $L
if [ -n "$(git status --porcelain -- src scripts splits manifests docs)" ]; then
  echo "uncommitted changes: commit the frozen state first"; exit 1
fi
FREEZE=$(git rev-parse --short HEAD)
BR=$(git rev-parse --abbrev-ref HEAD)
[ "$BR" != main ] && git fetch -q origin && git merge-base --is-ancestor "$FREEZE" "origin/$BR" || {
  echo "freeze commit $FREEZE is not pushed (or is on main)"; exit 1; }
[ -e results/a2/gates_val.json ] || { echo "validation gates not recorded"; exit 1; }
for f in results/h1/h1_test_*.json results/h2/h2_test_*.json results/h3/h3_test_*.json \
         results/a2/h1/h1_test_*.json results/a2/h2/h2_test_*.json results/a2/h3/h3_test_*.json \
         results/a2/gates_test.json; do
  [ -e "$f" ] && { echo "test results already exist ($f): test splits are evaluated once"; exit 1; }
done
echo "freeze $FREEZE on $BR" | tee $L/pipeline.log

# ---------------- Part A: stage-1 frozen models
SEL=results/train/h1_selection.json
BL=$($P -c "import json;print(json.load(open('$SEL'))['selected_learned'])")
BN=$($P -c "import json;print(json.load(open('$SEL'))['selected_agnostic'])")
SRC1=$($P -c "import json;r=json.load(open('results/h2/h2_val_vjepa2.json'));print(r['G_ZeroShot']['selected_source'] or min(r['sources'],key=lambda s:r['sources'][s]['true_vs_shuffled']['mean']))")
echo "A: stage-1 H1 $BL vs $BN; planning source $SRC1" | tee -a $L/pipeline.log
$P scripts/41_eval_h1.py --model $BL --agnostic $BN --split test --freeze-commit $FREEZE > $L/s1_h1.log 2>&1
for bb in dinov2 tijepa1; do
  $P scripts/41_eval_h1.py --backbone $bb --model h1_learned_$bb --agnostic h1_none_$bb \
     --split test --freeze-commit $FREEZE > $L/s1_h1_$bb.log 2>&1
done
MODELS=$(for s in perturbseq jump_crispr jump_orf string esm2; do printf "%s=h2_%s," $s $s; done)
$P scripts/42_eval_h2.py --models ${MODELS%,} --agnostic $BN --split test --freeze-commit $FREEZE > $L/s1_h2.log 2>&1
$P scripts/43_plan.py --source $SRC1 --model h2_$SRC1 --split test --freeze-commit $FREEZE > $L/s1_h3.log 2>&1
$P scripts/42_eval_h2.py --gene-split complex --models $SRC1=h2c_$SRC1 --agnostic h2c_none \
   --split test --freeze-commit $FREEZE --tag _complex > $L/s1_h2_complex.log 2>&1
$P scripts/43_plan.py --gene-split complex --source $SRC1 --model h2c_$SRC1 --split test \
   --freeze-commit $FREEZE --tag _complex > $L/s1_h3_complex.log 2>&1
echo "A done" | tee -a $L/pipeline.log

# ---------------- Part B: amendment 02
echo "B: embedding test plates ($(($(wc -l < splits/test_wells_to_embed.tsv) - 1)) wells)" | tee -a $L/pipeline.log
$P scripts/20_stream_embed.py --wells splits/test_wells_to_embed.tsv --fetcher native \
   --out-root data/latents_test > $L/embed_test.log 2>&1
for bb in vjepa2 dinov2 tijepa1; do
  $P -c "import sys; sys.path.insert(0,'src'); from geneworld import cache
print(cache.build('$bb', roots=['data/latents','data/latents_s2','data/latents_test'],
      tsvs=['splits/stage1_wells.tsv','splits/stage2_wells.tsv','splits/stage2_test_wells.tsv'],
      include_test=True, suffix='_test'))" > $L/cache_$bb.log 2>&1
done
T="--split test --suffix _test --freeze-commit $FREEZE"
$P scripts/61_eval_h1_a2.py $T > $L/h1.log 2>&1
$P scripts/62_eval_h2_a2.py $T > $L/h2.log 2>&1
$P scripts/63_plan_a2.py $T > $L/h3.log 2>&1
$P scripts/61_eval_h1_a2.py $T --norm global > $L/h1_global.log 2>&1
$P scripts/62_eval_h2_a2.py $T --norm global > $L/h2_global.log 2>&1
$P scripts/63_plan_a2.py $T --norm global > $L/h3_global.log 2>&1
for bb in dinov2 tijepa1; do
  $P scripts/61_eval_h1_a2.py $T --backbone $bb > $L/h1_$bb.log 2>&1
done
SRC=$($P -c "import json;print(json.load(open('results/a2/h2/h2_val_plate_vjepa2.json'))['planning_source'])")
$P scripts/62_eval_h2_a2.py $T --gene-split complex --sources $SRC > $L/h2_complex.log 2>&1
$P scripts/63_plan_a2.py $T --gene-split complex > $L/h3_complex.log 2>&1
$P scripts/64_controls_a2.py --split test --suffix _test --norm plate --freeze-commit $FREEZE > $L/controls_plate.log 2>&1
$P scripts/64_controls_a2.py --split test --suffix _test --norm global --freeze-commit $FREEZE > $L/controls_global.log 2>&1
$P scripts/65_gates_a2.py --split test > $L/gates.log 2>&1
cat $L/gates.log | tee -a $L/pipeline.log
echo "test evaluation done (freeze $FREEZE); GPU idle" | tee -a $L/pipeline.log
