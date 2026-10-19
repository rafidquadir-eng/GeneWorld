#!/usr/bin/env bash
# One-time test evaluation of the frozen stage-1 models. Requires a clean, pushed working tree;
# the current commit is recorded as the freeze commit in every test result.
set -eu
cd "$(dirname "$0")/.."
P=~/venv/bin/python
L=logs/stage1_test
mkdir -p $L
if [ -n "$(git status --porcelain -- src scripts splits manifests docs)" ]; then
  echo "uncommitted changes: commit the frozen state first"; exit 1
fi
FREEZE=$(git rev-parse --short HEAD)
BR=$(git rev-parse --abbrev-ref HEAD); [ "$BR" != main ] && git fetch -q origin && git merge-base --is-ancestor "$FREEZE" "origin/$BR" || {
  echo "freeze commit $FREEZE is not pushed"; exit 1; }
for f in results/h1/h1_test_*.json results/h2/h2_test_*.json results/h3/h3_test_*.json; do
  [ -e "$f" ] && { echo "test results already exist ($f): test splits are evaluated once"; exit 1; }
done
SEL=results/train/h1_selection.json
BL=$($P -c "import json;print(json.load(open('$SEL'))['selected_learned'])")
BN=$($P -c "import json;print(json.load(open('$SEL'))['selected_agnostic'])")
SRC=$($P -c "import json;r=json.load(open('results/h2/h2_val_vjepa2.json'));print(r['G_ZeroShot']['selected_source'] or min(r['sources'],key=lambda s:r['sources'][s]['true_vs_shuffled']['mean']))")
echo "freeze $FREEZE: H1 $BL vs $BN; planning source $SRC" | tee $L/pipeline.log
$P scripts/41_eval_h1.py --model $BL --agnostic $BN --split test --freeze-commit $FREEZE > $L/h1.log 2>&1
for bb in dinov2 tijepa1; do
  $P scripts/41_eval_h1.py --backbone $bb --model h1_learned_$bb --agnostic h1_none_$bb \
     --split test --freeze-commit $FREEZE > $L/h1_$bb.log 2>&1
done
MODELS=$(for s in perturbseq jump_crispr jump_orf string esm2; do printf "%s=h2_%s," $s $s; done)
$P scripts/42_eval_h2.py --models ${MODELS%,} --agnostic $BN --split test --freeze-commit $FREEZE > $L/h2.log 2>&1
$P scripts/43_plan.py --source $SRC --model h2_$SRC --split test --freeze-commit $FREEZE > $L/h3.log 2>&1
$P scripts/42_eval_h2.py --gene-split complex --models $SRC=h2c_$SRC --agnostic h2c_none \
   --split test --freeze-commit $FREEZE --tag _complex > $L/h2_complex.log 2>&1
$P scripts/43_plan.py --gene-split complex --source $SRC --model h2c_$SRC --split test \
   --freeze-commit $FREEZE --tag _complex > $L/h3_complex.log 2>&1
echo "test evaluation done (freeze $FREEZE); GPU idle" | tee -a $L/pipeline.log
