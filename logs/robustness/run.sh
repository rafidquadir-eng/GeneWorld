#!/usr/bin/env bash
# Robustness controls R1-R4 (docs/STAGE2_ROBUSTNESS_PLAN.md, plan commit 78c8775).
# R1: 6 training runs on GPUs 1-6 (one each), then evaluation; R2-R4 on GPU 0 (val, then test,
# which uses the validation-chosen lambda). Exploratory, after the test.
set -u
cd "$(dirname "$0")/../.."
P=~/venv/bin/python
L=logs/robustness
log() { echo "[$(date -u "+%m-%d %H:%M:%S")] $*" | tee -a $L/pipeline.log; }
[ -e $L/steps.tsv ] || printf "step\texit_code\tseconds\n" > $L/steps.tsv
run() {   # run <name> <gpu> cmd...
  local name=$1 gpu=$2; shift 2
  local t0=$(date +%s) rc=0
  CUDA_VISIBLE_DEVICES=$gpu "$@" > $L/$name.log 2>&1 || rc=$?
  printf "%s\t%s\t%s\n" "$name" "$rc" "$(( $(date +%s) - t0 ))" >> $L/steps.tsv
  [ $rc = 0 ] && log "done $name (GPU $gpu)" || log "step $name FAILED (exit $rc); see $L/$name.log"
  return $rc
}
log "start (code $(git rev-parse --short HEAD))"
( run r234_val 0 $P scripts/93_robustness.py r234 --split val &&
  run r234_test 0 $P scripts/93_robustness.py r234 --split test ) &
R234=$!
pids=(); g=1
for fam in h1_learned h1_none; do for s in 0 1 2; do
  run r1_train_${fam}_s$s $g $P scripts/93_robustness.py r1-train --family $fam --seed $s &
  pids+=($!); g=$((g + 1))
done; done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
if [ $rc = 0 ]; then
  run r1_eval_val 1 $P scripts/93_robustness.py r1-eval --split val &
  a=$!
  run r1_eval_test 2 $P scripts/93_robustness.py r1-eval --split test &
  b=$!
  wait $a || rc=1; wait $b || rc=1
fi
wait $R234 || rc=1
[ $rc = 0 ] && log "robustness GPU work done" || log "robustness FAILED (see steps.tsv)"
