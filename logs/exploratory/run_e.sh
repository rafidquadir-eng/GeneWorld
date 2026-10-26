#!/usr/bin/env bash
# v3 S15 driver: E1, E2, E3 (exploratory, after the test). Started 10-26 by the author's
# instruction "run everything else". Phase 1 (preparation only, no evaluation) starts after the
# test embed commit and uses GPUs 2-7; phase 2 (every exploratory number) starts only after
# scripts/run_stage1_test.sh logs "test evaluation done". E3 embed changes the shard manifest,
# so it runs only after the test script has finished.
set -u
cd "$(dirname "$0")/../.."
P=~/venv/bin/python
L=logs/exploratory
T=logs/stage1_test/pipeline.log
log() { echo "[$(date -u "+%m-%d %H:%M:%S")] $*" | tee -a $L/pipeline.log; }
step() {
  local name=$1; shift
  local t0=$(date +%s) rc=0
  "$@" > $L/$name.log 2>&1 || rc=$?
  printf "%s\t%s\t%s\n" "$name" "$rc" "$(( $(date +%s) - t0 ))" >> $L/steps.tsv
  [ $rc = 0 ] || { log "step $name FAILED (exit $rc); see $L/$name.log"; exit 1; }
}
waitfor() { until grep -qE "$1" $T; do
  grep -q "FAILED" $T && { log "test script failed; stopping"; exit 1; }; sleep 60; done; }
[ -e $L/steps.tsv ] || printf "step\texit_code\tseconds\n" > $L/steps.tsv

log "phase 1 waits for the test embed commit"
waitfor "embed commit .* \(test\); allowed-paths check passed"
log "phase 1: E1 cache (validation base) and 6 E1 models on GPUs 2-7; E3 candidates train/val"
step e1_cache_val $P scripts/80_e1_bscore.py cache --base vjepa2
pids=(); g=2
for fam in h1_learned h1_none; do for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$g $P scripts/80_e1_bscore.py train --family $fam --seed $s \
      > $L/e1_train_${fam}_s$s.log 2>&1 & pids+=($!); g=$((g + 1))
done; done
step e3_candidates_train $P scripts/83_e3_partial.py candidates --split train
step e3_candidates_val $P scripts/83_e3_partial.py candidates --split val
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
printf "e1_train_all\t%s\t-\n" $rc >> $L/steps.tsv
[ $rc = 0 ] || { log "an E1 training run FAILED; see $L/e1_train_*.log"; exit 1; }
log "phase 1 done; waiting for the test evaluation"

waitfor "test evaluation done"
FREEZE=$(grep -oP "test: freeze \K\w+" $T | tail -1)
log "phase 2 (after the test; freeze $FREEZE)"
export CUDA_VISIBLE_DEVICES=0
step e1_cache_test $P scripts/80_e1_bscore.py cache --base vjepa2_test
step e1_probes_val $P scripts/80_e1_bscore.py probes --base vjepa2
step e1_probes_test $P scripts/80_e1_bscore.py probes --base vjepa2_test
step e1_eval_val $P scripts/80_e1_bscore.py eval --split val
step e1_eval_test $P scripts/80_e1_bscore.py eval --split test --base vjepa2_test \
     --freeze-commit $FREEZE
step e2 $P scripts/82_e2_forecast_probe.py
for s in test screenB; do
  step e3_candidates_$s $P scripts/83_e3_partial.py candidates --split $s
done
for s in train val test; do
  step e3_embed_$s $P scripts/83_e3_partial.py embed --split $s
done
step e3_cache_val $P scripts/83_e3_partial.py cache --base vjepa2
step e3_cache_test $P scripts/83_e3_partial.py cache --base vjepa2_test
step e3_eval_val $P scripts/83_e3_partial.py eval --split val --base vjepa2
step e3_eval_test $P scripts/83_e3_partial.py eval --split test --base vjepa2_test \
     --freeze-commit $FREEZE
step e3_compare $P scripts/83_e3_partial.py compare
log "exploratory done; GPU idle"
