#!/usr/bin/env bash
# S15 resume after the E2 failure of 22:28 UTC 10-26 (read-only mask in
# explore.within_series_probe, fixed); runs the steps of run_e.sh from e2 onward.
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
FREEZE=$(grep -oP "test: freeze \K\w+" $T | tail -1)
export CUDA_VISIBLE_DEVICES=0
log "resume from e2 (freeze $FREEZE)"
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
