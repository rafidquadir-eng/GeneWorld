#!/usr/bin/env bash
# Resume of the S18 rehearsal after step h2_global stopped at 10-26 01:59 because the global
# secondary models were still training (S18 ran in parallel with S12, author 10-25). Runs the
# remaining commands of scripts/run_stage1_test.sh --rehearsal unchanged, once their models exist.
set -eu
cd ~/GeneWorld
export GENEWORLD_REHEARSAL=1 CUDA_VISIBLE_DEVICES=2
P=~/venv/bin/python
L=logs/rehearsal
R=results/a2/rehearsal
log() { echo "[$(date -u "+%m-%d %H:%M:%S")] $*" | tee -a $L/pipeline.log; }
step() {
  local name=$1; shift
  local t0=$(date +%s) rc=0
  "$@" > $L/$name.log 2>&1 || rc=$?
  printf "%s\t%s\t%s\n" "$name" "$rc" "$(( $(date +%s) - t0 ))" >> $L/steps.tsv
  [ $rc = 0 ] || { log "step $name FAILED (exit $rc); see $L/$name.log"; exit 1; }
}
need() {   # wait until every secondary results file of the given gene split exists
  until $P - "$1" <<'PY'
import importlib.util as u, sys
s = u.spec_from_file_location("r", "scripts/run_a2_jobs.py"); m = u.module_from_spec(s); s.loader.exec_module(m)
import os; os.environ.pop("GENEWORLD_REHEARSAL", None)
from pathlib import Path
names = [m.a2.run_name(j["norm"], j["backbone"], j["family"], j["config"], j["seed"], j["gene_split"])
         for j in m.jobs("secondary") if j["gene_split"] == sys.argv[1]]
sys.exit(0 if names and all(Path(f"results/a2/train/{n}.json").exists() for n in names) else 1)
PY
  do sleep 60; done
}
PLATES=LT0076_16,LT0049_42,LT0098_13
FREEZE=e7aa35e
REH="rehearsal-$FREEZE"
T="--split test --suffix _rehearsal --freeze-commit $REH"
V="--split val --suffix _rehearsal"
log "resume: waiting for the global and backbone secondary models"
need gene
log "resume: global and backbone models present"
step h2_global $P scripts/62_eval_h2_a2.py $V --norm global
step h3_global $P scripts/63_plan_a2.py $V --norm global
for bb in dinov2 tijepa1; do
  step h1_$bb $P scripts/61_eval_h1_a2.py $T --backbone $bb
done
step controls_plate $P scripts/64_controls_a2.py --split test --suffix _rehearsal --norm plate --freeze-commit $REH
step controls_global $P scripts/64_controls_a2.py --split test --suffix _rehearsal --norm global --freeze-commit $REH
SRC=$($P -c "import json;print(json.load(open('$R/h2/h2_val_plate_vjepa2.json'))['planning_source'])")
cp $R/h2/h2_val_plate_vjepa2.json $R/h2/h2_test_plate_vjepa2.json
cp $R/h3/h3_val_plate_vjepa2_$SRC.json $R/h3/h3_test_plate_vjepa2_$SRC.json
step gates $P scripts/65_gates_a2.py --split test
EXCL="{f'data/{r}/{b}/{p}.npz' for r in ('latents','latents_s2') for b in ('vjepa2','dinov2','tijepa1') for p in '$PLATES'.split(',')}"
step cache_screenB $P -c "import sys; sys.path.insert(0,'src'); from geneworld import cache
print(cache.build('vjepa2', roots=['data/latents','data/latents_s2','data/latents_rehearsal/plates'], tsvs=['logs/rehearsal/rehearsal_screenB_wells.tsv','logs/rehearsal/stage1_wells_no_test.tsv','splits/stage2_wells.tsv'], include_test=False, suffix='_rehearsal_screenB', exclude=$EXCL))"
step h1_screenB $P scripts/61_eval_h1_a2.py --split screenB --norm global --suffix _rehearsal_screenB --freeze-commit $REH
log "resume: waiting for the complex-split models"
need complex
step h2_complex $P scripts/62_eval_h2_a2.py $V --gene-split complex --sources $SRC
step h3_complex $P scripts/63_plan_a2.py $V --gene-split complex
step summary $P scripts/77_rehearsal.py summary
log "rehearsal done (base $FREEZE; resumed after h2_global)"
