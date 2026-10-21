#!/usr/bin/env bash
# Amendment 02, steps 3 (completion) to 6, unattended, validation only. Commits and pushes to
# the current (non-main) branch after each step. Test plates are not touched here.
set -u
cd "$(dirname "$0")/.."
P=~/venv/bin/python
L=logs/a2_val
mkdir -p $L results/a2
log() { echo "[$(date -u "+%m-%d %H:%M:%S")] $*" | tee -a $L/pipeline.log; }
commit() {
  BR=$(git rev-parse --abbrev-ref HEAD)
  if [ "$BR" = main ]; then log "refusing to commit on main"; exit 1; fi
  git add -A results/a2 reports manifests scripts src splits >/dev/null 2>&1
  git add -f logs/a2_val logs/embed_stage2.log >/dev/null 2>&1
  git commit -qm "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && log "committed: $1"
}

EMBED_PID=${1:-}
START=${2:-3}   # resume at step 3, 4 or 5 (completed steps are committed and not redone)
if [ -n "$EMBED_PID" ]; then
  log "waiting for stage-2 embed $EMBED_PID"
  while kill -0 "$EMBED_PID" 2>/dev/null; do sleep 120; done
fi

if [ "$START" -le 3 ]; then
log "step 3 completion: stage-2 data and backup audit"
$P - <<'EOF' 2>&1 | tee -a $L/pipeline.log
import json, glob, os, numpy as np, pandas as pd
from pathlib import Path
want = pd.read_csv("splits/stage2_wells.tsv", sep="\t")
got = {(Path(f).stem, str(w)) for f in glob.glob("data/latents_s2/vjepa2/*.npz")
       for w in np.load(f)["wells"]}
fails = json.load(open("logs/embed_failures_stage2_wells.json")) \
    if os.path.exists("logs/embed_failures_stage2_wells.json") else None
bad = []
for f in glob.glob("data/latents_s2/*/*.npz"):
    b = Path.home() / "LambdaFilesystem/geneworld_backups" / f
    if not b.exists() or b.stat().st_size != os.path.getsize(f):
        bad.append(f)
r = {"wells_selected": int(len(want)), "wells_embedded": int(len(got)),
     "plates_selected": int(want.Plate.nunique()),
     "plates_embedded": int(len({p for p, _ in got})),
     "failures_logged": None if fails is None else len(fails),
     "failure_examples": (fails or [])[:20],
     "shards": len(glob.glob("data/latents_s2/*/*.npz")), "shards_not_backed_up": bad}
json.dump(r, open("results/a2/stage2_data.json", "w"), indent=1)
print({k: v for k, v in r.items() if k != "failure_examples"})
EOF
commit "Amendment 02 step 3 complete: stage-2 latents embedded and backed up (results/a2/stage2_data.json)"
fi
if [ "$START" -le 4 ]; then

log "step 4: caches (per-plate normalisation and global) and plate-identity probe"
for bb in vjepa2 dinov2 tijepa1; do
  $P -c "import sys; sys.path.insert(0,'src'); from geneworld import cache; print(cache.build('$bb'))" \
     > $L/cache_$bb.log 2>&1 || { log "cache $bb FAILED"; exit 1; }
  cp data/cache/$bb/stats.json results/a2/cache_stats_$bb.json
done
$P scripts/53_plate_probe.py --split val --out results/a2/plate_probe.json > $L/plate_probe.log 2>&1
$P scripts/53_plate_probe.py --split train --out results/a2/plate_probe_train.json >> $L/plate_probe.log 2>&1
grep -v -i warn $L/plate_probe.log | tee -a $L/pipeline.log
commit "Amendment 02 step 4: per-plate negative-control normalisation; plate-identity probe re-run (results/a2/plate_probe*.json); global pipeline kept as sensitivity"
fi

log "step 5: grid (12 families x 8 configurations x 3 seeds)"
$P scripts/run_a2_jobs.py --phase grid --workers 3 >> $L/grid.log 2>&1
tail -1 $L/grid.log | tee -a $L/pipeline.log
$P scripts/60_select_a2.py > $L/selection.log 2>&1 || { log "selection FAILED"; cat $L/selection.log; exit 1; }
cat $L/selection.log | tee -a $L/pipeline.log
commit "Amendment 02 step 5: grid trained (3 seeds); selected configurations recorded (results/a2/selection.json) before any gate"

log "H2 validation (planning source)"
$P scripts/62_eval_h2_a2.py --split val > $L/h2_val.log 2>&1; tail -3 $L/h2_val.log | tee -a $L/pipeline.log
log "secondary training (global preprocessing, alternative backbones, complex split)"
$P scripts/run_a2_jobs.py --phase secondary --workers 3 >> $L/secondary.log 2>&1
tail -1 $L/secondary.log | tee -a $L/pipeline.log
log "validation evaluations"
$P scripts/61_eval_h1_a2.py --split val > $L/h1_val.log 2>&1; tail -1 $L/h1_val.log | tee -a $L/pipeline.log
$P scripts/63_plan_a2.py --split val > $L/h3_val.log 2>&1; tail -1 $L/h3_val.log | tee -a $L/pipeline.log
for n in global; do
  $P scripts/61_eval_h1_a2.py --split val --norm $n > $L/h1_val_$n.log 2>&1
  $P scripts/62_eval_h2_a2.py --split val --norm $n > $L/h2_val_$n.log 2>&1
  $P scripts/63_plan_a2.py --split val --norm $n > $L/h3_val_$n.log 2>&1
done
for bb in dinov2 tijepa1; do
  $P scripts/61_eval_h1_a2.py --split val --backbone $bb > $L/h1_val_$bb.log 2>&1
  tail -1 $L/h1_val_$bb.log | tee -a $L/pipeline.log
done
SRC=$($P -c "import json;print(json.load(open('results/a2/h2/h2_val_plate_vjepa2.json'))['planning_source'])")
$P scripts/62_eval_h2_a2.py --split val --gene-split complex --sources $SRC > $L/h2_val_complex.log 2>&1
$P scripts/63_plan_a2.py --split val --gene-split complex > $L/h3_val_complex.log 2>&1
$P scripts/64_controls_a2.py --norm plate > $L/controls_plate.log 2>&1
$P scripts/64_controls_a2.py --norm global > $L/controls_global.log 2>&1
grep -l Traceback $L/*.log | tee -a $L/pipeline.log
commit "Amendment 02 step 5: validation evaluations (H1 with replicate baselines and phenotype probe CIs, H2, H3; global sensitivity; alternative backbones; complex split; controls)"

log "step 6: amended gates on validation"
$P scripts/65_gates_a2.py --split val > $L/gates.log 2>&1; cat $L/gates.log | tee -a $L/pipeline.log
commit "Amendment 02 step 6: G-Signal-2, G-ZeroShot-2, G-Plan-2 on validation recorded (results/a2/gates_val.json) before the freeze"
log "validation phase done; GPU idle"
