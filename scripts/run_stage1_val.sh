#!/usr/bin/env bash
# Stage-1 validation-only pipeline (design A.10 gates on validation splits). Test splits are
# not touched here; they are evaluated once, after freezing, by run_stage1_test.sh.
set -u
cd "$(dirname "$0")/.."
P=~/venv/bin/python
L=logs/stage1_val
mkdir -p $L results/gates
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $L/pipeline.log; }

EMBED_PID=${1:-}
if [ -n "$EMBED_PID" ]; then
  log "waiting for embed process $EMBED_PID"
  while kill -0 "$EMBED_PID" 2>/dev/null; do sleep 60; done
fi

log "G-Data"
$P - <<'EOF' 2>&1 | tee -a $L/pipeline.log
import json, glob, numpy as np
n = sum(len(np.load(f)["wells"]) for f in glob.glob("data/latents/vjepa2/*.npz"))
plates = len(glob.glob("data/latents/vjepa2/*.npz"))
fails = json.load(open("logs/embed_failures.json")) if __import__("os").path.exists("logs/embed_failures.json") else []
r = {"rule": ">= 10,000 movies decoded and embedded", "movies_embedded": n, "plates": plates,
     "failures": len(fails), "pass": n >= 10000}
json.dump(r, open("results/gates/G_Data.json", "w"), indent=1); print(r)
EOF

log "static-clip control embeds"
$P scripts/20_stream_embed.py --static --wells splits/static_control_wells.tsv > $L/embed_static.log 2>&1

train() { $P scripts/40_train_predictor.py "$@" > $L/train_$2.log 2>&1; tail -1 $L/train_$2.log | tee -a $L/pipeline.log; }

log "H1 configuration selection (validation plates)"
for mode in learned none; do
  train --name h1_${mode}_A --mode $mode --width 512 --depth 6 --rollout 2
  train --name h1_${mode}_B --mode $mode --width 512 --depth 6 --rollout 6
  train --name h1_${mode}_C --mode $mode --width 768 --depth 12 --rollout 2
done
read -r BL BN CFG < <($P - <<'EOF'
import json
v = {n: json.load(open(f"results/train/{n}.json"))["best_val"] for m in ("learned", "none")
     for n in (f"h1_{m}_A", f"h1_{m}_B", f"h1_{m}_C")}
bl = min((n for n in v if "learned" in n), key=v.get)
bn = min((n for n in v if "none" in n), key=v.get)
json.dump({"val_scores": v, "selected_learned": bl, "selected_agnostic": bn},
          open("results/train/h1_selection.json", "w"), indent=1)
print(bl, bn, bl[-1])
EOF
)
log "selected $BL / $BN (config $CFG)"
case $CFG in A) W=512; D=6; R=2;; B) W=512; D=6; R=6;; C) W=768; D=12; R=2;; esac
$P scripts/41_eval_h1.py --model $BL --agnostic $BN --split val > $L/h1_val.log 2>&1
tail -8 $L/h1_val.log | tee -a $L/pipeline.log

log "H2 encoding sources (validation genes)"
MODELS=""
for s in perturbseq jump_crispr jump_orf string esm2; do
  train --name h2_${s} --mode encoding --source $s --width $W --depth $D --rollout $R
  MODELS="$MODELS,$s=h2_$s"
done
$P scripts/42_eval_h2.py --models ${MODELS#,} --agnostic $BN --split val > $L/h2_val.log 2>&1
tail -8 $L/h2_val.log | tee -a $L/pipeline.log
SRC=$($P -c "import json;r=json.load(open('results/h2/h2_val_vjepa2.json'));print(r['G_ZeroShot']['selected_source'] or min(r['sources'],key=lambda s:r['sources'][s]['true_vs_shuffled']['mean']))")
log "planning source: $SRC"

log "H3 planning (validation genes)"
$P scripts/43_plan.py --source $SRC --model h2_$SRC --split val > $L/h3_val.log 2>&1
tail -4 $L/h3_val.log | tee -a $L/pipeline.log

log "alternative backbones (H1, validation plates)"
for bb in dinov2 tijepa1; do
  train --name h1_learned_${bb} --backbone $bb --mode learned --width $W --depth $D --rollout $R
  train --name h1_none_${bb} --backbone $bb --mode none --width $W --depth $D --rollout $R
  $P scripts/41_eval_h1.py --backbone $bb --model h1_learned_${bb} --agnostic h1_none_${bb} \
     --split val > $L/h1_val_${bb}.log 2>&1
  tail -8 $L/h1_val_${bb}.log | tee -a $L/pipeline.log
done

log "controls"
$P scripts/44_static_control.py --model $BL > $L/static.log 2>&1
for bb in dinov2 tijepa1; do $P scripts/44_static_control.py --backbone $bb >> $L/static.log 2>&1; done
$P scripts/45_leakage.py > $L/leakage.log 2>&1

log "complex-held-out split (H2/H3 robustness, source $SRC)"
train --name h2c_none --gene-split complex --mode none --width $W --depth $D --rollout $R
train --name h2c_${SRC} --gene-split complex --mode encoding --source $SRC --width $W --depth $D --rollout $R
$P scripts/42_eval_h2.py --gene-split complex --models $SRC=h2c_$SRC --agnostic h2c_none \
   --split val --tag _complex > $L/h2_val_complex.log 2>&1
tail -3 $L/h2_val_complex.log | tee -a $L/pipeline.log
$P scripts/43_plan.py --gene-split complex --source $SRC --model h2c_$SRC --split val \
   --tag _complex > $L/h3_val_complex.log 2>&1
tail -2 $L/h3_val_complex.log | tee -a $L/pipeline.log
log "validation pipeline done; GPU idle"
