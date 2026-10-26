#!/usr/bin/env bash
# Start the complex-split secondary runs on GPUs 0-2 once (a) H2 validation has fixed the
# planning source and (b) the S12 secondary runner has launched all of its jobs (author 10-25:
# "parallelize whatever you can"). The orchestrator's own secondary_complex step then finds them
# running (live locks) or done, and skips them.
cd ~/GeneWorld
P=~/venv/bin/python
log() { echo "[$(date -u "+%m-%d %H:%M:%S")] $*" | tee -a logs/a2_val/pipeline.log; }
while true; do
  N=$($P - <<'PY'
import importlib.util as u, json, os
from pathlib import Path
s = u.spec_from_file_location("r", "scripts/run_a2_jobs.py"); m = u.module_from_spec(s); s.loader.exec_module(m)
a2 = m.a2
names = [a2.run_name(j["norm"], j["backbone"], j["family"], j["config"], j["seed"], j["gene_split"])
         for j in m.jobs("secondary") if j["gene_split"] == "gene"]
started = sum((a2.RES_DIR / "train" / f"{n}.json").exists() or (a2.CK_DIR / f"{n}.lock").exists()
              for n in names)
h2 = os.path.exists("results/a2/h2/h2_val_plate_vjepa2.json")
print(f"{started} {len(names)} {int(h2)}")
PY
)
  set -- $N
  if [ "$3" = 1 ] && [ "$1" = "$2" ]; then break; fi
  sleep 60
done
log "complex-split secondary runs started early on GPUs 0,1,2 (secondary runner has launched all $2 jobs; planning source fixed)"
CUDA_VISIBLE_DEVICES=0,1,2 $P scripts/run_a2_jobs.py --phase secondary --only-split complex >> logs/a2_val/complex_early.log 2>&1
log "complex-split secondary runs (early) finished: $(tail -1 logs/a2_val/complex_early.log)"
