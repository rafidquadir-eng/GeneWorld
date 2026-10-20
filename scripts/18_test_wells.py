"""Test-plate wells to embed at the freeze (amendment 02 section 8).

The frozen same-wells layout's test-plate wells (`splits/stage2_test_wells.tsv`) minus those
already embedded in stage 1 (`data/latents/vjepa2`). Written before the freeze; embedded only
after it, by `scripts/run_stage1_test.sh`.
"""
from __future__ import annotations

import glob
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    t = pd.read_csv(REPO / "splits" / "stage2_test_wells.tsv", sep="\t")
    done = {(Path(f).stem, str(w)) for f in glob.glob(str(REPO / "data/latents/vjepa2/*.npz"))
            for w in np.load(f)["wells"]}
    todo = t[[(p, w) not in done for p, w in zip(t.Plate, t.Well, strict=True)]]
    f = REPO / "splits" / "test_wells_to_embed.tsv"
    todo.to_csv(f, sep="\t", index=False)
    print(f"{len(t)} test-layout wells, {len(t) - len(todo)} already embedded (stage 1), "
          f"{len(todo)} to embed on {todo.Plate.nunique()} plates; sha256 "
          f"{hashlib.sha256(f.read_bytes()).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
