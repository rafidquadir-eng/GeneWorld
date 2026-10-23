"""Completion plan v2, S2: confirm the frames missing from the stage-2 fetch (CPU only).

The failed wells are the stage-2 selection (`splits/stage2_wells.tsv`) minus the wells present
in the V-JEPA 2 stage-2 shards (`data/latents_s2/vjepa2/*.npz`); the embed log kept only the
first failing frame of each well. Every one of the 93 level-1 frames of every failed well is
re-requested (HEAD on the BIA S3 chunk) with 3 attempts and exponential back-off (1, 2, 4 s).
Writes `results/a2/missing_frames.json`: per failed well its plate, split, gene, siRNA, primary
flags, the phenotype classes its siRNA carries on training plates (screenA annotation rows on
training plates only), and the missing frame indices; wells whose missing frames are all >= 88
are flagged (F9: only clip 13, frames 77-92, uses them). Frames that are now available are
listed only; nothing is added to train or validation.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from geneworld.annotations import well_annotations  # noqa: E402
from geneworld.io.mitocheck import N_FRAMES, chunk_url  # noqa: E402

ATTEMPTS = 3
STREAMS = 64
F9_FIRST = 88
_local = threading.local()


def sid(x) -> str | None:
    """siRNA identifier as a string ('28431', not '28431.0'); None if missing."""
    if pd.isna(x):
        return None
    return str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)


def head(url: str) -> int:
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
    code = -1
    for att in range(ATTEMPTS):
        try:
            code = s.head(url, timeout=60).status_code
        except requests.RequestException:
            code = -1
        if code == 200:
            return code
        if att < ATTEMPTS - 1:
            time.sleep(2 ** att)
    return code


def main() -> int:
    t0 = time.time()
    sel = pd.read_csv(REPO / "splits" / "stage2_wells.tsv", sep="\t").rename(
        columns={"Gene Symbol": "gene", "siRNA Identifier": "sirna"})
    got = {(f.stem, str(w)) for f in (REPO / "data" / "latents_s2" / "vjepa2").glob("*.npz")
           for w in np.load(f)["wells"]}
    keys = list(zip(sel.Plate, sel.Well, strict=True))
    fail = sel[[k not in got for k in keys]].reset_index(drop=True)
    print(f"selected {len(sel)}, embedded {len(got & set(keys))}, "
          f"failed {len(fail)}", flush=True)

    urls = [(i, t, chunk_url(r.uuid, r.Well, 1, t)) for i, r in enumerate(fail.itertuples())
            for t in range(N_FRAMES)]
    with ThreadPoolExecutor(STREAMS) as ex:
        codes = list(ex.map(lambda u: head(u[2]), urls))
    status = np.full((len(fail), N_FRAMES), -1)
    for (i, t, _), c in zip(urls, codes, strict=True):
        status[i, t] = c

    # phenotype classes of each siRNA on training plates (screenA rows on training plates)
    ann = well_annotations("A").reset_index()
    psplit = pd.read_csv(REPO / "splits" / "plate_split.tsv", sep="\t")
    train_plates = set(psplit.loc[psplit.split == "train", "plate"])
    tr = ann[ann.Plate.isin(train_plates)]
    cls_of = tr.groupby(tr["siRNA Identifier"].map(sid)).classes.agg(
        lambda s: sorted(set().union(*s)))
    h1p = pd.read_csv(REPO / "splits" / "amend02_h1_primary_wells.tsv", sep="\t")
    h1p = set(zip(h1p.Plate, h1p.Well, strict=True))
    h2p = set(pd.read_csv(REPO / "splits" / "amend02_h2_primary_genes.tsv", sep="\t").gene)

    wells = []
    for i, r in enumerate(fail.itertuples()):
        miss = [int(t) for t in np.flatnonzero(status[i] == 404)]
        other = {int(t): int(status[i, t]) for t in np.flatnonzero(
            (status[i] != 200) & (status[i] != 404))}
        wells.append({
            "plate": r.Plate, "well": r.Well, "plate_split": r.plate_split,
            "gene": None if pd.isna(r.gene) else r.gene,
            "sirna": sid(r.sirna),
            "control": "" if pd.isna(r.control) else r.control, "gene_split": r.gene_split,
            "h1_primary_well": (r.Plate, r.Well) in h1p,
            "h2_primary_gene": (not pd.isna(r.gene)) and r.gene in h2p,
            "stage2_phenotypic_sirna": bool(r.phenotypic),
            "training_plate_classes": cls_of.get(sid(r.sirna), []),
            "missing_frames": miss, "other_status": other,
            "all_missing_ge_88": bool(miss) and min(miss) >= F9_FIRST,
            "recovered": not miss and not other})
    w = pd.DataFrame(wells)
    per_plate = sel.groupby("Plate").size()
    lost = w.groupby("plate").size()
    whole = sorted(p for p, n in lost.items() if n == per_plate[p])

    def counts(d: pd.DataFrame) -> dict:
        return {"wells": int(len(d)), "plates": int(d.plate.nunique()),
                "h1_primary_wells": int(d.h1_primary_well.sum()),
                "h2_primary_gene_wells": int(d.h2_primary_gene.sum()),
                "all_missing_ge_88": int(d.all_missing_ge_88.sum())}

    out = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rule": f"HEAD on every level-1 frame chunk, {ATTEMPTS} attempts, back-off 1/2/4 s",
        "wells_selected": int(len(sel)), "wells_failed": int(len(w)),
        "frames_requested": int(status.size),
        "frames_missing": int((status == 404).sum()),
        "frames_other_status": int(((status != 200) & (status != 404)).sum()),
        "wells_recovered": int(w.recovered.sum()),
        "whole_plates_missing": whole,
        "by_split": {s: counts(d) for s, d in w.groupby("plate_split")},
        "missing_frame_histogram": {int(k): int(v) for k, v in sorted(
            pd.Series([t for m in w.missing_frames for t in m]).value_counts().items())},
        "wells": wells, "minutes": (time.time() - t0) / 60}
    (REPO / "results" / "a2" / "missing_frames.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("wells",
                                                               "missing_frame_histogram")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
