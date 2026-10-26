"""S18 rehearsal of the test path on data already used (completion plan v4).

  choose   seeded choice of 3 validation plates (one with stage-1 shards, two with stage-2
           shards only); prints them
  wells    --plates P1,P2,P3: embed lists of those plates' cached wells and of 100 seeded
           training wells (50 from stage-1 shards, 50 from stage-2-only shards), in
           logs/rehearsal_embed/
  tables   well tables for the rehearsal caches in logs/rehearsal/: the plates relabelled
           'test' (test path) and 'screenB' (Part C path), and stage-1 wells without the real
           test plates, so that no test well can enter a rehearsal cache
  compare  per backbone, every re-embedded well against its stored latents (each stored copy:
           stage-1 and stage-2 shards): byte identity, max |difference|, relative L2 per well
           and clip (raw space); in the control-normalised space (stored per-plate statistics
           of the validation cache), the mean squared difference over clips 0-12, tiles and
           dimensions as a fraction of the stage-1 H1 action-conditioned validation error
           (results/h1/h1_val_vjepa2.json, all horizons), and a systematic-shift test: per
           dimension a two-sided sign-flip test over wells of the mean difference, BH over
           dimensions.
           Criterion (v4, fixed before the run): byte-identical, or MSE <= 1% of that error and
           every BH q > 0.01.
  summary  results/a2/rehearsal/summary.json from the comparison, the embed records and the
           rehearsal evaluation logs; numbers of the rehearsal evaluation are not reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

R = REPO / "results" / "a2" / "rehearsal"
LOG = REPO / "logs" / "rehearsal_embed"
BACKBONES = ("vjepa2", "dinov2", "tijepa1")
TABLES = ["splits/stage1_wells.tsv", "splits/stage2_wells.tsv"]
REH = "data/latents_rehearsal"
MSE_FRACTION, Q_MIN = 0.01, 0.01


def well_tables() -> pd.DataFrame:
    t = pd.concat([pd.read_csv(REPO / f, sep="\t") for f in TABLES])
    t["Well"] = t.Well.astype(str)
    return t.drop_duplicates(["Plate", "Well"])


def stored(plate: str, bb: str) -> dict[str, Path]:
    return {s: REPO / s / bb / f"{plate}.npz" for s in ("data/latents", "data/latents_s2")
            if (REPO / s / bb / f"{plate}.npz").exists()}


def choose(seed: int) -> list[str]:
    meta = pd.read_csv(REPO / "data/cache/vjepa2/meta.tsv", sep="\t", low_memory=False)
    rng = np.random.default_rng(seed)
    val = sorted(meta.loc[meta.plate_split == "val", "plate"].unique())
    s1 = [p for p in val if "data/latents" in stored(p, "vjepa2")]
    s2 = [p for p in val if list(stored(p, "vjepa2")) == ["data/latents_s2"]]
    return [str(rng.choice(s1))] + [str(x) for x in rng.choice(s2, 2, replace=False)]


def sample_train(seed: int, n_train: int, plates: list[str]) -> dict:
    meta = pd.read_csv(REPO / "data/cache/vjepa2/meta.tsv", sep="\t", low_memory=False)
    rng = np.random.default_rng(seed + 1)
    tr = meta[meta.plate_split == "train"]
    s1w = {(p, str(w)) for p in set(tr.plate)
           if (REPO / f"data/latents/vjepa2/{p}.npz").exists()
           for w in np.load(REPO / f"data/latents/vjepa2/{p}.npz")["wells"]}
    src = np.array(["s1" if (p, str(w)) in s1w else "s2"
                    for p, w in zip(tr.plate, tr.well, strict=True)])
    pick = []
    for s in ("s1", "s2"):
        rows = np.flatnonzero(src == s)
        pick += list(rng.choice(rows, n_train // 2, replace=False))
    train = tr.iloc[sorted(pick)][["plate", "well"]]
    return {"train": list(map(list, train.values))}


def tables(plates: list[str]) -> None:
    out = REPO / "logs" / "rehearsal"
    out.mkdir(parents=True, exist_ok=True)
    reh = pd.read_csv(LOG / "rehearsal_plates.tsv", sep="\t")
    assert set(reh.Plate) == set(plates) and (reh.plate_split == "val").all()
    for split in ("test", "screenB"):
        reh.assign(plate_split=split).to_csv(out / f"rehearsal_{split}_wells.tsv", sep="\t",
                                            index=False)
    s1 = pd.read_csv(REPO / "splits/stage1_wells.tsv", sep="\t")
    s1[s1.plate_split != "test"].to_csv(out / "stage1_wells_no_test.tsv", sep="\t",
                                        index=False)
    print(f"tables for {plates}: {len(reh)} wells; stage-1 test-plate rows removed: "
          f"{int((s1.plate_split == 'test').sum())}")


def write_lists(sel: dict) -> None:
    t = well_tables()
    meta = pd.read_csv(REPO / "data/cache/vjepa2/meta.tsv", sep="\t", low_memory=False)
    inplates = meta[meta.plate.isin(sel["plates"])][["plate", "well"]]
    keys = set(map(tuple, inplates.values))
    plates = t[[(p, w) in keys for p, w in zip(t.Plate, t.Well, strict=True)]]
    keys_t = set(map(tuple, sel["train"]))
    train = t[[(p, w) in keys_t for p, w in zip(t.Plate, t.Well, strict=True)]]
    LOG.mkdir(parents=True, exist_ok=True)
    plates.to_csv(LOG / "rehearsal_plates.tsv", sep="\t", index=False)
    train.to_csv(LOG / "rehearsal_train.tsv", sep="\t", index=False)
    print(f"{len(plates)} wells on plates {sel['plates']}; {len(train)} training wells")


def sign_flip_q(d: np.ndarray, n: int = 10_000, seed: int = 0) -> np.ndarray:
    """d (wells, dims): two-sided sign-flip p per dimension of the mean, then BH."""
    rng = np.random.default_rng(seed)
    obs = np.abs(d.mean(0))
    flips = rng.choice([-1.0, 1.0], size=(n, d.shape[0]))
    null = np.abs(flips @ d / d.shape[0])                      # (n, dims)
    p = (1 + (null >= obs).sum(0)) / (n + 1)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def compare() -> dict:
    s1 = json.loads((REPO / "results/h1/h1_val_vjepa2.json").read_text())
    level = s1["mean_error"]["action_conditioned"]["all_horizons"]
    res = {"error_level": {"value": level, "source": "results/h1/h1_val_vjepa2.json "
                           "mean_error.action_conditioned.all_horizons (stage 1)"},
           "criterion": f"byte-identical, or normalised-space MSE <= {MSE_FRACTION:.0%} of "
                        f"the error level and every BH q > {Q_MIN} (sign-flip over wells per "
                        "dimension)", "backbones": {}}
    for bb in BACKBONES:
        st = np.load(REPO / f"data/cache/{bb}/plate_stats.npz")
        mu = dict(zip(st["plates"], st["mu"], strict=True))
        sd = dict(zip(st["plates"], st["sigma"], strict=True))
        ident, maxabs, rel, dn = [], 0.0, [], []
        for f in sorted((REPO / REH).rglob(f"{bb}/*.npz")):
            new = np.load(f)
            for old_p in stored(f.stem, bb).values():
                old = np.load(old_p)
                idx = {w: i for i, w in enumerate(old["wells"].astype(str))}
                for i, w in enumerate(new["wells"].astype(str)):
                    if w not in idx:
                        continue
                    a = old["latents"][idx[w]]
                    b = new["latents"][i]
                    ident.append(bool(a.tobytes() == b.tobytes()))
                    a32, b32 = a.astype(np.float32), b.astype(np.float32)
                    maxabs = max(maxabs, float(np.abs(a32 - b32).max()))
                    num = np.linalg.norm((a32 - b32).reshape(14, -1), axis=1)
                    rel.append(num / np.maximum(np.linalg.norm(a32.reshape(14, -1), axis=1),
                                                1e-12))
                    if f.stem in mu:
                        ta, tb = a32.mean(2), b32.mean(2)                  # (14, 4, D)
                        dn.append(((tb - ta) / sd[f.stem][:, None])[:13])  # clips 0-12
        rel = np.array(rel)
        dn = np.array(dn)                                                  # (n, 13, 4, D)
        mse = float((dn ** 2).mean()) if len(dn) else None
        q = sign_flip_q(dn.mean((1, 2))) if len(dn) else np.array([1.0])
        byte = bool(all(ident)) if ident else False
        ok = byte or (mse is not None and mse <= MSE_FRACTION * level and q.min() > Q_MIN)
        res["backbones"][bb] = {
            "n_comparisons": len(ident), "byte_identical_fraction": float(np.mean(ident)),
            "all_byte_identical": byte, "max_abs_difference_raw": maxabs,
            "relative_l2_raw": {"median": float(np.median(rel)), "max": float(rel.max()),
                                "per_clip_median": np.median(rel, 0).round(6).tolist()},
            "normalised_mse": mse, "normalised_mse_fraction_of_error_level":
                None if mse is None else mse / level,
            "normalised_mse_primary_clips_8_12": float((dn[:, [8, 12]] ** 2).mean())
                if len(dn) else None,
            "shift_test": {"n_dimensions": int(len(q)), "min_q_bh": float(q.min()),
                           "n_q_le_0.01": int((q <= Q_MIN).sum())},
            "pass": bool(ok)}
    res["pass"] = all(v["pass"] for v in res["backbones"].values())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["choose", "wells", "tables", "compare", "summary"])
    ap.add_argument("--plates", default="")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--n-train", type=int, default=100)
    args = ap.parse_args()
    R.mkdir(parents=True, exist_ok=True)
    plates = [p for p in args.plates.split(",") if p]
    if args.cmd == "choose":
        print(",".join(choose(args.seed)))
        return 0
    if args.cmd == "tables":
        tables(plates)
        return 0
    if args.cmd == "wells":
        ps = pd.read_csv(REPO / "splits/plate_split.tsv", sep="\t").set_index("plate").split
        if not plates or any(ps.get(p) != "val" for p in plates):
            sys.exit(f"--plates must be validation plates: {plates}")
        sel = {"plates": plates, **sample_train(args.seed, args.n_train, plates)}
        (R / "selection_of_wells.json").write_text(json.dumps({"seed": args.seed, **sel},
                                                              indent=1))
        write_lists(sel)
        return 0
    if args.cmd == "compare":
        res = compare()
        (R / "compare.json").write_text(json.dumps(res, indent=1))
        for bb, v in res["backbones"].items():
            print(bb, "pass" if v["pass"] else "FAIL", {k: v[k] for k in (
                "all_byte_identical", "normalised_mse_fraction_of_error_level")},
                "min q", v["shift_test"]["min_q_bh"])
        return 0 if res["pass"] else 1
    out = {"label": "rehearsal on validation and training data already used; not results",
           "selection": json.loads((R / "selection_of_wells.json").read_text()),
           "compare": json.loads((R / "compare.json").read_text())}
    for tag in ("rehearsal_embed_plates", "rehearsal_embed_train"):
        f = REPO / "logs" / tag / "embed_record.json"
        if f.exists():
            r = json.loads(f.read_text())
            out[tag] = {"gpus": {g: {k: v[k] for k in ("plates", "n_wells", "exit_code",
                                                         "minutes")}
                                 for g, v in r["gpus"].items()},
                        "streams_total": r["streams_total"], "n_failures": r["n_failures"],
                        "problems": r["problems"]}
    steps = REPO / "logs" / "rehearsal" / "steps.tsv"
    if steps.exists():
        st = pd.read_csv(steps, sep="\t")
        out["evaluation_steps"] = st.to_dict("records")
        out["evaluation_all_ok"] = bool((st.exit_code == 0).all())
        # a step rerun after a failed attempt counts once, by its last run; the failed
        # attempts stay listed (the strict flag above is kept as recorded)
        out["evaluation_last_run_ok"] = bool((st.groupby("step").exit_code.last() == 0).all())
        out["evaluation_failed_attempts"] = st[st.exit_code != 0].step.tolist()
    (R / "summary.json").write_text(json.dumps(out, indent=1))
    print("written results/a2/rehearsal/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
